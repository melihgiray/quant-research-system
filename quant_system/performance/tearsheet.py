"""Tearsheet: one call turns a return stream into a readable report + plots.

``format_tearsheet`` produces the terminal summary; ``save_report_plots`` writes
the figures the CLI drops into ``reports/``. Plotting uses the non-interactive
Agg backend so it works headless (CI, SSH, cron) with no display.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless: never tries to open a window
import matplotlib.pyplot as plt

from ..config import TRADING_DAYS_PER_YEAR
from .analytics import compute_metrics
from .significance import significance_summary
from ..risk.metrics import (
    drawdown_series, historical_var, conditional_var, parametric_var,
)


def _fmt_pct(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.2%}"


def _fmt_num(x: float, nd: int = 2) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def format_tearsheet(
    returns: pd.Series,
    turnover: Optional[pd.Series] = None,
    rf_annual: float = 0.0,
    title: str = "STRATEGY",
    extra_lines: Optional[List[str]] = None,
    periods: int = TRADING_DAYS_PER_YEAR,
    n_trials: Optional[int] = None,
) -> str:
    """Render a full performance tearsheet as a fixed-width text block.

    Parameters
    ----------
    returns : pd.Series
        The (out-of-sample) net return stream to summarise.
    turnover : pd.Series, optional
        Daily one-way turnover, for the annualised-turnover line.
    rf_annual : float
        Annual risk-free rate used in Sharpe/Sortino.
    title : str
        Heading shown at the top of the sheet.
    extra_lines : list[str], optional
        Extra rows appended verbatim (e.g. the factor-decomposition summary).
    n_trials : int, optional
        If set, append a PSR/DSR/minTRL significance block treating this as the
        number of strategy/parameter configurations searched.
    """
    m = compute_metrics(returns, turnover=turnover, rf_annual=rf_annual, periods=periods)
    r = returns.dropna()
    span = f"{r.index.min().date()} -> {r.index.max().date()}" if not r.empty else "(empty)"
    years = m["n_periods"] / periods if m["n_periods"] else 0.0

    width = 56
    line = "=" * width
    rows = [
        line,
        f" {title} - PERFORMANCE TEARSHEET".ljust(width),
        line,
        f" Period               {span}",
        f" Observations         {m['n_periods']}  (~{years:.1f} yrs)",
        f" Risk-free (ann.)     {rf_annual:.2%}",
        "-" * width,
        f" Annualised return    {_fmt_pct(m['ann_return'])}",
        f" Annualised vol       {_fmt_pct(m['ann_vol'])}",
        f" Sharpe ratio         {_fmt_num(m['sharpe'])}",
        f" Sortino ratio        {_fmt_num(m['sortino'])}",
        f" Calmar ratio         {_fmt_num(m['calmar'])}",
        "-" * width,
        f" Max drawdown         {_fmt_pct(m['max_drawdown'])}",
        f" Max DD duration      {m['max_dd_duration']} days",
        f" Hit rate (daily)     {_fmt_pct(m['hit_rate'])}",
        f" Avg win / Avg loss   {_fmt_pct(m['avg_win'])} / {_fmt_pct(m['avg_loss'])}",
        f" Profit factor        {_fmt_num(m['profit_factor'])}",
        f" Annualised turnover  {_fmt_num(m['ann_turnover'], 1)}x",
        "-" * width,
        f" 95% VaR (daily)      {_fmt_pct(historical_var(r, 0.95))}  "
        f"(Gaussian {_fmt_pct(parametric_var(r, 0.95))})",
        f" 95% CVaR (daily)     {_fmt_pct(conditional_var(r, 0.95))}",
    ]
    if n_trials is not None and not r.empty:
        rows.append("-" * width)
        rows.extend(f" {ln}" for ln in significance_summary(
            r, n_trials=n_trials, periods=periods, rf_annual=rf_annual))
    if extra_lines:
        rows.append("-" * width)
        rows.extend(f" {ln}" for ln in extra_lines)
    rows.append(line)
    return "\n".join(rows)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_equity_curve(returns: pd.Series, path: str, title: str = "Equity curve") -> str:
    """Cumulative growth of $1 on a log scale (so compounding reads linearly)."""
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(equity.index, equity.values, lw=1.4, color="#1f77b4")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_ylabel("Growth of $1 (log)")
    ax.grid(True, alpha=0.3)
    return _save(fig, path)


def plot_drawdown(returns: pd.Series, path: str, title: str = "Drawdown") -> str:
    """Underwater plot - shaded peak-to-trough declines over time."""
    dd = drawdown_series(returns) * 100.0
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.fill_between(dd.index, dd.values, 0.0, color="#d62728", alpha=0.4)
    ax.set_title(title)
    ax.set_ylabel("Drawdown (%)")
    ax.grid(True, alpha=0.3)
    return _save(fig, path)


def plot_regime_overlay(price: pd.Series, regimes: pd.Series, path: str,
                        title: str = "Regime overlay") -> str:
    """Price line with the high-vol/defensive regime shaded - eyeball the detector."""
    fig, ax = plt.subplots(figsize=(9, 4.0))
    ax.plot(price.index, price.values, lw=1.2, color="#222222", label="price")
    reg = regimes.reindex(price.index).ffill()
    # Shade contiguous spans where regime == 1 (defensive/high-vol).
    in_span, start = False, None
    for t, v in reg.items():
        if v == 1 and not in_span:
            in_span, start = True, t
        elif v != 1 and in_span:
            ax.axvspan(start, t, color="#ff7f0e", alpha=0.18)
            in_span = False
    if in_span:
        ax.axvspan(start, reg.index[-1], color="#ff7f0e", alpha=0.18)
    ax.set_title(f"{title} (shaded = defensive/high-vol)")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    return _save(fig, path)


def plot_factor_scatter(strat_excess: pd.Series, market_excess: pd.Series, path: str,
                        alpha_daily: float = 0.0, beta: float = 0.0,
                        title: str = "Strategy vs market") -> str:
    """Scatter of strategy vs market excess returns with the fitted CAPM line.

    The slope is market beta; the intercept (annualised) is alpha. A near-flat,
    low-slope cloud is the picture of a market-neutral alpha source.
    """
    s = strat_excess.align(market_excess, join="inner")
    x = s[1].values
    y = s[0].values
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.scatter(x, y, s=8, alpha=0.35, color="#1f77b4")
    xs = np.linspace(np.nanmin(x), np.nanmax(x), 50)
    ax.plot(xs, alpha_daily + beta * xs, color="#d62728", lw=1.6,
            label=f"β={beta:.2f}, α={alpha_daily*252:+.1%}/yr")
    ax.axhline(0, color="grey", lw=0.6)
    ax.axvline(0, color="grey", lw=0.6)
    ax.set_xlabel("Market excess return (daily)")
    ax.set_ylabel("Strategy excess return (daily)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    return _save(fig, path)


def plot_signal_distribution(values: pd.Series, path: str,
                             title: str = "Signal distribution") -> str:
    """Histogram of the raw signal (e.g. gross exposure or ML probabilities)."""
    v = pd.Series(values).dropna()
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.hist(v.values, bins=40, color="#2ca02c", alpha=0.75)
    ax.set_title(title)
    ax.set_xlabel("Signal value")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3)
    return _save(fig, path)


def save_report_plots(
    reports_dir: str,
    returns: pd.Series,
    price: Optional[pd.Series] = None,
    regimes: Optional[pd.Series] = None,
    strat_excess: Optional[pd.Series] = None,
    market_excess: Optional[pd.Series] = None,
    alpha_daily: float = 0.0,
    beta: float = 0.0,
    signal_values: Optional[pd.Series] = None,
    prefix: str = "strategy",
) -> List[str]:
    """Write every available plot to ``reports_dir`` and return the file paths.

    Plots whose inputs are not provided are silently skipped, so the same call
    works whether or not regime/factor data is available.
    """
    os.makedirs(reports_dir, exist_ok=True)
    saved: List[str] = []
    saved.append(plot_equity_curve(returns, os.path.join(reports_dir, f"{prefix}_equity.png")))
    saved.append(plot_drawdown(returns, os.path.join(reports_dir, f"{prefix}_drawdown.png")))
    if price is not None and regimes is not None:
        saved.append(plot_regime_overlay(price, regimes,
                                         os.path.join(reports_dir, f"{prefix}_regime.png")))
    if strat_excess is not None and market_excess is not None:
        saved.append(plot_factor_scatter(strat_excess, market_excess,
                                         os.path.join(reports_dir, f"{prefix}_factor.png"),
                                         alpha_daily=alpha_daily, beta=beta))
    if signal_values is not None:
        saved.append(plot_signal_distribution(signal_values,
                                              os.path.join(reports_dir, f"{prefix}_signal.png")))
    return saved
