"""Performance analytics - every headline metric, not just Sharpe.

A Sharpe ratio on its own hides a lot. It says nothing about tail risk, path
dependence, or how the returns were earned. So this module computes the rest of
the usual stats too, with a short note on each about what it adds beyond Sharpe.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..config import TRADING_DAYS_PER_YEAR
from ..risk.metrics import max_drawdown, max_drawdown_duration


def _excess(returns: pd.Series, rf_annual: float, periods: int) -> pd.Series:
    """Convert an annual risk-free rate to per-period and subtract it."""
    return returns - rf_annual / periods


def annualized_return(returns: pd.Series, periods: int = TRADING_DAYS_PER_YEAR) -> float:
    """Geometric (compound) annualised return - the rate you actually realise."""
    r = returns.dropna()
    if r.empty:
        return float("nan")
    growth = float((1.0 + r).prod())
    years = len(r) / periods
    if years <= 0 or growth <= 0:
        return float("nan")
    return growth ** (1.0 / years) - 1.0


def annualized_vol(returns: pd.Series, periods: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualised volatility (sqrt-time scaling of per-period std)."""
    r = returns.dropna()
    return float(r.std() * np.sqrt(periods)) if not r.empty else float("nan")


def sharpe_ratio(returns: pd.Series, rf_annual: float = 0.0,
                 periods: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualised Sharpe = mean excess return / volatility, scaled by sqrt(periods).

    Rewards return per unit of *total* volatility. Penalises upside the same as
    downside, which is its main weakness - hence we also report Sortino.
    """
    r = _excess(returns, rf_annual, periods).dropna()
    if r.empty or r.std() == 0:
        return float("nan")
    return float(np.sqrt(periods) * r.mean() / r.std())


def sortino_ratio(returns: pd.Series, rf_annual: float = 0.0,
                  periods: int = TRADING_DAYS_PER_YEAR) -> float:
    """Like Sharpe but divides by *downside* deviation only.

    Volatility from upside surprises is not risk. Sortino measures return per unit
    of harmful (below-target) deviation, so a strategy with big up-days and small
    down-days scores better here than under Sharpe.
    """
    r = _excess(returns, rf_annual, periods).dropna()
    downside = r[r < 0]
    if r.empty or downside.empty:
        return float("nan")
    dd = np.sqrt((downside ** 2).mean())
    if dd == 0:
        return float("nan")
    return float(np.sqrt(periods) * r.mean() / dd)


def calmar_ratio(returns: pd.Series, periods: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualised return / |max drawdown|.

    A path-aware ratio: it asks "how much do I earn per unit of the worst loss I
    had to stomach?" Favoured by managed-futures/CTA shops who care about MDD.
    """
    mdd = max_drawdown(returns)
    if mdd is None or np.isnan(mdd) or mdd == 0:
        return float("nan")
    return float(annualized_return(returns, periods) / abs(mdd))


def hit_rate(returns: pd.Series) -> float:
    """Fraction of periods with a positive return (the daily-level win rate)."""
    r = returns.dropna()
    nonzero = r[r != 0]
    return float((nonzero > 0).mean()) if not nonzero.empty else float("nan")


def win_loss_stats(returns: pd.Series) -> Dict[str, float]:
    """Average win, average loss, and profit factor.

    profit_factor = gross profit / gross loss. > 1 means winners outweigh losers
    in aggregate; it is the single number a discretionary trader instinctively
    reaches for, and it cross-checks the Sharpe (a high Sharpe with PF ~ 1 means
    you are winning on consistency, not on magnitude).
    """
    r = returns.dropna()
    wins, losses = r[r > 0], r[r < 0]
    gross_win = float(wins.sum())
    gross_loss = float(-losses.sum())
    return {
        "avg_win": float(wins.mean()) if not wins.empty else float("nan"),
        "avg_loss": float(losses.mean()) if not losses.empty else float("nan"),
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else float("inf"),
    }


def annualized_turnover(turnover: Optional[pd.Series],
                        periods: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualised one-way turnover (x of portfolio traded per year).

    High turnover is where transaction costs and capacity limits bite, so it is
    reported alongside returns: a great gross Sharpe at 50x turnover may be
    uninvestable once costs scale with size.
    """
    if turnover is None or len(turnover) == 0:
        return float("nan")
    return float(turnover.mean() * periods)


def compute_metrics(
    returns: pd.Series,
    turnover: Optional[pd.Series] = None,
    rf_annual: float = 0.0,
    periods: int = TRADING_DAYS_PER_YEAR,
) -> Dict[str, float]:
    """Compute the full metric panel as a flat dict (used by the tearsheet)."""
    r = returns.dropna()
    wl = win_loss_stats(r)
    return {
        "n_periods": int(len(r)),
        "ann_return": annualized_return(r, periods),
        "ann_vol": annualized_vol(r, periods),
        "sharpe": sharpe_ratio(r, rf_annual, periods),
        "sortino": sortino_ratio(r, rf_annual, periods),
        "calmar": calmar_ratio(r, periods),
        "max_drawdown": max_drawdown(r),
        "max_dd_duration": max_drawdown_duration(r),
        "hit_rate": hit_rate(r),
        "avg_win": wl["avg_win"],
        "avg_loss": wl["avg_loss"],
        "profit_factor": wl["profit_factor"],
        "ann_turnover": annualized_turnover(turnover, periods),
    }
