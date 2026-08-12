"""A single self-contained HTML tearsheet.

``build_html_report`` turns a return stream into one HTML string with the
headline metrics, a per-year table, an equity-and-drawdown chart and a rolling
Sharpe (and rolling beta, if a benchmark is given). Every figure is embedded as a
base64 PNG, so the report is one file with no external assets: it opens the same
on any machine, offline, and can be committed straight into the repo.
"""

from __future__ import annotations

import base64
import io
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .analytics import compute_metrics
from .rolling import per_year_table, rolling_beta, rolling_sharpe
from ..risk.metrics import drawdown_series


def _fig_to_data_uri(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _pct(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.1%}"


def _num(x: float, nd: int = 2) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def _metrics_html(m: dict) -> str:
    rows = [
        ("Ann. return", _pct(m["ann_return"])),
        ("Ann. volatility", _pct(m["ann_vol"])),
        ("Sharpe", _num(m["sharpe"])),
        ("Sortino", _num(m["sortino"])),
        ("Calmar", _num(m["calmar"])),
        ("Max drawdown", _pct(m["max_drawdown"])),
        ("Hit rate", _pct(m["hit_rate"])),
        ("Ann. turnover", f"{_num(m['ann_turnover'], 1)}x"),
    ]
    cells = "".join(f"<div class='metric'><span>{k}</span><b>{v}</b></div>" for k, v in rows)
    return f"<section class='metrics'>{cells}</section>"


def _per_year_html(df: pd.DataFrame) -> str:
    head = "<tr><th>Year</th><th>Return</th><th>Vol</th><th>Sharpe</th><th>Max DD</th><th>Days</th></tr>"
    body = ""
    for year, row in df.iterrows():
        body += (f"<tr><td>{year}</td><td>{_pct(row['return'])}</td>"
                 f"<td>{_pct(row['vol'])}</td><td>{_num(row['sharpe'])}</td>"
                 f"<td>{_pct(row['max_drawdown'])}</td><td>{int(row['days'])}</td></tr>")
    return f"<table class='years'>{head}{body}</table>"


def _equity_drawdown_fig(returns: pd.Series):
    equity = (1.0 + returns).cumprod()
    dd = drawdown_series(returns)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(equity.index, equity.values, lw=1.3, color="#1f77b4")
    ax1.axhline(1.0, color="grey", lw=0.7)
    ax1.set_ylabel("Growth of $1")
    ax1.grid(True, alpha=0.3)
    ax2.fill_between(dd.index, dd.values, 0.0, color="#d62728", alpha=0.4)
    ax2.set_ylabel("Drawdown")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _rolling_fig(series: pd.Series, label: str, color: str, baseline: float = 0.0):
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(series.index, series.values, lw=1.2, color=color)
    ax.axhline(baseline, color="grey", lw=0.7)
    ax.set_ylabel(label)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def build_html_report(returns: pd.Series,
                      benchmark: Optional[pd.Series] = None,
                      turnover: Optional[pd.Series] = None,
                      title: str = "Strategy",
                      rf_annual: float = 0.0,
                      sharpe_window: int = 126) -> str:
    """Build a one-file HTML tearsheet and return it as a string."""
    returns = returns.dropna()
    metrics = compute_metrics(returns, turnover=turnover, rf_annual=rf_annual)
    span = f"{returns.index[0].date()} to {returns.index[-1].date()}"

    figures = [("Equity and drawdown", _fig_to_data_uri(_equity_drawdown_fig(returns)))]
    rs = rolling_sharpe(returns, window=sharpe_window, rf_annual=rf_annual)
    figures.append((f"Rolling Sharpe ({sharpe_window}d)",
                    _fig_to_data_uri(_rolling_fig(rs, "Sharpe", "#1f77b4"))))
    if benchmark is not None:
        rb = rolling_beta(returns, benchmark, window=sharpe_window)
        figures.append((f"Rolling beta ({sharpe_window}d)",
                        _fig_to_data_uri(_rolling_fig(rb, "Beta", "#2ca02c"))))

    charts = "".join(f"<figure><figcaption>{name}</figcaption>"
                     f"<img alt='{name}' src='{uri}'></figure>" for name, uri in figures)

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} tearsheet</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem auto;
         max-width: 900px; color: #1a1a1a; }}
  h1 {{ margin-bottom: 0.2rem; }}
  .span {{ color: #666; margin-top: 0; }}
  .metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.6rem;
             margin: 1.4rem 0; }}
  .metric {{ background: #f5f6f8; border-radius: 8px; padding: 0.7rem 0.9rem;
            display: flex; flex-direction: column; }}
  .metric span {{ color: #666; font-size: 0.8rem; }}
  .metric b {{ font-size: 1.25rem; }}
  table.years {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
  table.years th, table.years td {{ border-bottom: 1px solid #e2e4e8; padding: 0.4rem 0.6rem;
                                   text-align: right; }}
  table.years th:first-child, table.years td:first-child {{ text-align: left; }}
  figure {{ margin: 1.4rem 0; }}
  figcaption {{ color: #444; font-weight: 600; margin-bottom: 0.3rem; }}
  img {{ max-width: 100%; height: auto; }}
</style></head>
<body>
  <h1>{title}</h1>
  <p class="span">Out-of-sample, {span}, net of costs.</p>
  {_metrics_html(metrics)}
  <h2>By year</h2>
  {_per_year_html(per_year_table(returns, rf_annual=rf_annual))}
  {charts}
</body></html>"""
