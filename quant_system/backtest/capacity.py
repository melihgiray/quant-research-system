"""Transaction-cost sensitivity and strategy capacity.

The square-root impact model (`costs.py`) makes cost grow with capital, so the
same alpha is worth less at scale. "Capacity" is the AUM at which trading frictions
eat enough of the edge that the strategy stops being worth running. Every PM asks
this; here it falls straight out of sweeping the capital input and watching the
net Sharpe decay.

We define capacity as the capital at which the *net* Sharpe falls to half of the
*frictionless* (zero-cost) Sharpe - a transparent, defensible convention.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..config import CostConfig, TRADING_DAYS_PER_YEAR
from .engine import run_backtest


def _sharpe(returns: pd.Series, periods: int) -> float:
    sd = returns.std()
    return float(np.sqrt(periods) * returns.mean() / sd) if sd > 0 else float("nan")


def sweep_capital(weights: pd.DataFrame, price_data, base_cost: CostConfig,
                  capitals: Sequence[float],
                  periods: int = TRADING_DAYS_PER_YEAR) -> pd.DataFrame:
    """Run the backtest across a range of capital levels.

    Returns a DataFrame indexed by capital with the net annualised return, net
    Sharpe, and the annualised cost drag (gross minus net return).
    """
    gross = run_backtest(weights, price_data, cost=None).returns
    gross_ann = float((1 + gross).prod() ** (periods / len(gross)) - 1)
    rows = []
    for cap in capitals:
        cost = CostConfig(half_spread_bps=base_cost.half_spread_bps,
                          impact_eta=base_cost.impact_eta, capital=cap,
                          adv_lookback=base_cost.adv_lookback,
                          vol_lookback=base_cost.vol_lookback,
                          default_daily_vol=base_cost.default_daily_vol)
        res = run_backtest(weights, price_data, cost=cost)
        net_ann = float((1 + res.returns).prod() ** (periods / len(res.returns)) - 1)
        rows.append({
            "capital": cap,
            "net_ann_return": net_ann,
            "net_sharpe": _sharpe(res.returns, periods),
            "cost_drag_ann": gross_ann - net_ann,
            "ann_cost": float(res.costs.sum() * periods / len(res.costs)),
        })
    df = pd.DataFrame(rows).set_index("capital")
    df.attrs["gross_sharpe"] = _sharpe(gross, periods)
    return df


def estimate_capacity(sweep: pd.DataFrame, half_of_frictionless: bool = True) -> dict:
    """Estimate the capital where net Sharpe falls to half the frictionless Sharpe.

    Linear interpolation in log-capital between the bracketing points. Returns the
    capacity (or a sentinel) plus the frictionless and threshold Sharpe.
    """
    gross_sharpe = sweep.attrs.get("gross_sharpe", float("nan"))
    if not np.isfinite(gross_sharpe) or gross_sharpe <= 0:
        return {"capacity": float("nan"), "gross_sharpe": gross_sharpe,
                "threshold": float("nan"), "note": "frictionless Sharpe <= 0: no capacity"}
    threshold = 0.5 * gross_sharpe if half_of_frictionless else gross_sharpe
    caps = sweep.index.to_numpy(float)
    sr = sweep["net_sharpe"].to_numpy(float)
    below = np.where(sr < threshold)[0]
    if below.size == 0:
        return {"capacity": float("inf"), "gross_sharpe": gross_sharpe,
                "threshold": threshold, "note": "Sharpe holds above threshold across the swept range"}
    i = below[0]
    if i == 0:
        return {"capacity": float(caps[0]), "gross_sharpe": gross_sharpe,
                "threshold": threshold, "note": "already below threshold at smallest capital"}
    # Interpolate in log-capital between i-1 (above) and i (below).
    x0, x1 = np.log10(caps[i - 1]), np.log10(caps[i])
    y0, y1 = sr[i - 1], sr[i]
    frac = (threshold - y0) / (y1 - y0) if y1 != y0 else 0.0
    cap = 10 ** (x0 + frac * (x1 - x0))
    return {"capacity": float(cap), "gross_sharpe": gross_sharpe,
            "threshold": threshold, "note": ""}


def plot_capacity(sweep: pd.DataFrame, path: str, capacity: Optional[float] = None,
                  title: str = "Capacity: net Sharpe vs capital") -> str:
    """Net Sharpe against capital (log x), with the capacity point marked."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(sweep.index, sweep["net_sharpe"], "o-", color="#1f77b4", label="net Sharpe")
    gross = sweep.attrs.get("gross_sharpe")
    if gross is not None and np.isfinite(gross):
        ax.axhline(gross, ls="--", color="grey", lw=1, label=f"frictionless ({gross:.2f})")
        ax.axhline(0.5 * gross, ls=":", color="#d62728", lw=1, label="half frictionless")
    if capacity is not None and np.isfinite(capacity):
        ax.axvline(capacity, color="#2ca02c", lw=1.2)
        ax.text(capacity, ax.get_ylim()[0], f"  capacity≈${capacity/1e6:.0f}M",
                color="#2ca02c", va="bottom", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Capital ($)")
    ax.set_ylabel("Annualised Sharpe (net)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def capacity_summary(sweep: pd.DataFrame, cap: dict) -> List[str]:
    """Tearsheet-ready lines for the capacity analysis."""
    def money(x):
        if not np.isfinite(x):
            return "n/a"
        return f">${sweep.index.max()/1e6:.0f}M" if x == float("inf") else f"${x/1e6:.0f}M"
    lines = ["CAPACITY / COST SENSITIVITY",
             f" Frictionless Sharpe   {cap['gross_sharpe']:.2f}",
             f" Capacity (half-Sharpe)   {money(cap['capacity'])}"]
    if cap.get("note"):
        lines.append(f" note: {cap['note']}")
    return lines
