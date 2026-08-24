"""Benchmark-relative performance metrics.

How a strategy does *against a benchmark*, not just in absolute terms: the active
return per unit of active risk (information ratio) and the volatility of that
active return (tracking error).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import TRADING_DAYS_PER_YEAR


def _active(returns: pd.Series, benchmark: pd.Series) -> pd.Series:
    return (returns - benchmark).dropna()


def tracking_error(returns: pd.Series, benchmark: pd.Series,
                   periods: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualised volatility of the return difference to the benchmark."""
    active = _active(returns, benchmark)
    if len(active) < 2:
        return float("nan")
    return float(active.std(ddof=1) * np.sqrt(periods))


def information_ratio(returns: pd.Series, benchmark: pd.Series,
                      periods: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualised active return over tracking error.

    Measures skill relative to the benchmark: consistent outperformance with low
    active risk scores high. Undefined (nan) when the strategy tracks the benchmark
    exactly (zero active risk)."""
    active = _active(returns, benchmark)
    if len(active) < 2:
        return float("nan")
    te = active.std(ddof=1)
    if te == 0:
        return float("nan")
    return float((active.mean() * periods) / (te * np.sqrt(periods)))


def capture_ratios(returns: pd.Series, benchmark: pd.Series):
    """(up_capture, down_capture): mean strategy return over mean benchmark return
    on the benchmark's up days and down days respectively.

    An up capture above 1 means the strategy beats the benchmark when it rises; a
    down capture below 1 means it loses less when the benchmark falls. A defensive
    strategy wants a high up capture and a low down capture."""
    joined = pd.concat([returns.rename("r"), benchmark.rename("b")], axis=1).dropna()
    r, b = joined["r"], joined["b"]
    up, down = b > 0, b < 0

    def ratio(mask):
        if not mask.any() or b[mask].mean() == 0:
            return float("nan")
        return float(r[mask].mean() / b[mask].mean())

    return ratio(up), ratio(down)
