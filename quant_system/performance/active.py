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


def alpha_beta(returns: pd.Series, benchmark: pd.Series,
               periods: int = TRADING_DAYS_PER_YEAR, rf_annual: float = 0.0):
    """(annualised alpha, beta) from a CAPM regression of returns on the benchmark.

    Beta is the sensitivity to the benchmark; alpha is the annualised return left
    over once that market exposure is priced out, the part not explained by simply
    holding beta of the benchmark."""
    joined = pd.concat([returns.rename("r"), benchmark.rename("b")], axis=1).dropna()
    if len(joined) < 2:
        return float("nan"), float("nan")
    r = joined["r"] - rf_annual / periods
    b = joined["b"] - rf_annual / periods
    var = b.var()
    if var == 0:
        return float("nan"), float("nan")
    beta = float(r.cov(b) / var)
    alpha = float((r.mean() - beta * b.mean()) * periods)
    return alpha, beta


def treynor_ratio(returns: pd.Series, benchmark: pd.Series,
                  periods: int = TRADING_DAYS_PER_YEAR, rf_annual: float = 0.0) -> float:
    """Annualised excess return per unit of market beta (Treynor).

    Like the Sharpe ratio but dividing by systematic risk (beta) instead of total
    volatility, so it rewards return earned above what the market exposure alone
    would give. A levered version of the same book has the same Treynor."""
    _, beta = alpha_beta(returns, benchmark, periods, rf_annual)
    if not np.isfinite(beta) or beta == 0:
        return float("nan")
    excess_annual = returns.dropna().mean() * periods - rf_annual
    return float(excess_annual / beta)
