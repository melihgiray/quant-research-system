"""Tail-risk and drawdown metrics.

VaR and CVaR are reported as *positive loss* numbers (a 95% VaR of 0.02 means
"on the worst 5% of days we expect to lose at least 2%"). CVaR (a.k.a. expected
shortfall) is the average loss *conditional on* breaching VaR - it is coherent
(sub-additive) where VaR is not, which is why post-2008 regulation moved to it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def historical_var(returns: pd.Series, level: float = 0.95) -> float:
    """Historical (non-parametric) Value at Risk, as a positive loss fraction.

    Parameters
    ----------
    returns : pd.Series
        Periodic returns.
    level : float
        Confidence level, e.g. 0.95 for 95% VaR.
    """
    r = returns.dropna()
    if r.empty:
        return float("nan")
    q = np.quantile(r.values, 1.0 - level)
    return float(-q)


def conditional_var(returns: pd.Series, level: float = 0.95) -> float:
    """Conditional VaR / expected shortfall: mean loss beyond the VaR threshold."""
    r = returns.dropna()
    if r.empty:
        return float("nan")
    threshold = np.quantile(r.values, 1.0 - level)
    tail = r.values[r.values <= threshold]
    if tail.size == 0:
        return float(-threshold)
    return float(-tail.mean())


def parametric_var(returns: pd.Series, level: float = 0.95) -> float:
    """Gaussian (variance-covariance) VaR. Assumes normality - reported alongside
    the historical figure precisely so the gap reveals fat tails."""
    r = returns.dropna()
    if r.empty:
        return float("nan")
    mu, sigma = r.mean(), r.std()
    z = stats.norm.ppf(1.0 - level)
    return float(-(mu + z * sigma))


def cornish_fisher_var(returns: pd.Series, level: float = 0.95) -> float:
    """Cornish-Fisher (modified) VaR: the Gaussian quantile corrected for skew and
    fat tails.

    Parametric VaR assumes normality and so understates the loss when returns are
    left-skewed or heavy-tailed. The Cornish-Fisher expansion adjusts the normal
    z-quantile using the sample skewness and excess kurtosis, capturing the shape
    of the tail without fitting a specific distribution. Reported as a positive
    loss fraction.
    """
    r = returns.dropna()
    if len(r) < 3:
        return float("nan")
    mu, sigma = r.mean(), r.std()
    s = stats.skew(r.values)
    k = stats.kurtosis(r.values, fisher=True)        # excess kurtosis (0 for normal)
    z = stats.norm.ppf(1.0 - level)
    z_cf = (z
            + (z ** 2 - 1) / 6 * s
            + (z ** 3 - 3 * z) / 24 * k
            - (2 * z ** 3 - 5 * z) / 36 * s ** 2)
    return float(-(mu + z_cf * sigma))


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Running drawdown (<= 0) of the cumulative equity curve from its peak."""
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    peak = equity.cummax()
    return equity / peak - 1.0


def max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough decline as a negative fraction."""
    dd = drawdown_series(returns)
    return float(dd.min()) if len(dd) else float("nan")


def max_drawdown_duration(returns: pd.Series) -> int:
    """Longest stretch (in periods) spent below a previous equity peak.

    This is the "time under water" - often more painful to live through than the
    depth itself, so we report it explicitly.
    """
    dd = drawdown_series(returns)
    if dd.empty:
        return 0
    underwater = dd < 0
    longest = current = 0
    for flag in underwater.values:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return int(longest)
