"""Tail-risk and drawdown metrics.

VaR and CVaR are reported as *positive loss* numbers (a 95% VaR of 0.02 means
"on the worst 5% of days we expect to lose at least 2%"). CVaR (a.k.a. expected
shortfall) is the average loss *conditional on* breaching VaR - it is coherent
(sub-additive) where VaR is not, which is why post-2008 regulation moved to it.
"""

from __future__ import annotations

from dataclasses import dataclass

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


@dataclass
class EVTTail:
    """Extreme-value tail estimate from a peaks-over-threshold GPD fit.

    ``var`` and ``es`` are positive loss fractions at the requested level; ``xi``
    is the fitted GPD shape (tail index): xi > 0 is a genuinely heavy tail, xi ~ 0
    is exponential, xi < 0 is bounded. ``threshold`` is the loss cut-off and
    ``n_exceedances`` how many observations sat beyond it.
    """
    var: float
    es: float
    xi: float
    threshold: float
    n_exceedances: int


def evt_tail(returns: pd.Series,
             level: float = 0.99,
             threshold_quantile: float = 0.90) -> EVTTail:
    """Tail VaR and expected shortfall from extreme-value theory.

    Historical VaR at a deep quantile is estimated from only a handful of points
    and is noisy. Extreme-value theory instead fits a Generalized Pareto
    distribution to the exceedances over a high threshold (the peaks-over-
    threshold method), which is the limiting shape of tail exceedances, and reads
    the deep quantile off that fit. This extrapolates into the tail in a
    principled way rather than being capped by the worst observed loss.
    """
    if level <= threshold_quantile:
        raise ValueError("level must exceed threshold_quantile")
    r = returns.dropna()
    losses = -r.values
    if losses.size == 0:
        return EVTTail(float("nan"), float("nan"), float("nan"), float("nan"), 0)

    u = float(np.quantile(losses, threshold_quantile))
    exceed = losses[losses > u] - u
    n, nu = losses.size, exceed.size
    if nu < 10:
        return EVTTail(float("nan"), float("nan"), float("nan"), u, int(nu))

    xi, _, beta = stats.genpareto.fit(exceed, floc=0.0)
    ratio = (n / nu) * (1.0 - level)
    if abs(xi) < 1e-8:
        var = u + beta * (-np.log(ratio))
    else:
        var = u + (beta / xi) * (ratio ** (-xi) - 1.0)
    es = var / (1.0 - xi) + (beta - xi * u) / (1.0 - xi) if xi < 1 else float("inf")
    return EVTTail(float(var), float(es), float(xi), u, int(nu))


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Running drawdown (<= 0) of the cumulative equity curve from its peak."""
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    peak = equity.cummax()
    return equity / peak - 1.0


def max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough decline as a negative fraction."""
    dd = drawdown_series(returns)
    return float(dd.min()) if len(dd) else float("nan")


def ulcer_index(returns: pd.Series) -> float:
    """Root-mean-square drawdown: penalises both the depth and the duration of
    drawdowns, unlike max drawdown which sees only the single worst point."""
    dd = drawdown_series(returns)
    if dd.empty:
        return float("nan")
    return float(np.sqrt(np.mean(dd.to_numpy() ** 2)))


def pain_ratio(returns: pd.Series, periods: int = 252) -> float:
    """Annualised return divided by the ulcer index: a drawdown-aware return/risk
    ratio that rewards staying out of deep, long underwater stretches."""
    r = returns.dropna()
    ui = ulcer_index(returns)
    if r.empty or not np.isfinite(ui) or ui == 0:
        return float("nan")
    growth = float((1.0 + r).prod())
    if growth <= 0:
        return float("nan")
    annualised = growth ** (periods / len(r)) - 1.0
    return float(annualised / ui)


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
