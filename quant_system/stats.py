"""Time-series statistics for diagnosing return behaviour.

Small, dependency-light estimators for questions the strategies care about: does a
series trend, mean-revert, or wander (Hurst); does it look like a random walk
(variance ratio); how much does it remember (autocorrelation).
"""

from __future__ import annotations

import numpy as np


def hurst_exponent(series, min_lag: int = 2, max_lag: int = 80) -> float:
    """Hurst exponent via the scaling of lagged-difference dispersion.

    For a series whose increments scale as ``lag ** H``, the standard deviation of
    ``series[t + lag] - series[t]`` grows like ``lag ** H``, so H is the slope of
    log-dispersion against log-lag. H ~ 0.5 is a random walk, H > 0.5 is
    persistent (trending), H < 0.5 is mean-reverting.
    """
    x = np.asarray(series, dtype=float)
    lags = np.arange(min_lag, max_lag)
    tau = np.array([np.std(x[lag:] - x[:-lag]) for lag in lags])
    good = tau > 0
    if good.sum() < 2:
        return float("nan")
    slope = np.polyfit(np.log(lags[good]), np.log(tau[good]), 1)[0]
    return float(slope)


def variance_ratio(series, q: int = 2) -> float:
    """Lo-MacKinlay variance ratio: per-period variance of q-step vs 1-step moves.

    Under a random walk the variance of q-period changes is q times the variance
    of 1-period changes, so the ratio is ~1. A ratio above 1 signals positive
    autocorrelation (trending), below 1 signals mean reversion.
    """
    x = np.asarray(series, dtype=float)
    if q < 2 or len(x) <= q:
        raise ValueError("need q >= 2 and more observations than q")
    var_1 = np.diff(x).var(ddof=1)
    var_q = (x[q:] - x[:-q]).var(ddof=1)
    if var_1 == 0:
        return float("nan")
    return float((var_q / q) / var_1)


def autocorrelation(series, lag: int = 1) -> float:
    """Sample autocorrelation at ``lag``: how much the series remembers itself.

    Near 0 for white noise; for an AR(1) with coefficient phi it is about phi at
    lag 1 and phi**k at lag k.
    """
    x = np.asarray(series, dtype=float)
    x = x - x.mean()
    n = len(x)
    if lag < 1 or lag >= n:
        raise ValueError("need 1 <= lag < len(series)")
    c0 = np.dot(x, x) / n
    if c0 == 0:
        return float("nan")
    c_lag = np.dot(x[:-lag], x[lag:]) / n
    return float(c_lag / c0)


def half_life(spread) -> float:
    """Half-life of mean reversion (in bars) from an Ornstein-Uhlenbeck fit.

    Regress the change in the spread on its lagged level, ``ds = a + b * s_{t-1}``;
    a mean-reverting spread has ``b < 0`` and reverts halfway to its mean in
    ``-ln(2) / b`` bars. This is what a pairs trade uses to size a holding period.
    Returns inf when the series does not mean-revert (``b >= 0``).
    """
    s = np.asarray(spread, dtype=float)
    if len(s) < 3:
        return float("nan")
    ds = np.diff(s)
    lagged = s[:-1]
    design = np.column_stack([np.ones_like(lagged), lagged])
    b = np.linalg.lstsq(design, ds, rcond=None)[0][1]
    if b >= 0:
        return float("inf")
    return float(-np.log(2) / b)


def _dickey_fuller_stat(y: np.ndarray) -> float:
    """t-statistic on the lagged level in a Dickey-Fuller regression of dy on y_{t-1}.

    A large positive value means the level pulls further away rather than
    reverting, i.e. explosive (bubble-like) behaviour."""
    lagged = y[:-1]
    dy = np.diff(y)
    design = np.column_stack([np.ones_like(lagged), lagged])
    dof = len(dy) - 2
    if dof <= 0:
        return float("nan")
    beta, *_ = np.linalg.lstsq(design, dy, rcond=None)
    resid = dy - design @ beta
    s2 = float(resid @ resid) / dof
    se = np.sqrt(s2 * np.linalg.inv(design.T @ design)[1, 1])
    return float(beta[1] / se) if se > 0 else float("nan")


def sadf(series, min_window: int = 40, stride: int = 3) -> np.ndarray:
    """Supremum Augmented Dickey-Fuller statistic for explosiveness (bubble) detection.

    For each end point, take the largest Dickey-Fuller statistic over all
    backward-expanding start points; a spike means the series is behaving
    explosively up to that point (Phillips-Shi-Yu / Lopez de Prado, ch. 17). Random
    walks stay low, bubbles push it sharply positive. ``stride`` subsamples the
    start points to keep it tractable. The first ``2 * min_window`` values are NaN.
    """
    x = np.asarray(series, dtype=float)
    n = len(x)
    out = np.full(n, np.nan)
    for t in range(min_window * 2, n):
        best = -np.inf
        for t0 in range(0, t - min_window, stride):
            stat = _dickey_fuller_stat(x[t0:t + 1])
            if np.isfinite(stat) and stat > best:
                best = stat
        if np.isfinite(best):
            out[t] = best
    return out


def jarque_bera(returns):
    """Jarque-Bera test of normality: returns (statistic, p-value).

    Combines skewness and excess kurtosis into a single statistic. A small p-value
    rejects normality, which for returns usually means fat tails or skew, exactly
    the shape that makes Gaussian VaR understate risk."""
    from scipy.stats import jarque_bera as _jb
    x = np.asarray(returns, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan"), float("nan")
    result = _jb(x)
    return float(result.statistic), float(result.pvalue)
