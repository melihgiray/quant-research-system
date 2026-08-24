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
