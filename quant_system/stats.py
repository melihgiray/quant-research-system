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
