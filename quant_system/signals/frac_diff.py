"""Fractional differentiation (Lopez de Prado, Advances in Financial ML, ch. 5).

A price series is non-stationary, which most models dislike; the usual fix,
first differencing (returns), makes it stationary but throws away almost all of
the memory in the level. Fractional differentiation takes a real-valued order
``d`` between 0 and 1: enough to pass a stationarity test, but no more, so the
transformed series keeps most of its correlation with the original level.

This uses the fixed-width-window (FFD) variant: the binomial weights are
truncated once they fall below a threshold, giving every observation the same
finite window rather than an expanding one.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def ffd_weights(d: float, threshold: float = 1e-4, max_lags: int = 100000) -> np.ndarray:
    """Fixed-width fractional-difference weights by lag, ``w[0]`` the current bar.

    ``w[k] = -w[k-1] * (d - k + 1) / k`` from ``w[0] = 1``, truncated once the
    magnitude drops below ``threshold``. For ``d = 1`` this is ``[1, -1]`` (a
    first difference); for ``d = 0`` it is ``[1]`` (the identity).
    """
    weights = [1.0]
    k = 1
    while k < max_lags:
        next_w = -weights[-1] * (d - k + 1) / k
        if abs(next_w) < threshold:
            break
        weights.append(next_w)
        k += 1
    return np.asarray(weights, dtype=float)


def frac_diff_ffd(series: pd.Series, d: float, threshold: float = 1e-4) -> pd.Series:
    """Fractionally difference a series at order ``d`` with a fixed-width window.

    The first ``len(weights) - 1`` observations are NaN (not enough history for a
    full window). The result has the same index as ``series``.
    """
    weights = ffd_weights(d, threshold)
    width = len(weights) - 1
    values = np.asarray(series, dtype=float)
    out = np.full(len(values), np.nan)
    flipped = weights[::-1]                       # align w[width] with the oldest bar
    for t in range(width, len(values)):
        out[t] = float(np.dot(flipped, values[t - width:t + 1]))
    return pd.Series(out, index=series.index)


def min_ffd_order(series: pd.Series,
                  d_grid: Optional[Sequence[float]] = None,
                  threshold: float = 1e-4,
                  adf_pmax: float = 0.05,
                  min_obs: int = 30) -> Optional[Tuple[float, float]]:
    """Smallest ``d`` on the grid whose FFD series is stationary (ADF p < ``adf_pmax``).

    Walks the grid from low to high and returns the first ``(d, adf_pvalue)`` that
    passes an augmented Dickey-Fuller test, i.e. the least differencing that
    achieves stationarity and so keeps the most memory. Orders whose window is
    wider than the data (too few points after warm-up) are skipped. Returns None
    if nothing on the grid stationarises the series.
    """
    from statsmodels.tsa.stattools import adfuller

    if d_grid is None:
        d_grid = np.round(np.arange(0.0, 1.01, 0.1), 2)
    for d in d_grid:
        fd = frac_diff_ffd(series, float(d), threshold).dropna()
        if len(fd) < min_obs or fd.nunique() < 2:
            continue
        pvalue = float(adfuller(fd.to_numpy(), maxlag=1, regression="c", autolag=None)[1])
        if pvalue < adf_pmax:
            return float(d), pvalue
    return None
