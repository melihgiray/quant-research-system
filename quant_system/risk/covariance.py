"""Ledoit-Wolf covariance shrinkage.

The sample covariance is noisy and often ill-conditioned when the number of assets
is close to the number of observations, which makes anything that inverts it (mean-
variance weights, and to a lesser extent the risk-budget allocators) unstable.
Ledoit-Wolf shrinks the sample covariance toward a well-conditioned target, a
scaled identity, by an amount chosen to minimise expected error: heavy when data is
scarce or the target is close to the truth, light when the sample estimate is
reliable.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def ledoit_wolf_shrinkage(returns) -> Tuple[np.ndarray, float]:
    """Shrunk covariance toward a scaled identity (Ledoit and Wolf, 2004).

    ``returns`` is a T x N array or DataFrame. Returns the shrunk covariance and
    the shrinkage intensity in [0, 1]. When ``returns`` is a DataFrame the
    covariance is returned as a labelled DataFrame.
    """
    labels = list(returns.columns) if isinstance(returns, pd.DataFrame) else None
    X = np.asarray(returns, dtype=float)
    T, N = X.shape
    X = X - X.mean(axis=0)
    sample = (X.T @ X) / T

    mu = np.trace(sample) / N
    target = mu * np.eye(N)
    dispersion = np.sum((sample - target) ** 2) / N          # ||S - F||^2 / N

    # Average squared error of the per-observation covariance estimate.
    rss = (X ** 2).sum(axis=1)
    term1 = np.sum(rss ** 2)
    quad = np.einsum("ti,ij,tj->t", X, sample, X).sum()
    noise = (term1 - 2.0 * quad + T * np.sum(sample ** 2)) / (N * T ** 2)
    noise = min(noise, dispersion)

    shrink = float(noise / dispersion) if dispersion > 0 else 0.0
    shrink = min(max(shrink, 0.0), 1.0)
    shrunk = shrink * target + (1.0 - shrink) * sample
    if labels is not None:
        shrunk = pd.DataFrame(shrunk, index=labels, columns=labels)
    return shrunk, shrink
