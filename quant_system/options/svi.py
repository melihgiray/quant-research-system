"""Raw SVI (stochastic volatility inspired) parameterisation of a smile.

This is the published raw-SVI formula (Gatheral, "A parsimonious arbitrage-free
implied volatility parameterization", 2004; Gatheral and Jacquier, "Arbitrage-free
SVI volatility surfaces", 2014), implemented from the paper rather than ported
from any specific codebase. It fits one expiry's total implied variance as a
function of log-forward-moneyness with five interpretable parameters:

    w(k) = a + b * ( rho * (k - m) + sqrt( (k - m)^2 + sigma^2 ) )

where ``w = iv^2 * T`` is total variance and ``k = ln(K / F)``. The parameters:
``a`` sets the overall level, ``b >= 0`` the wing slope, ``rho in (-1, 1)`` the
skew, ``m`` the horizontal shift of the minimum, and ``sigma > 0`` how rounded
the vertex is. The two wings are asymptotically linear in k with slopes
``b(1 +/- rho)``, which is why total variance is SVI's natural coordinate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class SVIParams:
    """The five raw-SVI parameters for a single expiry."""
    a: float
    b: float
    rho: float
    m: float
    sigma: float


def svi_total_variance(k, params: SVIParams):
    """Total implied variance w(k) under raw SVI, vectorised over ``k``."""
    k = np.asarray(k, dtype=float)
    centred = k - params.m
    return params.a + params.b * (params.rho * centred
                                  + np.sqrt(centred ** 2 + params.sigma ** 2))


def fit_svi_slice(k, w, weights: Optional[np.ndarray] = None,
                  max_nfev: int = 5000) -> Tuple[SVIParams, float]:
    """Calibrate the five raw-SVI parameters to one expiry's (k, w) points.

    Least-squares fit with the SVI feasibility bounds enforced directly: ``b >= 0``,
    ``rho in (-1, 1)`` and ``sigma > 0``. ``weights`` (e.g. inverse bid-ask width)
    lets tighter quotes pull the fit harder. Returns the fitted parameters and the
    root-mean-square total-variance error.
    """
    k = np.asarray(k, dtype=float)
    w = np.asarray(w, dtype=float)
    if k.shape != w.shape:
        raise ValueError("k and w must have the same shape")
    if len(k) < 5:
        raise ValueError("need at least 5 points to fit five SVI parameters")

    sw = np.ones_like(w) if weights is None else np.asarray(weights, dtype=float)
    sqrt_w = np.sqrt(np.clip(sw, 0.0, None))

    x0 = [max(float(w.min()), 1e-6), 0.1, -0.3, float(k[np.argmin(w)]), 0.1]
    lower = [-np.inf, 0.0, -0.999, -np.inf, 1e-6]
    upper = [np.inf, np.inf, 0.999, np.inf, np.inf]

    def residual(x):
        model = svi_total_variance(k, SVIParams(*x))
        return (model - w) * sqrt_w

    sol = least_squares(residual, x0, bounds=(lower, upper), max_nfev=max_nfev)
    params = SVIParams(*sol.x)
    rmse = float(np.sqrt(np.mean((svi_total_variance(k, params) - w) ** 2)))
    return params, rmse
