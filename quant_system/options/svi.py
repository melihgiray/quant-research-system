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

import numpy as np


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
