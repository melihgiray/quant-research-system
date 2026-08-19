"""SSVI: the surface SVI parameterisation with guaranteed no-arbitrage conditions.

Raw SVI fits each expiry freely, so nothing stops a fitted slice from carrying
butterfly arbitrage (the previous module penalises it after the fact). SSVI
(Gatheral and Jacquier, "Arbitrage-free SVI volatility surfaces", 2014) is the
sub-family whose parameters come with *sufficient* conditions for no arbitrage,
so a slice that satisfies them is provably free of butterfly arbitrage rather
than checked and hoped.

A single SSVI slice is

    w(k) = (theta / 2) * ( 1 + rho * psi * k + sqrt( (psi*k + rho)^2 + 1 - rho^2 ) )

where ``theta > 0`` is the at-the-money total variance, ``rho in (-1, 1)`` the
skew, and ``psi > 0`` the curvature. Implemented from the paper. SSVI is a special
case of raw SVI, so ``ssvi_to_svi_params`` maps it back and everything in
``svi.py`` (evaluation, g(k), plotting) works on the result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .svi import SVIParams


@dataclass(frozen=True)
class SSVIParams:
    """The three SSVI slice parameters."""
    theta: float          # at-the-money total variance (> 0)
    rho: float            # skew, in (-1, 1)
    psi: float            # curvature (> 0)


def ssvi_total_variance(k, params: SSVIParams):
    """Total implied variance w(k) under a single SSVI slice, vectorised over k."""
    k = np.asarray(k, dtype=float)
    u = params.psi * k
    return 0.5 * params.theta * (1.0 + params.rho * u
                                 + np.sqrt((u + params.rho) ** 2 + 1.0 - params.rho ** 2))


def ssvi_to_svi_params(params: SSVIParams) -> SVIParams:
    """Rewrite an SSVI slice as the equivalent raw-SVI parameters.

    SSVI is a sub-family of raw SVI, so this is exact: it lets the SVI evaluation,
    the g(k) arbitrage check and the plotting all operate on an SSVI slice.
    """
    theta, rho, psi = params.theta, params.rho, params.psi
    return SVIParams(
        a=theta * (1.0 - rho ** 2) / 2.0,
        b=theta * psi / 2.0,
        rho=rho,
        m=-rho / psi,
        sigma=np.sqrt(1.0 - rho ** 2) / psi,
    )
