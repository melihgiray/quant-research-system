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
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import least_squares

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


def ssvi_butterfly_free(params: SSVIParams, tol: float = 0.0) -> bool:
    """Whether a slice meets SSVI's sufficient conditions for no butterfly arbitrage.

    From Gatheral and Jacquier (2014): the slice is free of butterfly arbitrage if

        theta * psi * (1 + |rho|) < 4      and      theta * psi^2 * (1 + |rho|) <= 4.

    These are sufficient, not necessary, so a slice that fails them is not
    guaranteed to be arbitrageable, only not guaranteed clean. ``theta`` and
    ``psi`` must be positive for SSVI to be well defined.
    """
    if params.theta <= 0 or params.psi <= 0 or not -1.0 < params.rho < 1.0:
        return False
    factor = params.theta * (1.0 + abs(params.rho))
    return bool(factor * params.psi < 4.0 + tol
                and factor * params.psi ** 2 <= 4.0 + tol)


def fit_ssvi_slice(k, w, weights: Optional[np.ndarray] = None,
                   penalty_weight: float = 1e4,
                   margin: float = 1e-3,
                   max_nfev: int = 8000) -> Tuple[SSVIParams, float]:
    """Calibrate an SSVI slice, kept inside the no-arbitrage region.

    Fits ``theta, rho, psi`` by least squares with a penalty whenever the two
    Gatheral-Jacquier quantities exceed ``4 - margin``, so the fitted slice stays
    provably free of butterfly arbitrage. Because SSVI cannot represent an
    arbitrageable smile, fitting it to one trades data fit for a clean surface
    rather than reproducing the violation. Returns the parameters and the
    data-only RMS total-variance error.
    """
    k = np.asarray(k, dtype=float)
    w = np.asarray(w, dtype=float)
    if len(k) < 5:
        raise ValueError("need at least 5 points to fit an SSVI slice")

    sw = np.ones_like(w) if weights is None else np.asarray(weights, dtype=float)
    sqrt_w = np.sqrt(np.clip(sw, 0.0, None))
    bound = 4.0 - margin

    theta0 = float(np.interp(0.0, k, w)) if k.min() <= 0 <= k.max() else float(w.min())
    x0 = [max(theta0, 1e-4), -0.3, 1.0]
    lower = [1e-8, -0.999, 1e-6]
    upper = [np.inf, 0.999, np.inf]

    def residual(x):
        params = SSVIParams(*x)
        data = (ssvi_total_variance(k, params) - w) * sqrt_w
        factor = params.theta * (1.0 + abs(params.rho))
        c1 = np.clip(factor * params.psi - bound, 0.0, None)
        c2 = np.clip(factor * params.psi ** 2 - bound, 0.0, None)
        penalty = np.sqrt(penalty_weight) * np.array([c1, c2])
        return np.concatenate([data, penalty])

    sol = least_squares(residual, x0, bounds=(lower, upper), max_nfev=max_nfev)
    params = SSVIParams(*sol.x)
    rmse = float(np.sqrt(np.mean((ssvi_total_variance(k, params) - w) ** 2)))
    return params, rmse


@dataclass(frozen=True)
class SSVISurface:
    """A full SSVI surface: one skew and one curvature law across all expiries.

    A single ``rho`` and a power-law curvature ``phi(theta) = eta * theta^-gamma``
    tie the slices together, and the at-the-money total variance ``thetas`` (one
    per maturity, sorted) sets each slice. This shared structure is what lets the
    surface be calendar-arbitrage-free, not just each slice butterfly-free.
    """
    rho: float
    eta: float
    gamma: float
    maturities: np.ndarray
    thetas: np.ndarray


def ssvi_phi(theta, eta: float, gamma: float):
    """Power-law SSVI curvature phi(theta) = eta * theta^-gamma."""
    return eta * np.power(np.asarray(theta, dtype=float), -gamma)


def ssvi_surface_slice(surface: SSVISurface, i: int) -> SSVIParams:
    """The SSVI slice parameters for the ``i``-th maturity of a surface."""
    theta = float(surface.thetas[i])
    return SSVIParams(theta=theta, rho=surface.rho,
                      psi=float(ssvi_phi(theta, surface.eta, surface.gamma)))


def ssvi_surface_arbitrage_free(surface: SSVISurface, tol: float = 1e-6) -> Tuple[bool, bool]:
    """Return (butterfly_free, calendar_free) for a whole SSVI surface.

    Butterfly: every slice meets the per-slice conditions. Calendar (Gatheral and
    Jacquier 2014): the at-the-money total variance is non-decreasing in maturity
    and ``d/dtheta (theta * phi(theta))`` stays in ``[0, (1/rho^2)(1 + sqrt(1 -
    rho^2))]``. For the power law that derivative is ``eta*(1-gamma)*theta^-gamma``,
    so ``gamma <= 1`` gives the lower bound and small ``|rho|`` makes the upper
    bound loose (it is vacuous at rho = 0).
    """
    thetas = np.asarray(surface.thetas, dtype=float)
    butterfly = all(ssvi_butterfly_free(ssvi_surface_slice(surface, i))
                    for i in range(len(thetas)))

    monotone = bool(np.all(np.diff(thetas) >= -tol))
    dtheta_phi = surface.eta * (1.0 - surface.gamma) * np.power(thetas, -surface.gamma)
    lower_ok = bool(np.all(dtheta_phi >= -tol))
    if surface.rho == 0:
        upper_ok = True
    else:
        upper = (1.0 / surface.rho ** 2) * (1.0 + np.sqrt(1.0 - surface.rho ** 2))
        upper_ok = bool(np.all(dtheta_phi <= upper + tol))
    calendar = monotone and lower_ok and upper_ok
    return butterfly, calendar
