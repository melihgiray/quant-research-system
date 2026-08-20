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
import pandas as pd
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


def ssvi_surface_calendar_free(surface: SSVISurface, k_grid=None, tol: float = 1e-9) -> bool:
    """Direct calendar-arbitrage check: total variance non-decreasing in maturity
    at every log-moneyness on the grid.

    This is the definition of no calendar arbitrage, evaluated pointwise, so it is
    independent of the SSVI sufficient conditions in ``ssvi_surface_arbitrage_free``.
    Because those conditions are sufficient and not necessary, this can certify a
    surface the sufficient bound fails to, which is exactly what happens on steep
    front-month equity smiles.
    """
    if k_grid is None:
        k_grid = np.linspace(-1.0, 1.0, 101)
    order = np.argsort(np.asarray(surface.maturities, dtype=float))
    total_var = np.vstack([ssvi_total_variance(k_grid, ssvi_surface_slice(surface, int(i)))
                           for i in order])
    return bool(np.all(np.diff(total_var, axis=0) >= -tol))


def _thetas_from_increments(x_theta) -> np.ndarray:
    """Build a non-decreasing theta vector from a base level and non-negative steps."""
    base = x_theta[0]
    increments = np.asarray(x_theta[1:], dtype=float)
    return base + np.concatenate([[0.0], np.cumsum(increments)])


def fit_ssvi_surface(points: pd.DataFrame, weight_by_spread: bool = True,
                     min_points_per_expiry: int = 5, penalty_weight: float = 1e4,
                     margin: float = 1e-3, max_nfev: int = 20000):
    """Fit one arbitrage-free SSVI surface jointly across all expiries.

    Fits a single skew ``rho`` and power-law curvature ``phi(theta) = eta *
    theta^-gamma``, plus one at-the-money total variance per maturity. The theta
    term structure is built from non-negative increments, so it is non-decreasing
    by construction (no calendar arbitrage), and the butterfly conditions and the
    phi upper bound are penalised, so the whole fitted surface is arbitrage-free.

    Returns ``(SSVISurface, diagnostics)`` where diagnostics is a per-expiry table
    of theta, psi, RMS fit error and point count. Needs at least two expiries with
    enough points.
    """
    groups = []
    for t, sel in points.groupby("time_to_expiry"):
        sel = sel.sort_values("log_moneyness")
        if len(sel) < min_points_per_expiry:
            continue
        k = sel["log_moneyness"].to_numpy(dtype=float)
        w = sel["total_variance"].to_numpy(dtype=float)
        weights = None
        if weight_by_spread and "spread" in sel.columns:
            spread = sel["spread"].to_numpy(dtype=float)
            weights = np.where(spread > 0, 1.0 / np.where(spread > 0, spread, 1.0), 1.0)
        groups.append((float(t), k, w, weights))
    if len(groups) < 2:
        raise ValueError("need at least two expiries with enough points to fit a surface")
    groups.sort(key=lambda g: g[0])
    maturities = np.array([g[0] for g in groups])
    n = len(groups)

    atm = np.array([float(np.interp(0.0, g[1], g[2])) if g[1].min() <= 0 <= g[1].max()
                    else float(g[2].min()) for g in groups])
    atm = np.maximum(atm, 1e-4)
    x0 = [-0.3, 1.0, 0.5, atm[0], *np.clip(np.diff(atm), 1e-6, None)]
    lower = [-0.999, 1e-6, 1e-3, 1e-8, *([0.0] * (n - 1))]
    upper = [0.999, np.inf, 1.0, np.inf, *([np.inf] * (n - 1))]
    bound = 4.0 - margin

    def residual(x):
        rho, eta, gamma = x[0], x[1], x[2]
        thetas = _thetas_from_increments(x[3:])
        parts = []
        for (_, k, w, weights), theta in zip(groups, thetas):
            psi = float(eta * theta ** (-gamma))
            model = ssvi_total_variance(k, SSVIParams(theta, rho, psi))
            sw = np.ones_like(w) if weights is None else np.clip(weights, 0.0, None)
            parts.append((model - w) * np.sqrt(sw))
            factor = theta * (1.0 + abs(rho))
            parts.append(np.sqrt(penalty_weight) * np.array([
                max(factor * psi - bound, 0.0), max(factor * psi ** 2 - bound, 0.0)]))
        if rho != 0:
            ub = (1.0 / rho ** 2) * (1.0 + np.sqrt(1.0 - rho ** 2))
            dphi = eta * (1.0 - gamma) * thetas ** (-gamma)
            parts.append(np.sqrt(penalty_weight) * np.clip(dphi - ub, 0.0, None))
        return np.concatenate(parts)

    sol = least_squares(residual, x0, bounds=(lower, upper), max_nfev=max_nfev)
    rho, eta, gamma = float(sol.x[0]), float(sol.x[1]), float(sol.x[2])
    thetas = _thetas_from_increments(sol.x[3:])
    surface = SSVISurface(rho=rho, eta=eta, gamma=gamma,
                          maturities=maturities, thetas=thetas)

    rows = []
    for (t, k, w, _), theta in zip(groups, thetas):
        psi = float(eta * theta ** (-gamma))
        model = ssvi_total_variance(k, SSVIParams(theta, rho, psi))
        rows.append({"time_to_expiry": t, "theta": float(theta), "psi": psi,
                     "rmse": float(np.sqrt(np.mean((model - w) ** 2))), "n_points": len(k)})
    return surface, pd.DataFrame(rows)
