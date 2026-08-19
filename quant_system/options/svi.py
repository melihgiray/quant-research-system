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
import pandas as pd
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


def svi_implied_vol(k, params: SVIParams, time_to_expiry: float):
    """Black-Scholes implied vol implied by an SVI slice at time_to_expiry.

    SVI models total variance ``w = iv^2 * T``; this converts back to a vol so a
    fitted slice can be plotted against quoted implied vols. ``time_to_expiry``
    must be positive.
    """
    if time_to_expiry <= 0:
        raise ValueError("time_to_expiry must be positive")
    w = np.maximum(svi_total_variance(k, params), 0.0)
    return np.sqrt(w / time_to_expiry)


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


def fit_svi_slice_no_arb(k, w, weights: Optional[np.ndarray] = None,
                         penalty_weight: float = 1e4,
                         margin: float = 1e-3,
                         grid_pad: float = 0.5,
                         max_nfev: int = 8000) -> Tuple[SVIParams, float]:
    """Calibrate SVI while penalising butterfly arbitrage in the fitted slice.

    The unconstrained fit reproduces whatever butterfly violations sit in the raw
    quotes. This augments the least-squares residual with a penalty on Gatheral's
    g(k) falling below ``margin`` (a small positive cushion) over a padded grid,
    so the optimiser trades a little data fit for a slice whose implied density
    stays positive. A larger ``penalty_weight`` pushes harder toward
    no-arbitrage at the cost of RMSE. The method drives g toward the boundary
    rather than proving positivity, so it is a strong soft constraint, not a
    certificate. Returns the fitted parameters and the data-only RMS error.
    """
    k = np.asarray(k, dtype=float)
    w = np.asarray(w, dtype=float)
    if len(k) < 5:
        raise ValueError("need at least 5 points to fit five SVI parameters")

    sw = np.ones_like(w) if weights is None else np.asarray(weights, dtype=float)
    sqrt_w = np.sqrt(np.clip(sw, 0.0, None))
    grid = np.linspace(k.min() - grid_pad, k.max() + grid_pad, 200)

    x0 = [max(float(w.min()), 1e-6), 0.1, -0.3, float(k[np.argmin(w)]), 0.1]
    lower = [-np.inf, 0.0, -0.999, -np.inf, 1e-6]
    upper = [np.inf, np.inf, 0.999, np.inf, np.inf]

    def residual(x):
        params = SVIParams(*x)
        data = (svi_total_variance(k, params) - w) * sqrt_w
        g = durability_g(grid, params)
        penalty = np.sqrt(penalty_weight) * np.clip(margin - g, 0.0, None)  # 0 where g >= margin
        return np.concatenate([data, penalty])

    sol = least_squares(residual, x0, bounds=(lower, upper), max_nfev=max_nfev)
    params = SVIParams(*sol.x)
    rmse = float(np.sqrt(np.mean((svi_total_variance(k, params) - w) ** 2)))
    return params, rmse


def svi_derivatives(k, params: SVIParams):
    """Return (w, w', w'') of raw SVI at ``k`` analytically."""
    k = np.asarray(k, dtype=float)
    centred = k - params.m
    root = np.sqrt(centred ** 2 + params.sigma ** 2)
    w = params.a + params.b * (params.rho * centred + root)
    w1 = params.b * (params.rho + centred / root)
    w2 = params.b * params.sigma ** 2 / root ** 3
    return w, w1, w2


def durability_g(k, params: SVIParams):
    """Gatheral's g(k): the risk-neutral density is proportional to it, so a slice
    is free of butterfly arbitrage exactly where g(k) >= 0."""
    k = np.asarray(k, dtype=float)
    w, w1, w2 = svi_derivatives(k, params)
    return (1 - k * w1 / (2 * w)) ** 2 - (w1 ** 2 / 4) * (1 / w + 0.25) + w2 / 2


def is_butterfly_arbitrage_free(params: SVIParams, k_grid=None,
                                tol: float = 1e-8) -> Tuple[bool, float]:
    """Check a fitted slice for butterfly arbitrage over a grid of log-moneyness.

    Returns (is_free, min_g): the slice is arbitrage-free when total variance is
    positive everywhere and Gatheral's g(k) stays non-negative. ``min_g`` is the
    worst point, so a small negative value flags a marginal violation.
    """
    if k_grid is None:
        k_grid = np.linspace(-2.0, 2.0, 401)
    g = durability_g(k_grid, params)
    w = svi_total_variance(k_grid, params)
    min_g = float(np.min(g))
    is_free = bool((w > 0).all() and min_g >= -tol)
    return is_free, min_g


def fit_svi_points(points: pd.DataFrame, weight_by_spread: bool = True,
                   min_points: int = 5, arbitrage_free: bool = False) -> pd.DataFrame:
    """Fit an SVI slice to every expiry in a surface's point table.

    ``points`` needs ``time_to_expiry``, ``log_moneyness`` and ``total_variance``
    columns (as produced by the surface builder); a ``spread`` column, when
    present and ``weight_by_spread`` is set, weights tighter quotes more heavily.
    With ``arbitrage_free`` each slice is fit with the butterfly penalty, trading
    a little RMSE for a density that stays positive. Returns one row per expiry
    with the fitted parameters, the RMS fit error, and the no-arbitrage verdict.
    """
    fitter = fit_svi_slice_no_arb if arbitrage_free else fit_svi_slice
    rows = []
    for t, sel in points.groupby("time_to_expiry"):
        sel = sel.sort_values("log_moneyness")
        k = sel["log_moneyness"].to_numpy(dtype=float)
        w = sel["total_variance"].to_numpy(dtype=float)
        if len(k) < min_points:
            continue
        weights = None
        if weight_by_spread and "spread" in sel.columns:
            spread = sel["spread"].to_numpy(dtype=float)
            weights = np.where(spread > 0, 1.0 / np.where(spread > 0, spread, 1.0), 1.0)
        params, rmse = fitter(k, w, weights=weights)
        arb_free, min_g = is_butterfly_arbitrage_free(params)
        rows.append({"time_to_expiry": float(t), "a": params.a, "b": params.b,
                     "rho": params.rho, "m": params.m, "sigma": params.sigma,
                     "rmse": rmse, "arb_free": arb_free, "min_g": min_g,
                     "n_points": int(len(k))})
    return pd.DataFrame(rows)


def fit_svi_surface(surface, arbitrage_free: bool = False) -> pd.DataFrame:
    """Fit SVI to each expiry of a built :class:`VolSurface` (see ``fit_svi_points``)."""
    return fit_svi_points(surface.points, arbitrage_free=arbitrage_free)
