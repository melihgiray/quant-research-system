"""Transaction-cost models.

Two components, matching how an execution desk decomposes trading cost:

1. Half-spread  — you cross half the bid/ask on every share, independent of size.
                  Linear in traded notional. ~1-2 bps for liquid large-caps.

2. Market impact — your own order pushes the price against you. The empirically
                   supported functional form is the *square-root* law:

                       impact_fraction = eta * sigma * sqrt(Q / V)

   where Q = shares traded, V = average daily volume, sigma = daily volatility,
   eta ~ 0.1. Cost in dollars is impact_fraction * traded_notional.

Why square-root and not linear?  Empirically (Almgren 2005; Kyle/Obizhaeva;
the BARRA/Citadel-style models every desk uses) impact grows roughly with the
*square root* of participation rate, not linearly. Intuition: liquidity replenishes
as you trade, so the marginal share is cheaper than the first — concave, not linear.
A linear model massively over-penalises large orders and under-penalises small ones.
Crucially, square-root impact makes cost depend on *capital* (via Q), so the same
strategy is cheaper at $1M than at $1B — exactly the capacity story a PM cares about.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import CostConfig


def half_spread_cost(turnover_fraction: float, half_spread_bps: float) -> float:
    """Spread cost as a fraction of portfolio value.

    Parameters
    ----------
    turnover_fraction : float
        One-way traded notional as a fraction of portfolio (sum_i |dw_i|).
    half_spread_bps : float
        Half the bid/ask spread in basis points.

    Returns
    -------
    float
        Cost as a fraction of portfolio value (linear in turnover).
    """
    return turnover_fraction * (half_spread_bps / 1e4)


def square_root_impact(participation: np.ndarray, sigma: np.ndarray, eta: float) -> np.ndarray:
    """Per-name market-impact cost as a fraction of *that name's* traded notional.

    impact = eta * sigma * sqrt(Q / V)

    Parameters
    ----------
    participation : array
        Q / V, the order size as a fraction of average daily volume.
    sigma : array
        Daily volatility of the name (decimal, e.g. 0.02 = 2%/day).
    eta : float
        Dimensionless impact coefficient (~0.1 for equities).

    Returns
    -------
    array
        Impact as a fraction of traded notional, per name.
    """
    participation = np.clip(np.asarray(participation, dtype=float), 0.0, None)
    sigma = np.nan_to_num(np.asarray(sigma, dtype=float), nan=0.0)
    return eta * sigma * np.sqrt(participation)


def transaction_cost_fraction(
    delta_weights: pd.Series,
    prices: pd.Series,
    adv_shares: pd.Series,
    sigma: pd.Series,
    cost: CostConfig,
) -> float:
    """Total cost of one rebalance, as a fraction of portfolio value.

    Combines linear half-spread and square-root impact across all traded names:

        cost_frac = sum_i |dw_i| * ( spread_frac + eta * sigma_i * sqrt(Q_i / V_i) )

    with Q_i = |dw_i| * capital / price_i  (shares traded) and V_i = ADV in shares.
    Note the spread term's capital cancels, but impact does not — larger capital
    means larger participation Q/V means higher per-unit impact. That capital
    dependence is the whole reason to use this model.

    Parameters
    ----------
    delta_weights : pd.Series
        Change in held weight per name on this day (new - prev), indexed by ticker.
    prices : pd.Series
        Price per name (same index).
    adv_shares : pd.Series
        Average daily volume in shares per name.
    sigma : pd.Series
        Daily volatility per name.
    cost : CostConfig
        Cost assumptions (spread, eta, capital).

    Returns
    -------
    float
        Total transaction cost for the day as a fraction of portfolio value.
    """
    dw = delta_weights.abs()
    traded = dw[dw > 0]
    if traded.empty:
        return 0.0

    idx = traded.index
    px = prices.reindex(idx).astype(float)
    adv = adv_shares.reindex(idx).astype(float)
    sig = sigma.reindex(idx).astype(float)
    # Floor the vol input so impact is never understated when the rolling estimate
    # is missing/zero (e.g. trades during the warm-up window).
    sig = sig.where(np.isfinite(sig) & (sig > 0), cost.default_daily_vol)

    # Dollars traded per name and resulting participation rate Q/V.
    dollars = traded.values * cost.capital
    with np.errstate(divide="ignore", invalid="ignore"):
        shares = np.where(px.values > 0, dollars / px.values, 0.0)
        # Floor ADV to avoid divide-by-zero / absurd participation on bad data.
        adv_safe = np.where(adv.values > 0, adv.values, np.nan)
        participation = np.where(np.isnan(adv_safe), 0.0, shares / adv_safe)

    spread_frac = cost.half_spread_bps / 1e4
    impact_frac = square_root_impact(participation, sig.values, cost.impact_eta)

    # Per-name cost as a fraction of portfolio = |dw_i| * (spread + impact_i).
    per_name = traded.values * (spread_frac + impact_frac)
    return float(np.nansum(per_name))
