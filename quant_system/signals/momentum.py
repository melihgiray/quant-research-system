"""Momentum signals.

Cross-sectional momentum is the Jegadeesh & Titman (1993) effect: winners over
the past ~12 months keep winning over the next month. We use the "12-1"
formation (skip the most recent month) because the very last month tends to
*reverse* (short-term reversal / microstructure), so including it pollutes the
signal. Long the top quintile, short the bottom quintile, rebalanced monthly -
a dollar-neutral book, which is what makes the factor decomposition interesting
(market beta should come out near zero).

Time-series momentum (Moskowitz, Ooi & Pedersen 2012) is the complementary
absolute-momentum idea: go long an asset if its own past return is positive,
short if negative, sized by inverse volatility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import MomentumConfig


def _formation_return(close: pd.DataFrame, lookback: int, skip: int) -> pd.DataFrame:
    """The 12-1 formation return: P[t-skip] / P[t-lookback] - 1.

    Both prices are strictly in the past relative to date t, so this never peeks.
    The engine's shift adds the final execution lag on top.
    """
    return close.shift(skip) / close.shift(lookback) - 1.0


def cross_sectional_momentum(price_data, cfg: MomentumConfig = None) -> pd.DataFrame:
    """Long top-quantile / short bottom-quantile by 12-1 momentum, monthly rebalance.

    Parameters
    ----------
    price_data : PriceData
        Universe panel (use a cross-section of >= ~10 names).
    cfg : MomentumConfig
        lookback / skip / quantile / rebalance.

    Returns
    -------
    pd.DataFrame
        Daily target weights (held between rebalances). Dollar-neutral: the long
        leg sums to +0.5 and the short leg to -0.5 (gross = 1, net = 0).
    """
    cfg = cfg or MomentumConfig()
    close = price_data.close
    mom = _formation_return(close, cfg.lookback, cfg.skip)

    weights = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    # Rebalance dates: every `rebalance` trading days, once the lookback has warmed up.
    rebal_positions = range(cfg.lookback, len(close.index), cfg.rebalance)
    for pos in rebal_positions:
        d = close.index[pos]
        row = mom.loc[d].dropna()
        if len(row) < 5:                       # need a real cross-section to rank
            continue
        n_leg = max(1, int(round(len(row) * cfg.quantile)))
        longs = row.nlargest(n_leg).index
        shorts = row.nsmallest(n_leg).index
        w = pd.Series(0.0, index=close.columns)
        w[longs] = 0.5 / len(longs)
        w[shorts] = -0.5 / len(shorts)
        weights.loc[d] = w

    # Hold the last rebalance's weights until the next one; flat before the first.
    return weights.ffill().fillna(0.0)


def time_series_momentum(
    price_data,
    lookback: int = 252,
    skip: int = 21,
    vol_lookback: int = 63,
    target_vol_daily: float = 0.004,
    max_weight: float = 0.10,
) -> pd.DataFrame:
    """Absolute (time-series) momentum, inverse-vol sized.

    weight_i = sign(past return_i) * (target_vol / realised_vol_i), capped.

    Going long winners and short losers *per asset* (not relative to peers) makes
    this a trend-following book. Inverse-vol sizing equalises risk contribution so
    one jumpy name does not dominate.

    Parameters
    ----------
    lookback, skip : int
        Formation window and skip (same 12-1 logic as the cross-sectional version).
    vol_lookback : int
        Window for the realised-vol used in inverse-vol sizing.
    target_vol_daily : float
        Per-name daily vol target (≈ 0.004 ~ 6.3%/yr).
    max_weight : float
        Per-name absolute cap.
    """
    close = price_data.close
    mom = _formation_return(close, lookback, skip)
    vol = close.pct_change().rolling(vol_lookback, min_periods=vol_lookback // 2).std()
    sizing = (target_vol_daily / vol).clip(upper=max_weight / 0.5)
    raw = np.sign(mom) * sizing
    weights = raw.clip(lower=-max_weight, upper=max_weight)
    # Normalise gross to <= 1 so leverage is controlled.
    gross = weights.abs().sum(axis=1).replace(0.0, np.nan)
    scale = (1.0 / gross).clip(upper=1.0)
    return weights.mul(scale, axis=0).fillna(0.0)
