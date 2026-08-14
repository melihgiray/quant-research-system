"""Combine strategy sleeves into one book.

Each sleeve (momentum, pairs, ML) is its own return stream, validated out of
sample on its own universe. To run them as one portfolio we allocate capital
across sleeves inversely to each sleeve's recent volatility - a risk-parity
weighting that gives the calm sleeve more room and the jumpy one less - using
only trailing data, lagged a day, so the allocation never peeks at the return it
is about to earn.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..config import TRADING_DAYS_PER_YEAR
from .hrp import hrp_weights


def combine_weights(weights_by_sleeve: Dict[str, pd.DataFrame],
                    allocations: pd.DataFrame) -> pd.DataFrame:
    """Stitch per-sleeve weight matrices into one book on a shared ticker set.

    Each sleeve holds its own date x ticker weights (its book scaled to its own
    gross). ``allocations`` says how much of the portfolio each sleeve gets on a
    given day. The combined weight of a ticker is the allocation-weighted sum of
    that ticker's weight across the sleeves that trade it, over the union of all
    tickers. Sleeves trade different universes, so most cells are 0 for a sleeve.

    Parameters
    ----------
    weights_by_sleeve : dict[str, pd.DataFrame]
        One date x ticker weight matrix per sleeve, keyed by name. Keys must
        match the columns of ``allocations``.
    allocations : pd.DataFrame
        Date x sleeve capital allocation, e.g. from ``inverse_vol_allocations``.

    Returns
    -------
    pd.DataFrame
        Combined date x ticker weights on the union of tickers, aligned to the
        allocation index.
    """
    if not weights_by_sleeve:
        raise ValueError("need at least one sleeve")
    missing = set(weights_by_sleeve) - set(allocations.columns)
    if missing:
        raise ValueError(f"no allocation column for sleeves: {sorted(missing)}")

    tickers = sorted({t for w in weights_by_sleeve.values() for t in w.columns})
    index = allocations.index
    combined = pd.DataFrame(0.0, index=index, columns=tickers)
    for name, weights in weights_by_sleeve.items():
        aligned = weights.reindex(index=index, columns=tickers).fillna(0.0)
        combined = combined.add(aligned.mul(allocations[name], axis=0), fill_value=0.0)
    return combined


def _sleeve_frame(sleeve_returns: Dict[str, pd.Series]) -> pd.DataFrame:
    """Align a dict of sleeve return series into one date x sleeve frame."""
    if not sleeve_returns:
        raise ValueError("need at least one sleeve")
    return pd.DataFrame(dict(sleeve_returns)).sort_index()


def inverse_vol_allocations(sleeve_returns: Dict[str, pd.Series],
                            lookback: int = 63,
                            min_periods: int = 20) -> pd.DataFrame:
    """Daily capital allocation across sleeves, inverse to trailing volatility.

    Higher recent volatility means a smaller allocation, so each sleeve
    contributes a similar amount of risk. The trailing-vol estimate is lagged
    one day, so the weight used on day t depends only on returns through t-1.

    Parameters
    ----------
    sleeve_returns : dict[str, pd.Series]
        One return series per sleeve, keyed by name.
    lookback : int
        Window for the trailing realised-vol estimate (trading days).
    min_periods : int
        Minimum observations before a sleeve gets a vol estimate; until then it
        is not allocated to.

    Returns
    -------
    pd.DataFrame
        Allocations (date x sleeve). Each row sums to 1 over the sleeves with a
        usable vol estimate that day; a sleeve still warming up gets 0 and the
        rest are renormalised. Rows before any sleeve warms up are all 0.
    """
    frame = _sleeve_frame(sleeve_returns)
    vol = frame.rolling(lookback, min_periods=min_periods).std().shift(1)
    inv = 1.0 / vol.where(vol > 0)                      # NaN where vol missing/zero
    alloc = inv.div(inv.sum(axis=1), axis=0)            # normalise over available sleeves
    return alloc.fillna(0.0)


def hrp_allocations(sleeve_returns: Dict[str, pd.Series],
                    lookback: int = TRADING_DAYS_PER_YEAR,
                    refit_every: int = 21,
                    min_obs: Optional[int] = None) -> pd.DataFrame:
    """Daily capital allocation across sleeves by Hierarchical Risk Parity.

    Where ``inverse_vol_allocations`` ignores how the sleeves co-move, HRP
    clusters them on their correlation and splits the risk budget down that tree,
    so two sleeves that move together share a budget instead of each taking a full
    inverse-variance share. The allocation is recomputed on the trailing
    ``lookback`` window every ``refit_every`` days and held between refits; every
    row is fit on returns strictly before its day, so it is causal. Rows before
    ``min_obs`` (defaults to ``lookback``) are zero.
    """
    frame = _sleeve_frame(sleeve_returns)
    min_obs = min_obs or lookback
    alloc = pd.DataFrame(0.0, index=frame.index, columns=frame.columns)
    last = None
    for pos in range(len(frame)):
        if pos < min_obs:
            continue
        if (pos - min_obs) % refit_every == 0:
            window = frame.iloc[max(0, pos - lookback):pos].dropna()   # data strictly before pos
            if window.shape[0] >= 20 and window.shape[1] == frame.shape[1]:
                try:
                    last = hrp_weights(window)
                except Exception:
                    pass                                               # keep prior weights on failure
        if last is not None:
            alloc.iloc[pos] = last.reindex(frame.columns).fillna(0.0).to_numpy()
    return alloc


def blend_returns(sleeve_returns: Dict[str, pd.Series],
                  allocations: Optional[pd.DataFrame] = None,
                  lookback: int = 63,
                  min_periods: int = 20) -> pd.Series:
    """Blend sleeve return streams into one book via daily allocations.

    Each day's blended return is the allocation-weighted sum of the sleeve
    returns. When ``allocations`` is omitted, inverse-vol allocations are
    computed from the same streams, which is causal (the allocation for day t
    uses vol through t-1, then earns day t's return).

    Returns
    -------
    pd.Series
        The blended (pre vol-target) return stream.
    """
    frame = _sleeve_frame(sleeve_returns)
    if allocations is None:
        allocations = inverse_vol_allocations(
            sleeve_returns, lookback=lookback, min_periods=min_periods)
    alloc = allocations.reindex(index=frame.index, columns=frame.columns).fillna(0.0)
    return (frame.fillna(0.0) * alloc).sum(axis=1)


def volatility_target(returns: pd.Series,
                      target_vol: float = 0.10,
                      lookback: int = TRADING_DAYS_PER_YEAR,
                      max_leverage: float = 3.0) -> pd.Series:
    """Scale a return stream so its trailing annualised vol tracks ``target_vol``.

    The return-stream analogue of ``risk.sizing.vol_target_scale``: estimate the
    stream's realised annualised vol over a trailing window, multiply by
    target/realised, lag the scaler one day (size today on vol known through
    yesterday), and cap the multiplier at ``max_leverage``.

    Returns
    -------
    pd.Series
        The vol-targeted return stream, on the same index as ``returns``.
    """
    realised = returns.rolling(lookback, min_periods=lookback // 4).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    scaler = (target_vol / realised).replace([np.inf, -np.inf], np.nan)
    scaler = scaler.clip(upper=max_leverage).shift(1)   # size today on yesterday's vol
    return returns.mul(scaler)
