"""Combine strategy sleeves into one book.

Each sleeve (momentum, pairs, ML) is its own return stream, validated out of
sample on its own universe. To run them as one portfolio we allocate capital
across sleeves inversely to each sleeve's recent volatility - a risk-parity
weighting that gives the calm sleeve more room and the jumpy one less - using
only trailing data, lagged a day, so the allocation never peeks at the return it
is about to earn.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd


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
