"""Execution constraints: volume-participation caps with carried-over fills.

The base engine fills any trade in full on the day it is ordered. That is fine
for a small book, but a desk running real size will not print 40% of a name's
daily volume in one session; a common rule of thumb is to keep an order under
5-10% of ADV and work the remainder over the following days.

This module models exactly that. Each day, the trade toward the target book is
clipped to a per-name cap expressed in weight terms:

    cap_weight = max_participation * ADV_shares * price / capital

Whatever could not be filled stays open, and the book keeps chasing the target
on subsequent days. Two consequences worth knowing about:

  * The held book lags the target when the target moves fast, so a strategy
    with violent rebalances loses more of its paper edge at scale. The gap
    between target and held ("fill gap") is reported so this is visible.
  * Daily participation is bounded, which also keeps the square-root impact
    cost per fill honest: the model's Q/V can no longer exceed the cap.

The core function is pure: (target holdings, cap weights) in, achievable
holdings out. The engine wires it in when an ExecutionConfig is provided.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from ..config import ExecutionConfig


def participation_cap_weights(close: pd.DataFrame, adv_shares: pd.DataFrame,
                              capital: float, max_participation: float) -> pd.DataFrame:
    """Per-name, per-day trade cap in weight terms.

    Parameters
    ----------
    close : pd.DataFrame
        Prices (date x ticker), used to convert share volume to notional.
    adv_shares : pd.DataFrame
        Average daily volume in shares, already lagged by the caller so a cap
        on day T uses liquidity known strictly before T.
    capital : float
        Portfolio notional. Bigger capital means the same dollar cap is a
        smaller weight, which is the whole point.
    max_participation : float
        Fraction of ADV tradable per day (0.05 = 5%).

    Missing ADV yields an unbounded cap for that name/day rather than a zero
    cap, so bad volume data degrades to the old full-fill behaviour instead of
    silently freezing the book.
    """
    cap = (max_participation * adv_shares * close) / capital
    return cap.where(np.isfinite(cap) & (cap > 0), np.inf)


def constrained_holdings(target_held: pd.DataFrame,
                         cap_weights: pd.DataFrame) -> pd.DataFrame:
    """Chase the target book under per-day trade caps; unfilled amounts carry.

    Day by day, the trade is ``clip(target - current, -cap, +cap)``. The state
    (current holdings) carries across days, so a large rebalance is worked over
    several sessions instead of pretending it fills at once.

    Parameters
    ----------
    target_held : pd.DataFrame
        The book the strategy wants each day (already execution-lagged).
    cap_weights : pd.DataFrame
        Max tradable weight per name per day (np.inf = uncapped).

    Returns
    -------
    pd.DataFrame
        The achievable held book, same shape as ``target_held``.
    """
    cap = cap_weights.reindex_like(target_held).fillna(np.inf).to_numpy(dtype=float)
    tgt = target_held.fillna(0.0).to_numpy(dtype=float)
    out = np.empty_like(tgt)
    current = np.zeros(tgt.shape[1])
    for i in range(tgt.shape[0]):
        desired = tgt[i] - current
        trade = np.clip(desired, -cap[i], cap[i])
        current = current + trade
        out[i] = current
    return pd.DataFrame(out, index=target_held.index, columns=target_held.columns)


def apply_execution(target_held: pd.DataFrame, close: pd.DataFrame,
                    adv_shares: pd.DataFrame, capital: float,
                    execution: Optional[ExecutionConfig]) -> Tuple[pd.DataFrame, pd.Series]:
    """Turn a target book into an achievable one under the configured constraints.

    Returns (held, fill_gap) where fill_gap is the daily sum of |target - held|,
    a direct measure of how far execution lags the strategy's intent. With no
    execution config (or no cap set) the target passes through untouched and
    the gap is zero.
    """
    if execution is None or execution.max_participation is None:
        zero = pd.Series(0.0, index=target_held.index)
        return target_held, zero
    cap = participation_cap_weights(close, adv_shares, capital,
                                    execution.max_participation)
    held = constrained_holdings(target_held, cap)
    gap = (target_held.fillna(0.0) - held).abs().sum(axis=1)
    return held, gap
