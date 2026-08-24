"""Triple-barrier labeling (Lopez de Prado, Advances in Financial ML, ch. 3).

Fixed-horizon labels ("up in N days?") ignore the path: a position that spikes to
a profit and then reverses gets the same label as one that bled the whole way. The
triple-barrier method labels by which of three barriers a path touches first from
the entry: an upper barrier (a profit target), a lower barrier (a stop), and a
vertical barrier (a holding-period limit). The label is +1 if the profit target is
hit first, -1 if the stop is, and the sign of the return at the time limit if
neither is, so the label reflects what actually would have happened to the trade.

Barrier widths scale with volatility, so a target means the same thing in a calm
and a wild market.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


def ewm_volatility(close: pd.Series, span: int = 100) -> pd.Series:
    """Exponentially-weighted return volatility, for scaling the barriers."""
    return close.pct_change().ewm(span=span).std()


def cusum_events(series: pd.Series, threshold: float) -> List[int]:
    """Symmetric CUSUM filter: positions where a run of moves exceeds ``threshold``.

    (Lopez de Prado, ch. 2.) Sampling every bar wastes model capacity on
    quiet noise; this keeps only the bars where the cumulative move since the last
    event, up or down, breaks a threshold, so the events cluster where something
    happened. Running positive and negative sums accumulate the step-to-step
    changes and reset to zero whenever one crosses the threshold.

    Pass log prices to make ``threshold`` a cumulative-return level. Returns
    integer positions into ``series``, ready to feed ``triple_barrier_labels``.
    """
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    values = np.asarray(series, dtype=float)
    events: List[int] = []
    s_pos = 0.0
    s_neg = 0.0
    for i in range(1, len(values)):
        diff = values[i] - values[i - 1]
        if not np.isfinite(diff):
            continue
        s_pos = max(0.0, s_pos + diff)
        s_neg = min(0.0, s_neg + diff)
        if s_pos >= threshold:
            s_pos = 0.0
            events.append(i)
        elif s_neg <= -threshold:
            s_neg = 0.0
            events.append(i)
    return events


def triple_barrier_labels(close: pd.Series,
                          event_positions: Iterable[int],
                          pt: float = 1.0,
                          sl: float = 1.0,
                          vertical: int = 10,
                          vol: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Label each event by the first barrier its forward path touches.

    Parameters
    ----------
    close : pd.Series
        Price series.
    event_positions : iterable of int
        Integer positions into ``close`` where a position is opened.
    pt, sl : float
        Profit-target and stop widths as multiples of the event's volatility. A
        value of 0 disables that horizontal barrier.
    vertical : int
        Holding-period limit in bars.
    vol : np.ndarray, optional
        Per-bar volatility used for the barrier widths; defaults to
        ``ewm_volatility(close)``.

    Returns
    -------
    pd.DataFrame
        One row per event with the entry and touch timestamps, the realised
        return at the touch, and the label in {-1, 0, 1}.
    """
    prices = close.to_numpy(dtype=float)
    n = len(prices)
    vol = ewm_volatility(close).to_numpy() if vol is None else np.asarray(vol, dtype=float)

    rows = []
    for i in event_positions:
        if i < 0 or i >= n or not np.isfinite(vol[i]):
            continue
        entry = prices[i]
        upper = pt * vol[i]
        lower = -sl * vol[i]
        end = min(i + vertical, n - 1)

        touch, label = end, 0
        ret = prices[end] / entry - 1.0
        for j in range(i + 1, end + 1):
            r = prices[j] / entry - 1.0
            if pt > 0 and r >= upper:
                touch, label, ret = j, 1, r
                break
            if sl > 0 and r <= lower:
                touch, label, ret = j, -1, r
                break
        else:
            label = int(np.sign(ret))            # neither horizontal barrier hit: sign at the limit

        rows.append({"event": close.index[i], "touch": close.index[touch],
                     "ret": float(ret), "label": int(label)})
    return pd.DataFrame(rows)
