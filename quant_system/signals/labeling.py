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

from typing import Iterable, Optional

import numpy as np
import pandas as pd


def ewm_volatility(close: pd.Series, span: int = 100) -> pd.Series:
    """Exponentially-weighted return volatility, for scaling the barriers."""
    return close.pct_change().ewm(span=span).std()


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
