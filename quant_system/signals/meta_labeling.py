"""Meta-labeling (Lopez de Prado, Advances in Financial ML, ch. 3).

A directional model answers "which way": long or short. A meta-model answers a
different, easier question: given that the primary said long here, is it right,
and how sure are we. The primary sets the side; the meta-model sets the size, and
can veto a bet entirely by sizing it to zero. Splitting the two lets the second
model raise precision (cut the false positives the primary trades) without
touching the first model's recall.

The functions here operate on panels (date x ticker), so the primary can be any
signal that produces a P(up); the ML directional model is the one wired in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def primary_side(proba: pd.DataFrame, deadband: float = 0.0) -> pd.DataFrame:
    """The primary bet's side from P(up): +1 long, -1 short, 0 stand aside.

    Within ``deadband`` of a coin flip the primary takes no side, so the
    meta-model is never asked to grade a bet the primary would not have made.
    NaNs (no prediction) are preserved.
    """
    signal = proba - 0.5
    side = np.sign(signal)                                # -1/0/+1, NaN where no P(up)
    side = side.where(signal.abs() >= deadband, 0.0)      # inside the deadband -> stand aside
    return side.where(proba.notna())                      # restore NaN where there was no prediction


def meta_labels(side: pd.DataFrame, next_returns: pd.DataFrame) -> pd.DataFrame:
    """1 where the primary side made money, 0 where it lost, NaN where no bet.

    The meta-model's target: for each bet the primary actually took (side != 0),
    did the next-period return go the primary's way. Rows where the primary stood
    aside, or where the return is missing, are NaN and drop out of training.
    """
    aligned = next_returns.reindex_like(side)
    pnl = side * aligned
    labels = (pnl > 0).astype(float)
    return labels.where((side != 0) & side.notna() & aligned.notna())
