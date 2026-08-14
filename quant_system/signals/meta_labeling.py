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

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .ml_signal import FEATURE_NAMES, _fit_model, build_classifier

META_COLUMNS = FEATURE_NAMES + ["side"]      # the meta-model sees the features plus the side taken


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


def _pooled_meta_set(features, side, labels, fit_end, train_window):
    """Stack (features + side) rows the primary actually bet, with meta-labels.

    Only rows dated strictly before ``fit_end`` are used, mirroring the primary
    model's cut so the meta-model never trains on a label from the out-of-sample
    period.
    """
    start = fit_end - pd.tseries.offsets.BDay(train_window)
    X_parts, y_parts = [], []
    for t, feats in features.items():
        if t not in side.columns:
            continue
        df = feats[FEATURE_NAMES].copy()
        df["side"] = side[t]
        df["_y"] = labels[t]
        df = df.loc[(df.index >= start) & (df.index < fit_end)]
        df = df[df["side"].fillna(0.0) != 0.0].dropna()
        if not df.empty:
            X_parts.append(df[META_COLUMNS].to_numpy())
            y_parts.append(df["_y"].to_numpy())
    if not X_parts:
        return None, None
    return np.vstack(X_parts), np.concatenate(y_parts)


def train_meta_model(features: Dict[str, pd.DataFrame],
                     side: pd.DataFrame,
                     next_returns: pd.DataFrame,
                     fit_end,
                     train_window: int,
                     seed: int = 7):
    """Fit the secondary classifier that predicts whether the primary side is right.

    Returns the fitted model, or None if there is not enough data or the meta-
    labels are single-class (e.g. the primary took no losing bets in-sample).
    """
    labels = meta_labels(side, next_returns)
    X, y = _pooled_meta_set(features, side, labels, fit_end, train_window)
    if X is None or len(np.unique(y)) < 2:
        return None
    model = build_classifier(seed)
    _fit_model(model, X, y)
    return model


def meta_probability_panel(model, features: Dict[str, pd.DataFrame],
                           side: pd.DataFrame) -> pd.DataFrame:
    """P(primary is right) for every bet the primary took -> DataFrame(date x ticker).

    NaN where the primary stood aside or a feature is missing.
    """
    cols = {}
    for t, feats in features.items():
        if t not in side.columns:
            continue
        frame = feats[FEATURE_NAMES].copy()
        frame["side"] = side[t]
        p = pd.Series(np.nan, index=feats.index)
        valid = frame.dropna()
        valid = valid[valid["side"] != 0.0]
        if not valid.empty:
            p.loc[valid.index] = model.predict_proba(valid[META_COLUMNS].to_numpy())[:, 1]
        cols[t] = p
    return pd.DataFrame(cols)
