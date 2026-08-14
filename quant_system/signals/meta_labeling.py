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

from ..config import MLConfig
from .ml_signal import (
    FEATURE_NAMES,
    _fit_model,
    _pooled_training_set,
    _proba_panel,
    _weights_from_proba,
    build_classifier,
    compute_features,
)

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


def _neutralize_and_cap(signal: pd.DataFrame, max_weight: float) -> pd.DataFrame:
    """Cross-sectionally demean, gross-normalise to 1, and cap per name.

    The same neutrality construction the primary uses in ``_weights_from_proba``:
    demeaning keeps the book dollar-neutral, and the per-name cap is enforced by
    scaling the whole row so the zero row-sum is preserved.
    """
    signal = signal.sub(signal.mean(axis=1), axis=0)
    gross = signal.abs().sum(axis=1).replace(0.0, np.nan)
    signal = signal.div(gross, axis=0)
    mx = signal.abs().max(axis=1)
    cap_scale = (max_weight / mx).clip(upper=1.0)
    return signal.mul(cap_scale, axis=0).fillna(0.0)


def meta_sized_weights(side: pd.DataFrame,
                       meta_prob: pd.DataFrame,
                       max_weight: float = 0.10,
                       min_prob: float = 0.5) -> pd.DataFrame:
    """Size each primary bet by the meta-model's conviction above a coin flip.

    Conviction is ``max(P(correct) - min_prob, 0)``, so a bet the meta-model is
    not sure about (P below ``min_prob``) is sized to zero: the meta-model's job
    is to veto and scale, not to flip the side. The signed convictions are then
    made dollar-neutral and capped like the primary book.
    """
    conviction = (meta_prob - min_prob).clip(lower=0.0)
    signal = side.reindex_like(meta_prob).fillna(0.0) * conviction.fillna(0.0)
    return _neutralize_and_cap(signal, max_weight)


def meta_train_predict(price_data, fit_end, cfg: MLConfig = None,
                       max_weight: float = 0.10, seed: int = 7,
                       min_prob: float = 0.5) -> pd.DataFrame:
    """Walk-forward callback: primary sets the side, meta-model sets the size.

    Fits the directional model as the primary, takes its side, trains the meta-
    model to grade that side, and sizes by meta-conviction. If the meta-model
    cannot be fit (too little data, or the primary took only winning bets in
    sample), it falls back to the primary's own sizing so the sleeve degrades to
    the primary rather than going flat.
    """
    cfg = cfg or MLConfig()
    if fit_end is None:
        fit_end = price_data.close.index[-1]
    features = compute_features(price_data, cfg)
    rets = price_data.returns()
    next_returns = rets.shift(-1)

    X, y = _pooled_training_set(features, rets, fit_end, cfg.train_window)
    if X is None or len(np.unique(y)) < 2:
        return pd.DataFrame(0.0, index=price_data.close.index, columns=price_data.close.columns)

    primary = build_classifier(seed)
    _fit_model(primary, X, y)
    proba = _proba_panel(primary, features)
    side = primary_side(proba, cfg.prob_deadband)

    meta_model = train_meta_model(features, side, next_returns, fit_end, cfg.train_window, seed)
    if meta_model is None:
        return _weights_from_proba(proba, cfg.prob_deadband, max_weight)
    meta_prob = meta_probability_panel(meta_model, features, side)
    return meta_sized_weights(side, meta_prob, max_weight, min_prob=min_prob)
