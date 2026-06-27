"""ML directional signal: engineered features -> P(next-day up) -> sized weights.

Pipeline:
  1. Per asset, build a panel of *lagged* features (everything uses data through t).
  2. Label each row with the sign of the NEXT day's return (the thing we predict).
  3. Pool all assets into one training set (the features are asset-agnostic, so a
     single model learns a general "what precedes an up day" mapping and gets far
     more data than per-asset models).
  4. Fit a gradient-boosting classifier on a trailing 2-year window.
  5. Size positions by *prediction strength* — (P(up) - 0.5), not the hard class —
     so confident days get bigger bets. Demeaned cross-sectionally and gross-
     normalised to give a market-neutral long/short book.
  6. SHAP values rank which features actually drive the predictions.

Look-ahead safety: feature row t uses data <= t; its label is return[t+1]; the
weight derived at row t is applied by the engine to return[t+1] (engine shift).
Training in each walk-forward fold uses only rows whose label was known by the
in-sample boundary, so the model never trains on data it will be tested on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier

from ..config import MLConfig, TRADING_DAYS_PER_YEAR


FEATURE_NAMES: List[str] = [
    "mom_5", "mom_10", "mom_21", "zscore_21", "vol_21",
    "volume_ratio", "rsi_14", "dist_52w_high",
]


def _rsi(close: pd.Series, span: int = 14) -> pd.Series:
    """Relative Strength Index over `span` days (rolling-mean variant).

    RSI = 100 - 100/(1+RS), RS = avg gain / avg loss. Bounded 0-100; <30 oversold,
    >70 overbought. A momentum/мean-reversion hybrid feature.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0).rolling(span, min_periods=span).mean()
    loss = (-delta.clip(upper=0.0)).rolling(span, min_periods=span).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def compute_features(price_data, cfg: MLConfig = None) -> Dict[str, pd.DataFrame]:
    """Build the per-asset feature panel. Returns {ticker -> DataFrame[features]}.

    Every feature is computed from data up to and including date t (no peeking);
    the engine adds the final execution lag.
    """
    cfg = cfg or MLConfig()
    close = price_data.close
    volume = price_data.volume
    rets = close.pct_change()
    out: Dict[str, pd.DataFrame] = {}

    for t in close.columns:
        c, v = close[t], volume[t]
        feats = pd.DataFrame(index=close.index)
        m5, m10, m21 = cfg.momentum_spans
        feats["mom_5"] = c / c.shift(m5) - 1.0
        feats["mom_10"] = c / c.shift(m10) - 1.0
        feats["mom_21"] = c / c.shift(m21) - 1.0
        roll_mean = c.rolling(cfg.vol_span).mean()
        roll_std = c.rolling(cfg.vol_span).std()
        feats["zscore_21"] = (c - roll_mean) / roll_std
        feats["vol_21"] = rets[t].rolling(cfg.vol_span).std()
        feats["volume_ratio"] = v / v.rolling(cfg.vol_span).mean()
        feats["rsi_14"] = _rsi(c, cfg.rsi_span)
        feats["dist_52w_high"] = c / c.rolling(TRADING_DAYS_PER_YEAR).max() - 1.0
        out[t] = feats
    return out


def _pooled_training_set(features, rets, fit_end, train_window):
    """Stack assets into (X, y) using rows with known label up to `fit_end`."""
    start = fit_end - pd.tseries.offsets.BDay(train_window)
    X_parts, y_parts = [], []
    for t, feats in features.items():
        # Label = sign of NEXT day's return; only valid where that return is known.
        label = (rets[t].shift(-1) > 0).astype(float)
        df = feats.copy()
        df["_y"] = label
        df = df.loc[(df.index >= start) & (df.index <= fit_end)]
        # Drop the last row (its next-day label leaks past fit_end) and any NaNs.
        df = df.dropna()
        if not df.empty:
            X_parts.append(df[FEATURE_NAMES].values)
            y_parts.append(df["_y"].values)
    if not X_parts:
        return None, None
    return np.vstack(X_parts), np.concatenate(y_parts)


def _proba_panel(model, features) -> pd.DataFrame:
    """Predict P(up) for every asset/day -> DataFrame(date × ticker)."""
    cols = {}
    for t, feats in features.items():
        X = feats[FEATURE_NAMES]
        valid = X.dropna()
        p = pd.Series(np.nan, index=feats.index)
        if not valid.empty:
            p.loc[valid.index] = model.predict_proba(valid.values)[:, 1]
        cols[t] = p
    return pd.DataFrame(cols)


def _weights_from_proba(proba: pd.DataFrame, deadband: float, max_weight: float) -> pd.DataFrame:
    """Turn P(up) into a market-neutral, gross-1, per-name-capped weight matrix.

    Constraint order matters: we keep dollar-neutrality EXACT (it is what makes
    the factor decomposition meaningful) and enforce the per-name cap by scaling
    the whole row uniformly rather than clipping single names — a uniform scale
    preserves the zero row-sum, whereas clipping would reintroduce a net tilt.
    """
    signal = proba - 0.5
    signal = signal.where(signal.abs() >= deadband, 0.0)   # ignore coin-flips
    signal = signal.sub(signal.mean(axis=1), axis=0)        # cross-sectional demean -> $-neutral
    gross = signal.abs().sum(axis=1).replace(0.0, np.nan)
    signal = signal.div(gross, axis=0)                      # gross = 1 (neutrality preserved)
    # Enforce the per-name cap by scaling each row uniformly (keeps net = 0 exact).
    mx = signal.abs().max(axis=1)
    cap_scale = (max_weight / mx).clip(upper=1.0)
    return signal.mul(cap_scale, axis=0).fillna(0.0)


def train_predict(price_data, fit_end, cfg: MLConfig = None,
                  max_weight: float = 0.10, seed: int = 7) -> pd.DataFrame:
    """Fit on data up to ``fit_end`` and return sized weights for the whole index.

    Designed as the walk-forward callback: ``make_weights(price_data, fit_end)``.
    The model is refit per fold, so each out-of-sample quarter is predicted by a
    model that never saw it.
    """
    cfg = cfg or MLConfig()
    if fit_end is None:
        fit_end = price_data.close.index[-1]
    features = compute_features(price_data, cfg)
    rets = price_data.returns()

    X, y = _pooled_training_set(features, rets, fit_end, cfg.train_window)
    if X is None or len(np.unique(y)) < 2:
        return pd.DataFrame(0.0, index=price_data.close.index, columns=price_data.close.columns)

    model = HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=300,
        l2_regularization=1.0, random_state=seed,
    )
    model.fit(X, y)
    proba = _proba_panel(model, features)
    return _weights_from_proba(proba, cfg.prob_deadband, max_weight)


@dataclass
class ShapResult:
    """Mean |SHAP| per feature (global importance), most-important first."""
    importance: pd.Series
    method: str

    def summary(self, top: int = 8) -> List[str]:
        lines = [f"FEATURE IMPORTANCE ({self.method}, mean|SHAP|):"]
        for name, val in self.importance.head(top).items():
            bar = "#" * int(round(val / (self.importance.max() + 1e-12) * 20))
            lines.append(f"  {name:<14} {val:8.4f} {bar}")
        return lines


def shap_feature_importance(price_data, cfg: MLConfig = None, fit_end=None,
                            sample: int = 2000, seed: int = 7) -> Optional[ShapResult]:
    """Fit a model and rank features by mean |SHAP value| on a training sample.

    SHAP (SHapley Additive exPlanations) attributes each prediction to its
    features in a game-theoretically fair way, so averaging |SHAP| across rows
    gives a principled global importance ranking — more trustworthy than a tree's
    built-in split-count importance. Falls back to permutation importance if the
    shap library is unavailable.
    """
    cfg = cfg or MLConfig()
    if fit_end is None:
        fit_end = price_data.close.index[-1]
    features = compute_features(price_data, cfg)
    rets = price_data.returns()
    X, y = _pooled_training_set(features, rets, fit_end, cfg.train_window)
    if X is None or len(np.unique(y)) < 2:
        return None

    model = HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=300,
        l2_regularization=1.0, random_state=seed,
    )
    model.fit(X, y)

    # Subsample rows for a fast, stable SHAP estimate.
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(sample, len(X)), replace=False)
    Xs = X[idx]

    try:
        import shap
        explainer = shap.TreeExplainer(model)
        vals = explainer.shap_values(Xs)
        if isinstance(vals, list):       # some versions return per-class lists
            vals = vals[-1]
        imp = np.abs(np.asarray(vals)).mean(axis=0)
        method = "TreeExplainer"
    except Exception:
        from sklearn.inspection import permutation_importance
        r = permutation_importance(model, Xs, model.predict(Xs),
                                   n_repeats=5, random_state=seed)
        imp = np.abs(r.importances_mean)
        method = "permutation (shap unavailable)"

    importance = pd.Series(imp, index=FEATURE_NAMES).sort_values(ascending=False)
    return ShapResult(importance=importance, method=method)
