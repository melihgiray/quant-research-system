"""Tests for meta-labeling."""

import numpy as np
import pandas as pd
import pytest

from quant_system.signals.ml_signal import FEATURE_NAMES
from quant_system.signals.meta_labeling import (
    meta_labels,
    meta_probability_panel,
    primary_side,
    train_meta_model,
)


def _proba():
    idx = pd.bdate_range("2021-01-01", periods=3)
    return pd.DataFrame({
        "A": [0.70, 0.30, 0.52],     # long, short, near-flat
        "B": [0.48, np.nan, 0.90],   # near-flat, missing, long
    }, index=idx)


def test_side_is_long_short_or_flat():
    side = primary_side(_proba(), deadband=0.05)
    assert side.loc[side.index[0], "A"] == 1.0      # 0.70 -> long
    assert side.loc[side.index[1], "A"] == -1.0     # 0.30 -> short
    assert side.loc[side.index[2], "A"] == 0.0      # 0.52 within deadband -> flat
    assert side.loc[side.index[0], "B"] == 0.0      # 0.48 within deadband -> flat


def test_missing_prediction_stays_missing():
    side = primary_side(_proba(), deadband=0.05)
    assert np.isnan(side.loc[side.index[1], "B"])   # no P(up) -> no side


def test_zero_deadband_takes_a_side_everywhere_defined():
    side = primary_side(_proba(), deadband=0.0)
    assert side.loc[side.index[2], "A"] == 1.0      # 0.52 > 0.5 -> long with no deadband
    assert set(np.unique(side.dropna().to_numpy())) <= {-1.0, 0.0, 1.0}


def _sides_and_rets():
    idx = pd.bdate_range("2021-01-01", periods=3)
    side = pd.DataFrame({"A": [1.0, 1.0, 0.0], "B": [-1.0, -1.0, np.nan]}, index=idx)
    nxt = pd.DataFrame({"A": [0.02, -0.01, 0.03], "B": [-0.02, 0.01, np.nan]}, index=idx)
    return side, nxt


def test_meta_label_is_one_when_the_side_was_right():
    side, nxt = _sides_and_rets()
    labels = meta_labels(side, nxt)
    assert labels.loc[side.index[0], "A"] == 1.0    # long, +2% -> right
    assert labels.loc[side.index[1], "A"] == 0.0    # long, -1% -> wrong
    assert labels.loc[side.index[0], "B"] == 1.0    # short, -2% -> right (short made money)
    assert labels.loc[side.index[1], "B"] == 0.0    # short, +1% -> wrong


def test_meta_label_is_nan_where_no_bet_or_no_return():
    side, nxt = _sides_and_rets()
    labels = meta_labels(side, nxt)
    assert np.isnan(labels.loc[side.index[2], "A"])  # primary stood aside
    assert np.isnan(labels.loc[side.index[2], "B"])  # missing side and return


def _learnable_panels(n=500, seed=0):
    # The primary is right exactly when mom_5 > 0, so the meta-model has a real
    # pattern to learn: "trust the side when momentum agrees".
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    tickers = ["A", "B", "C"]
    features = {t: pd.DataFrame(rng.normal(size=(n, len(FEATURE_NAMES))),
                                index=idx, columns=FEATURE_NAMES) for t in tickers}
    side = pd.DataFrame(rng.choice([-1.0, 1.0], size=(n, 3)), index=idx, columns=tickers)
    nxt = pd.DataFrame(index=idx, columns=tickers, dtype=float)
    for t in tickers:
        right = features[t]["mom_5"] > 0
        nxt[t] = np.where(right, side[t] * 0.01, -side[t] * 0.01)
    return features, side, nxt


def test_meta_model_learns_when_to_trust_the_side():
    features, side, nxt = _learnable_panels()
    fit_end = features["A"].index[-1]
    model = train_meta_model(features, side, nxt, fit_end, train_window=600)
    assert model is not None
    proba = meta_probability_panel(model, features, side)
    # Valid probabilities where a bet exists, NaN where the primary stood aside.
    vals = proba.to_numpy()[~np.isnan(proba.to_numpy())]
    assert vals.min() >= 0.0 and vals.max() <= 1.0
    # The model should assign higher P(correct) when momentum agrees with the bet.
    agree = features["A"]["mom_5"] > 0
    pa = proba["A"]
    assert pa[agree].mean() > pa[~agree].mean()


def test_meta_model_is_none_on_single_class_labels():
    features, side, nxt = _learnable_panels()
    nxt = side * 0.01                                # every bet wins -> only one class
    fit_end = features["A"].index[-1]
    assert train_meta_model(features, side, nxt, fit_end, train_window=600) is None
