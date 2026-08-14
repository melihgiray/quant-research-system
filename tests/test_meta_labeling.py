"""Tests for meta-labeling."""

import numpy as np
import pandas as pd
import pytest

from quant_system.signals.ml_signal import FEATURE_NAMES
from quant_system.config import default_config
from quant_system.data.loader import load_price_data
from quant_system.data.universe import universe
from quant_system.signals.meta_labeling import (
    meta_labels,
    meta_probability_panel,
    meta_sized_weights,
    meta_train_predict,
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


def test_meta_sizing_vetoes_low_conviction_bets():
    idx = pd.bdate_range("2021-01-01", periods=2)
    side = pd.DataFrame({"A": [1.0, 1.0], "B": [-1.0, -1.0]}, index=idx)
    weak = pd.DataFrame({"A": [0.45, 0.50], "B": [0.40, 0.48]}, index=idx)   # all <= 0.5
    assert (meta_sized_weights(side, weak).to_numpy() == 0.0).all()          # nothing survives


def test_meta_sizing_book_is_neutral_and_capped():
    idx = pd.bdate_range("2021-01-01", periods=3)
    side = pd.DataFrame({"A": [1, 1, -1], "B": [-1, 1, 1], "C": [1, -1, 1]},
                        index=idx, dtype=float)
    prob = pd.DataFrame({"A": [0.9, 0.7, 0.8], "B": [0.8, 0.6, 0.9], "C": [0.7, 0.85, 0.6]},
                        index=idx)
    w = meta_sized_weights(side, prob, max_weight=0.5)
    assert np.allclose(w.sum(axis=1).to_numpy(), 0.0, atol=1e-9)
    assert w.abs().to_numpy().max() <= 0.5 + 1e-9
    assert w.abs().sum(axis=1).max() <= 1.0 + 1e-9


def test_meta_train_predict_is_a_valid_book():
    cfg = default_config()
    panel = load_price_data(universe("largecaps")[:6], "2018-01-01", "2022-12-31",
                            use_synthetic=True)
    w = meta_train_predict(panel, None, cfg.ml, max_weight=0.10)
    assert w.shape == (panel.close.shape[0], panel.close.shape[1])
    assert np.allclose(w.sum(axis=1).to_numpy(), 0.0, atol=1e-9)
    assert w.abs().to_numpy().max() <= 0.10 + 1e-9
