"""train_predict routes through the pipeline, calibration and sample weights."""

import dataclasses

import numpy as np

from quant_system.config import default_config
from quant_system.data.loader import load_price_data
from quant_system.data.universe import universe
from quant_system.signals.ml_signal import train_predict

CFG = default_config()
MAX_W = 0.10


def _panel():
    return load_price_data(universe("largecaps")[:6], "2018-01-01", "2022-12-31",
                           use_synthetic=True)


def _assert_valid_book(w, panel):
    assert w.shape == (panel.close.shape[0], panel.close.shape[1])
    # Dollar-neutral rows, gross at most 1, per-name cap respected.
    assert np.allclose(w.sum(axis=1).to_numpy(), 0.0, atol=1e-9)
    assert w.abs().sum(axis=1).max() <= 1.0 + 1e-9
    assert w.abs().to_numpy().max() <= MAX_W + 1e-9


def test_default_path_is_a_valid_book():
    panel = _panel()
    _assert_valid_book(train_predict(panel, None, CFG.ml, max_weight=MAX_W), panel)


def test_uniqueness_weighting_runs_and_stays_valid():
    # On this balanced panel every date carries all six assets, so label
    # concurrency is uniform and the uniqueness weights are all equal: the book is
    # unchanged by design, and must still satisfy every invariant. The weights
    # only bite on a ragged panel (see the direct _fit_model test for proof the
    # weight reaches the estimator).
    panel = _panel()
    weighted_cfg = dataclasses.replace(CFG.ml, uniqueness_weighting=True)
    _assert_valid_book(train_predict(panel, None, weighted_cfg, max_weight=MAX_W), panel)


def test_calibrated_and_weighted_paths_produce_valid_books():
    panel = _panel()
    for cfg in (
        dataclasses.replace(CFG.ml, calibrate="sigmoid"),
        dataclasses.replace(CFG.ml, calibrate="isotonic", uniqueness_weighting=True),
    ):
        _assert_valid_book(train_predict(panel, None, cfg, max_weight=MAX_W), panel)
