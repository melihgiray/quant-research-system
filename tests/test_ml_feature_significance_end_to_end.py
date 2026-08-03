"""End-to-end test for the FDR-controlled ML feature significance."""

import numpy as np
import pandas as pd

from quant_system.config import default_config
from quant_system.data.loader import load_price_data
from quant_system.data.universe import universe
from quant_system.signals.ml_signal import ml_feature_significance, FEATURE_NAMES

CFG = default_config()


def _panel():
    return load_price_data(universe("largecaps")[:6], "2018-01-01", "2022-12-31",
                           use_synthetic=True)


def test_report_covers_every_feature_with_valid_columns():
    report = ml_feature_significance(_panel(), CFG.ml, n_repeats=25, seed=7)
    assert report is not None
    assert set(report.index) == set(FEATURE_NAMES)
    assert list(report.columns) == ["importance", "p_value", "q_value", "keep"]
    assert ((report["p_value"] >= 0) & (report["p_value"] <= 1)).all()
    assert ((report["q_value"] >= 0) & (report["q_value"] <= 1)).all()
    assert report["keep"].dtype == bool


def test_keep_is_consistent_with_the_fdr_threshold():
    report = ml_feature_significance(_panel(), CFG.ml, alpha=0.10, n_repeats=25, seed=7)
    assert (report["keep"] == (report["q_value"] <= 0.10)).all()


def test_deterministic_for_a_fixed_seed():
    a = ml_feature_significance(_panel(), CFG.ml, n_repeats=25, seed=3)
    b = ml_feature_significance(_panel(), CFG.ml, n_repeats=25, seed=3)
    pd.testing.assert_frame_equal(a, b)
