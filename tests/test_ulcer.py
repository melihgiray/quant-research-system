"""Tests for the ulcer index and pain ratio."""

import numpy as np
import pandas as pd

from quant_system.risk.metrics import pain_ratio, ulcer_index


def _returns(values):
    idx = pd.bdate_range("2021-01-01", periods=len(values))
    return pd.Series(np.asarray(values, dtype=float), index=idx)


def test_ulcer_is_zero_with_no_drawdown():
    rising = _returns([0.01] * 50)                     # monotonically up, never underwater
    assert ulcer_index(rising) == 0.0


def test_ulcer_is_positive_with_a_drawdown():
    r = _returns([0.02, 0.02, -0.05, -0.05, 0.01])
    assert ulcer_index(r) > 0.0


def test_deeper_drawdown_has_a_larger_ulcer():
    shallow = _returns([0.02, -0.02, 0.02, 0.0])
    deep = _returns([0.02, -0.10, 0.02, 0.0])
    assert ulcer_index(deep) > ulcer_index(shallow)


def test_pain_ratio_is_positive_for_a_profitable_bumpy_series():
    r = _returns([0.03, -0.02, 0.03, -0.01, 0.03])
    assert pain_ratio(r) > 0


def test_pain_ratio_is_nan_without_a_drawdown():
    assert np.isnan(pain_ratio(_returns([0.01] * 20)))   # ulcer 0 -> undefined ratio
