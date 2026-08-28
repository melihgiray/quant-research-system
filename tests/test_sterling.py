"""Tests for the Sterling ratio."""

import numpy as np
import pandas as pd

from quant_system.risk.metrics import sterling_ratio


def _returns(values):
    return pd.Series(np.asarray(values, dtype=float))


def test_no_drawdown_is_nan():
    assert np.isnan(sterling_ratio(_returns([0.01] * 30)))


def test_profitable_bumpy_series_is_positive():
    assert sterling_ratio(_returns([0.03, -0.02, 0.03, -0.01, 0.03])) > 0


def test_deeper_drawdowns_lower_the_ratio():
    shallow = _returns([0.02, -0.02, 0.02, 0.0, 0.02])
    deep = _returns([0.02, -0.12, 0.02, 0.0, 0.02])
    assert sterling_ratio(deep) < sterling_ratio(shallow)
