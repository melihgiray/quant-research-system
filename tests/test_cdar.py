"""Tests for conditional drawdown at risk."""

import numpy as np
import pandas as pd

from quant_system.risk.metrics import conditional_drawdown_at_risk, max_drawdown


def _returns(values):
    return pd.Series(np.asarray(values, dtype=float))


def test_cdar_zero_without_drawdown():
    assert conditional_drawdown_at_risk(_returns([0.01] * 30)) == 0.0


def test_cdar_positive_and_within_max_drawdown():
    rng = np.random.default_rng(0)
    r = _returns(rng.normal(0.0, 0.02, 2000))
    cdar = conditional_drawdown_at_risk(r, level=0.95)
    assert cdar > 0
    assert cdar <= abs(max_drawdown(r)) + 1e-9           # tail average is at most the worst point


def test_deeper_drawdowns_raise_cdar():
    shallow = _returns([0.02, -0.03, 0.02, 0.0, 0.01])
    deep = _returns([0.02, -0.15, 0.02, 0.0, 0.01])
    assert conditional_drawdown_at_risk(deep) > conditional_drawdown_at_risk(shallow)
