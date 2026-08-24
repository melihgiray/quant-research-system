"""Tests for the tail ratio."""

import numpy as np
import pandas as pd

from quant_system.risk.metrics import tail_ratio


def test_symmetric_returns_have_tail_ratio_near_one():
    r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 20000))
    assert abs(tail_ratio(r) - 1.0) < 0.1


def test_right_skew_exceeds_one():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.exponential(0.01, 20000) - 0.01)   # long right tail
    assert tail_ratio(r) > 1.2


def test_left_skew_below_one():
    rng = np.random.default_rng(2)
    r = pd.Series(0.01 - rng.exponential(0.01, 20000))   # long left tail
    assert tail_ratio(r) < 0.85
