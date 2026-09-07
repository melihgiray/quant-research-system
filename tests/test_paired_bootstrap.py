"""Tests for paired stationary-bootstrap performance comparisons."""

import numpy as np
import pandas as pd

from quant_system.performance.bootstrap import (
    paired_bootstrap_difference,
    paired_sharpe_difference_interval,
)


def _series(seed: int) -> pd.Series:
    return pd.Series(np.random.default_rng(seed).normal(0.0005, 0.01, 120),
                     index=pd.bdate_range("2020-01-01", periods=120))


def test_paired_bootstrap_is_exactly_zero_for_identical_streams():
    returns = _series(7)
    samples = paired_bootstrap_difference(returns, returns, np.mean, n_boot=100, seed=2)
    assert np.allclose(samples, 0.0)


def test_paired_sharpe_interval_contains_its_point_estimate():
    left, right = _series(3), _series(4)
    interval = paired_sharpe_difference_interval(left, right, n_boot=100, seed=9)
    assert interval.low <= interval.point <= interval.high
