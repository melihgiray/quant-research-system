"""Tests for Day-2 stationary-bootstrap confidence intervals."""

import numpy as np
import pandas as pd

from quant_system.performance.bootstrap import (
    sharpe_confidence_interval, cagr_confidence_interval,
    bootstrap_distribution, bootstrap_summary, _ann_sharpe,
)


def _series(mean, std, n, seed=0):
    rng = np.random.default_rng(seed)
    r = rng.normal(mean, std, n)
    return pd.Series(r, index=pd.bdate_range("2015-01-01", periods=n))


def test_ci_brackets_point_estimate():
    r = _series(0.0003, 0.01, 1500, seed=1)
    ci = sharpe_confidence_interval(r, level=0.95, n_boot=500)
    assert ci.low <= ci.point <= ci.high
    assert ci.high > ci.low


def test_strong_series_ci_excludes_zero():
    r = _series(0.0012, 0.006, 2000, seed=2)        # very high Sharpe
    ci = sharpe_confidence_interval(r, level=0.95, n_boot=500)
    assert ci.low > 0                                # significant at 95%


def test_noise_series_ci_includes_zero():
    r = _series(0.0, 0.01, 1500, seed=3)
    ci = sharpe_confidence_interval(r, level=0.95, n_boot=500)
    assert ci.low < 0 < ci.high                      # cannot reject SR=0


def test_bootstrap_is_deterministic_with_seed():
    r = _series(0.0005, 0.01, 1000, seed=4)
    a = bootstrap_distribution(r, _ann_sharpe, n_boot=300, seed=11)
    b = bootstrap_distribution(r, _ann_sharpe, n_boot=300, seed=11)
    assert np.array_equal(a, b)


def test_summary_shape_and_short_series():
    r = _series(0.0005, 0.01, 800, seed=5)
    lines = bootstrap_summary(r, n_boot=200)
    assert len(lines) == 4 and "BOOTSTRAP" in lines[0]
    short = _series(0.001, 0.01, 5, seed=6)          # too short -> empty dist
    ci = cagr_confidence_interval(short, n_boot=200)
    assert np.isnan(ci.low)
