"""Tests for Day-1 Sharpe-significance tools (PSR / DSR / minTRL)."""

import math
import numpy as np
import pandas as pd
import pytest

from quant_system.performance.significance import (
    probabilistic_sharpe_ratio,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    min_track_record_length,
    significance_summary,
    sharpe_moments,
)


def _series(mean, std, n, seed=0):
    rng = np.random.default_rng(seed)
    r = rng.normal(mean, std, n)
    r = r - r.mean() + mean                      # pin the realised mean exactly
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.Series(r, index=idx)


def test_psr_high_for_strong_track_record():
    # Daily SR ~0.1 over 2000 days is overwhelmingly significant.
    r = _series(0.0008, 0.008, 2000, seed=1)
    assert probabilistic_sharpe_ratio(r, benchmark_sr_annual=0.0) > 0.99


def test_psr_near_half_for_zero_mean():
    r = _series(0.0, 0.01, 1500, seed=2)
    psr = probabilistic_sharpe_ratio(r, benchmark_sr_annual=0.0)
    assert 0.4 < psr < 0.6                        # SR ~ 0 -> coin flip


def test_expected_max_sharpe_grows_with_trials():
    v = 0.001
    assert expected_max_sharpe(v, 2) < expected_max_sharpe(v, 50) < expected_max_sharpe(v, 1000)
    assert expected_max_sharpe(v, 1) == 0.0       # a single trial needs no deflation


def test_deflation_reduces_significance():
    r = _series(0.0005, 0.009, 1500, seed=3)
    psr = probabilistic_sharpe_ratio(r, 0.0)
    dsr = deflated_sharpe_ratio(r, n_trials=100)["dsr"]
    assert dsr <= psr                              # searching 100 configs lowers the bar-clearing prob
    assert deflated_sharpe_ratio(r, n_trials=100)["sr_star_annual"] > 0


def test_min_trl_significant_for_long_strong_series():
    r = _series(0.0009, 0.008, 2500, seed=4)
    trl = min_track_record_length(r, benchmark_sr_annual=0.0, confidence=0.95)
    assert trl["significant_now"] is True
    assert math.isfinite(trl["min_years"])


def test_min_trl_infinite_when_sharpe_below_benchmark():
    r = _series(-0.0002, 0.01, 1000, seed=5)       # negative Sharpe
    trl = min_track_record_length(r, benchmark_sr_annual=0.0)
    assert trl["min_periods"] == float("inf")
    assert trl["significant_now"] is False


def test_summary_shape_and_nan_safety():
    r = _series(0.0005, 0.01, 1200, seed=6)
    lines = significance_summary(r, n_trials=10)
    assert len(lines) == 4 and "SIGNIFICANCE" in lines[0]
    # Degenerate input must not raise.
    tiny = pd.Series([0.01, 0.01], index=pd.bdate_range("2020-01-01", periods=2))
    m = sharpe_moments(tiny)
    assert math.isnan(m.sr)
    assert math.isnan(probabilistic_sharpe_ratio(tiny))
