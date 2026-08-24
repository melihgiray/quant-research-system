"""Tests for time-series statistics."""

import numpy as np

import pytest

from quant_system.stats import hurst_exponent, variance_ratio


def _random_walk(n=8000, seed=0):
    return np.cumsum(np.random.default_rng(seed).normal(0, 1, n))


def _persistent(n=20000, seed=1):
    rng = np.random.default_rng(seed)
    inc = np.zeros(n)
    for i in range(1, n):
        inc[i] = 0.7 * inc[i - 1] + rng.normal(0, 1)     # positively autocorrelated increments
    return np.cumsum(inc)


def _mean_reverting(n=8000, seed=2):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.4 * x[i - 1] + rng.normal(0, 1)          # OU-like, anti-persistent
    return x


def test_random_walk_has_hurst_near_half():
    assert abs(hurst_exponent(_random_walk()) - 0.5) < 0.06


def test_persistent_series_exceeds_half():
    assert hurst_exponent(_persistent()) > 0.55


def test_mean_reverting_series_is_below_half():
    assert hurst_exponent(_mean_reverting()) < 0.45


def test_variance_ratio_near_one_for_a_random_walk():
    assert abs(variance_ratio(_random_walk(), q=4) - 1.0) < 0.1


def test_variance_ratio_above_one_when_persistent():
    assert variance_ratio(_persistent(), q=4) > 1.2


def test_variance_ratio_below_one_when_mean_reverting():
    assert variance_ratio(_mean_reverting(), q=4) < 0.8


def test_variance_ratio_rejects_small_q():
    with pytest.raises(ValueError, match="q >= 2"):
        variance_ratio(_random_walk(), q=1)
