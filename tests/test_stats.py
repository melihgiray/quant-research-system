"""Tests for time-series statistics."""

import numpy as np

from quant_system.stats import hurst_exponent


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
