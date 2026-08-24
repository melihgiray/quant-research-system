"""Tests for benchmark-relative metrics."""

import numpy as np
import pandas as pd

from quant_system.performance.active import information_ratio, tracking_error


def _bench(n=3000, seed=0):
    return pd.Series(np.random.default_rng(seed).normal(0.0003, 0.01, n))


def test_tracking_the_benchmark_gives_zero_tracking_error():
    b = _bench()
    assert tracking_error(b, b) == 0.0
    assert np.isnan(information_ratio(b, b))              # zero active risk -> undefined


def test_constant_alpha_gives_positive_information_ratio():
    b = _bench()
    strat = b + 0.001                                     # pure constant outperformance
    assert information_ratio(strat, b) > 0
    assert tracking_error(strat, b) == 0.0 or tracking_error(strat, b) < 1e-9


def test_noisy_positive_alpha_scores_positive_but_finite():
    b = _bench(seed=1)
    rng = np.random.default_rng(2)
    strat = b + pd.Series(rng.normal(0.0002, 0.003, len(b)))
    ir = information_ratio(strat, b)
    assert ir > 0 and np.isfinite(ir)
    assert tracking_error(strat, b) > 0
