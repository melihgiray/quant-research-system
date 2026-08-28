"""Tests for CAPM alpha/beta."""

import numpy as np
import pandas as pd
import pytest

from quant_system.performance.active import alpha_beta


def _bench(n=4000, seed=0):
    return pd.Series(np.random.default_rng(seed).normal(0.0003, 0.01, n))


def test_leveraged_benchmark_has_that_beta_and_zero_alpha():
    b = _bench()
    alpha, beta = alpha_beta(2.0 * b, b)
    assert beta == pytest.approx(2.0, abs=1e-6)
    assert abs(alpha) < 1e-6


def test_constant_outperformance_is_pure_alpha():
    b = _bench(seed=1)
    alpha, beta = alpha_beta(b + 0.001, b)
    assert beta == pytest.approx(1.0, abs=1e-6)
    assert alpha == pytest.approx(0.001 * 252, rel=1e-6)


def test_positive_alpha_with_partial_beta():
    b = _bench(seed=2)
    rng = np.random.default_rng(3)
    strat = 0.5 * b + pd.Series(rng.normal(0.0002, 0.002, len(b)))
    alpha, beta = alpha_beta(strat, b)
    assert 0.3 < beta < 0.7
    assert alpha > 0
