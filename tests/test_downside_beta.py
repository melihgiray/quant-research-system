"""Tests for downside beta."""

import numpy as np
import pandas as pd
import pytest

from quant_system.performance.active import alpha_beta, downside_beta


def _bench(n=6000, seed=0):
    return pd.Series(np.random.default_rng(seed).normal(0.0, 0.01, n))


def test_leveraged_benchmark_has_that_downside_beta():
    b = _bench()
    assert downside_beta(1.5 * b, b) == pytest.approx(1.5, abs=1e-6)


def test_defensive_strategy_has_lower_downside_beta_than_overall():
    b = _bench(seed=1)
    strat = b.where(b > b.mean(), b * 0.3)   # cut participation on down days
    _, beta = alpha_beta(strat, b)
    assert downside_beta(strat, b) < beta
