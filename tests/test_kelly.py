"""Tests for the growth-optimal (Kelly) fraction."""

import numpy as np
import pandas as pd
import pytest

from quant_system.risk.sizing import growth_optimal_fraction


def test_kelly_equals_mean_over_variance():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.02, 100000))
    assert growth_optimal_fraction(r) == pytest.approx(r.mean() / r.var(), rel=1e-9)


def test_kelly_rises_with_edge_and_falls_with_variance():
    idx = pd.RangeIndex(50000)
    rng = np.random.default_rng(1)
    base = pd.Series(rng.normal(0.0005, 0.01, 50000), index=idx)
    more_edge = base + 0.0005
    assert growth_optimal_fraction(more_edge) > growth_optimal_fraction(base)
    noisier = pd.Series(base.to_numpy() * 3.0, index=idx)      # 3x scale -> ~1/3 the fraction
    assert growth_optimal_fraction(noisier) < growth_optimal_fraction(base)


def test_zero_variance_is_nan():
    assert np.isnan(growth_optimal_fraction(pd.Series([0.01, 0.01, 0.01])))
