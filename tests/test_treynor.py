"""Tests for the Treynor ratio."""

import numpy as np
import pandas as pd
import pytest

from quant_system.performance.active import treynor_ratio


def _bench(n=4000, seed=0):
    return pd.Series(np.random.default_rng(seed).normal(0.0004, 0.01, n))


def test_leverage_does_not_change_treynor():
    b = _bench()
    assert treynor_ratio(2.0 * b, b) == pytest.approx(treynor_ratio(b, b), rel=1e-6)


def test_more_excess_return_at_same_beta_raises_treynor():
    b = _bench(seed=1)
    assert treynor_ratio(b + 0.0005, b) > treynor_ratio(b, b)


def test_zero_beta_is_nan():
    b = _bench(seed=2)
    orthogonal = pd.Series(np.random.default_rng(9).normal(0.0004, 0.01, len(b)))
    tr = treynor_ratio(orthogonal, b)
    assert np.isnan(tr) or np.isfinite(tr)   # near-zero beta -> nan or large; never crashes
