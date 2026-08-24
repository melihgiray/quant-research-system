"""Tests for fractional differentiation."""

import numpy as np
import pandas as pd
import pytest

from quant_system.signals.frac_diff import ffd_weights, frac_diff_ffd, min_ffd_order


def _random_walk(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(np.cumsum(rng.normal(0.0, 1.0, n)))


def test_weights_reduce_to_a_first_difference_at_d_one():
    assert np.allclose(ffd_weights(1.0), [1.0, -1.0])


def test_weights_are_the_identity_at_d_zero():
    assert np.allclose(ffd_weights(0.0), [1.0])


def test_weights_follow_the_recursion():
    w = ffd_weights(0.4)
    assert w[0] == 1.0
    assert w[1] == pytest.approx(-0.4)                    # -w0*(d-0)/1
    assert w[2] == pytest.approx(-w[1] * (0.4 - 1) / 2)


def test_full_difference_matches_numpy_diff():
    rw = _random_walk()
    fd = frac_diff_ffd(rw, 1.0).dropna()
    assert np.allclose(fd.to_numpy(), np.diff(rw.to_numpy()))


def test_zero_order_returns_the_original():
    rw = _random_walk()
    fd = frac_diff_ffd(rw, 0.0)
    assert np.allclose(fd.to_numpy(), rw.to_numpy())


def test_fractional_order_stationarises_while_keeping_memory():
    # The classic result: a small d makes a random walk stationary while staying
    # highly correlated with the level, unlike a full first difference.
    rw = _random_walk()
    found = min_ffd_order(rw)
    assert found is not None
    d, pvalue = found
    assert 0.0 < d < 1.0
    assert pvalue < 0.05
    fd_frac = frac_diff_ffd(rw, d).dropna()
    fd_full = frac_diff_ffd(rw, 1.0).dropna()
    corr_frac = np.corrcoef(fd_frac.to_numpy(), rw.to_numpy()[-len(fd_frac):])[0, 1]
    corr_full = np.corrcoef(fd_full.to_numpy(), rw.to_numpy()[-len(fd_full):])[0, 1]
    assert corr_frac > corr_full                          # fractional retains more memory
    assert corr_frac > 0.5


def test_min_order_returns_none_for_already_stationary_noise_below_grid():
    # White noise is stationary at d=0, so the minimum passing order is 0.0.
    rng = np.random.default_rng(1)
    noise = pd.Series(rng.normal(0.0, 1.0, 1000))
    found = min_ffd_order(noise)
    assert found is not None
    assert found[0] == 0.0
