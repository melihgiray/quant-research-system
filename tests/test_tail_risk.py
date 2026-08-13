"""Tests for Cornish-Fisher, EVT tail risk, and stress scenarios."""

import numpy as np
import pandas as pd
import pytest

from quant_system.risk.metrics import cornish_fisher_var, parametric_var


def _normal(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0, 0.01, n))


def _left_skewed_fat(n=5000, seed=0):
    # Student-t (fat tails) with an added negative jump component (left skew).
    rng = np.random.default_rng(seed)
    body = rng.standard_t(4, n) * 0.008
    jumps = np.where(rng.random(n) < 0.03, -rng.gamma(2.0, 0.02, n), 0.0)
    return pd.Series(body + jumps)


def test_cornish_fisher_matches_gaussian_on_normal_data():
    r = _normal()
    assert cornish_fisher_var(r, 0.95) == pytest.approx(parametric_var(r, 0.95), rel=0.1)


def test_cornish_fisher_exceeds_gaussian_on_fat_left_tail():
    r = _left_skewed_fat()
    assert cornish_fisher_var(r, 0.99) > parametric_var(r, 0.99)   # fat tail -> bigger loss


def test_cornish_fisher_is_a_positive_loss():
    assert cornish_fisher_var(_left_skewed_fat(), 0.95) > 0
