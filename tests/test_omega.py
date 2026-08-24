"""Tests for the Omega ratio."""

import numpy as np
import pandas as pd

from quant_system.risk.metrics import omega_ratio


def _returns(values):
    return pd.Series(np.asarray(values, dtype=float))


def test_omega_is_one_for_symmetric_returns_at_zero():
    r = _returns([0.02, -0.02, 0.02, -0.02])
    assert omega_ratio(r, threshold=0.0) == 1.0


def test_omega_above_one_when_upside_dominates():
    r = _returns([0.03, 0.03, -0.01, 0.02, -0.01])
    assert omega_ratio(r, threshold=0.0) > 1.0


def test_raising_the_threshold_lowers_omega():
    rng = np.random.default_rng(0)
    r = _returns(rng.normal(0.001, 0.01, 5000))
    assert omega_ratio(r, threshold=0.0) > omega_ratio(r, threshold=0.005)


def test_no_shortfall_gives_infinite_omega():
    assert np.isinf(omega_ratio(_returns([0.01, 0.02, 0.03]), threshold=0.0))
