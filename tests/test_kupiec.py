"""Tests for the Kupiec proportion-of-failures VaR backtest."""

import numpy as np
import pandas as pd

from quant_system.risk.metrics import historical_var, kupiec_pof_test


def test_calibrated_var_is_not_rejected():
    r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 5000))
    var = historical_var(r, 0.95)                        # by construction ~5% breaches
    _, p = kupiec_pof_test(r, var, level=0.95)
    assert p > 0.05                                      # do not reject a calibrated VaR


def test_too_optimistic_var_is_rejected():
    r = pd.Series(np.random.default_rng(1).normal(0, 0.01, 5000))
    _, p = kupiec_pof_test(r, 0.002, level=0.95)         # far too small -> many breaches
    assert p < 0.01


def test_too_conservative_var_is_rejected():
    r = pd.Series(np.random.default_rng(2).normal(0, 0.01, 5000))
    _, p = kupiec_pof_test(r, 0.10, level=0.95)          # far too large -> ~no breaches
    assert p < 0.01
