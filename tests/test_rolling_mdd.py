"""Tests for rolling max drawdown."""

import numpy as np
import pandas as pd

from quant_system.performance.rolling import rolling_max_drawdown


def _series(values):
    idx = pd.bdate_range("2021-01-01", periods=len(values))
    return pd.Series(np.asarray(values, dtype=float), index=idx)


def test_warmup_is_nan_and_length_matches():
    r = _series(np.random.default_rng(0).normal(0, 0.01, 200))
    out = rolling_max_drawdown(r, window=50)
    assert out.iloc[:49].isna().all()
    assert len(out) == len(r)


def test_monotonic_gains_have_zero_drawdown():
    out = rolling_max_drawdown(_series([0.01] * 60), window=20)
    assert (out.dropna() == 0.0).all()


def test_a_drop_shows_up_as_negative():
    r = _series([0.01, 0.01, -0.20, 0.01, 0.01, 0.01])
    out = rolling_max_drawdown(r, window=4)
    assert out.dropna().min() <= -0.20 + 1e-9
