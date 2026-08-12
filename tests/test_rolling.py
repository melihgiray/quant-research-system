"""Tests for the rolling performance series."""

import numpy as np
import pandas as pd
import pytest

from quant_system.performance.rolling import rolling_beta, rolling_sharpe


def _series(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    return pd.Series(rng.normal(0.0005, 0.01, n), index=idx)


def test_rolling_sharpe_warmup_is_nan_then_defined():
    r = _series()
    rs = rolling_sharpe(r, window=100)
    assert rs.iloc[:99].isna().all()                     # not enough history yet
    assert rs.iloc[100:].notna().all()
    assert len(rs) == len(r)


def test_rolling_sharpe_sign_tracks_mean_return():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2020-01-01", periods=200)
    up = pd.Series(0.001 + rng.normal(0.0, 0.002, 200), index=idx)   # positive drift
    assert rolling_sharpe(up, window=60).dropna().mean() > 0


def test_rolling_beta_recovers_a_known_slope():
    bench = _series(seed=2)
    strat = 1.5 * bench                                   # exactly 1.5x the benchmark
    beta = rolling_beta(strat, bench, window=80).dropna()
    assert np.allclose(beta.to_numpy(), 1.5, atol=1e-6)


def test_rolling_beta_of_identity_is_one():
    bench = _series(seed=3)
    beta = rolling_beta(bench, bench, window=80).dropna()
    assert np.allclose(beta.to_numpy(), 1.0, atol=1e-9)


def test_rolling_beta_reindexes_to_strategy_dates():
    bench = _series(seed=4)
    strat = _series(seed=5)
    beta = rolling_beta(strat, bench, window=80)
    assert list(beta.index) == list(strat.index)
