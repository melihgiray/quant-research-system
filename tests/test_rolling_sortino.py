"""Tests for rolling Sortino."""

import numpy as np
import pandas as pd

from quant_system.performance.rolling import rolling_sortino


def _series(n=400, seed=0, drift=0.0005):
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(drift + np.random.default_rng(seed).normal(0, 0.005, n), index=idx)


def test_warmup_is_nan_and_length_matches():
    r = _series()
    s = rolling_sortino(r, window=100)
    assert s.iloc[:99].isna().all()
    assert len(s) == len(r)


def test_positive_drift_gives_positive_sortino():
    s = rolling_sortino(_series(drift=0.001), window=60)
    assert s.dropna().mean() > 0


def test_only_downside_penalised():
    # A series with big up days but tiny down days should have a high Sortino.
    idx = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(1)
    r = pd.Series(np.where(rng.random(300) < 0.5, 0.03, -0.001), index=idx)
    assert rolling_sortino(r, window=60).dropna().mean() > 1.0
