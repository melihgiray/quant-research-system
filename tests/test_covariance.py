"""Tests for Ledoit-Wolf covariance shrinkage."""

import numpy as np
import pandas as pd

from quant_system.risk.covariance import ledoit_wolf_shrinkage


def _correlated(T, N=15, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1, (N, N))
    true = a @ a.T / N + np.eye(N) * 0.1
    chol = np.linalg.cholesky(true)
    return rng.normal(0, 1, (T, N)) @ chol.T


def test_shrinkage_is_a_valid_intensity():
    _, s = ledoit_wolf_shrinkage(_correlated(50))
    assert 0.0 <= s <= 1.0


def test_more_data_shrinks_less():
    _, s_small = ledoit_wolf_shrinkage(_correlated(30, seed=1))
    _, s_large = ledoit_wolf_shrinkage(_correlated(4000, seed=1))
    assert s_large < s_small                            # sample covariance trusted with more data


def test_shrunk_matrix_is_symmetric_and_better_conditioned_when_data_scarce():
    X = _correlated(20, N=15, seed=2)
    shrunk, _ = ledoit_wolf_shrinkage(X)
    Xc = X - X.mean(axis=0)
    sample = Xc.T @ Xc / len(X)
    assert np.allclose(shrunk, shrunk.T)
    assert np.linalg.cond(shrunk) < np.linalg.cond(sample)


def test_labels_are_preserved_for_a_dataframe():
    df = pd.DataFrame(_correlated(60, N=4), columns=["a", "b", "c", "d"])
    shrunk, _ = ledoit_wolf_shrinkage(df)
    assert list(shrunk.columns) == ["a", "b", "c", "d"]
