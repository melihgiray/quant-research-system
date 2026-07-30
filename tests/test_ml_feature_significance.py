"""Tests for permutation-null ML feature significance."""

import numpy as np
from sklearn.linear_model import LogisticRegression

from quant_system.signals.ml_signal import permutation_importance_pvalues


def test_permutation_null_separates_signal_from_noise():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(1_200, 2))
    y = (2.5 * X[:, 0] + rng.normal(size=len(X)) > 0).astype(int)
    model = LogisticRegression().fit(X[:800], y[:800])

    result = permutation_importance_pvalues(
        model,
        X[800:],
        y[800:],
        feature_names=["signal", "noise"],
        n_repeats=99,
        seed=13,
    )

    assert result.loc["signal", "importance"] > result.loc["noise", "importance"]
    assert result.loc["signal", "p_value"] <= 0.05
    assert result.loc["noise", "p_value"] > 0.05


def test_permutation_null_is_deterministic():
    rng = np.random.default_rng(21)
    X = rng.normal(size=(300, 2))
    y = (X[:, 0] > 0).astype(int)
    model = LogisticRegression().fit(X, y)

    first = permutation_importance_pvalues(model, X, y, n_repeats=19, seed=5)
    second = permutation_importance_pvalues(model, X, y, n_repeats=19, seed=5)

    assert first.equals(second)
