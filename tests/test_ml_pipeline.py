"""Tests for the ML model factory: pipeline, calibration, sample weights."""

import numpy as np
import pytest

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from quant_system.signals.ml_signal import build_classifier


def _toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 8))
    # A learnable but noisy relationship so probabilities are not degenerate.
    logit = 0.8 * X[:, 0] - 0.5 * X[:, 3] + rng.normal(scale=0.5, size=n)
    y = (logit > 0).astype(float)
    return X, y


def test_factory_returns_a_pipeline_with_proba():
    model = build_classifier(seed=1)
    assert isinstance(model, Pipeline)
    assert "clf" in model.named_steps
    X, y = _toy()
    model.fit(X, y)
    p = model.predict_proba(X)[:, 1]
    assert p.min() >= 0.0 and p.max() <= 1.0


def test_pipeline_matches_bare_estimator():
    # The single-step pipeline must be a behaviour-preserving wrapper, so the
    # documented backtest numbers do not move under the refactor.
    X, y = _toy()
    bare = HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=300,
        l2_regularization=1.0, random_state=7,
    ).fit(X, y)
    piped = build_classifier(seed=7).fit(X, y)
    assert np.allclose(bare.predict_proba(X)[:, 1], piped.predict_proba(X)[:, 1])


def test_invalid_calibration_method_rejected():
    with pytest.raises(ValueError, match="calibrate must be"):
        build_classifier(calibrate="bogus")
