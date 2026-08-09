"""Tests for the ML model factory: pipeline, calibration, sample weights."""

import numpy as np
import pytest

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from quant_system.signals.ml_signal import _fit_model, build_classifier


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


def test_fit_model_routes_sample_weight_to_the_estimator():
    # Non-uniform sample weights must change the fit; equal weights must not (a
    # no-op reweight leaves the model identical to the unweighted fit).
    X, y = _toy(seed=5)
    n = len(y)
    unweighted = _fit_model(build_classifier(seed=5), X, y)
    equal = _fit_model(build_classifier(seed=5), X, y, sample_weight=np.ones(n))
    skewed = np.where(y > 0, 3.0, 0.5)                    # emphasise the positive rows
    reweighted = _fit_model(build_classifier(seed=5), X, y, sample_weight=skewed)
    p_unweighted = unweighted.predict_proba(X)[:, 1]
    assert np.allclose(p_unweighted, equal.predict_proba(X)[:, 1])
    assert not np.allclose(p_unweighted, reweighted.predict_proba(X)[:, 1])


def test_calibration_produces_valid_probabilities_matching_base_rate():
    # A calibrated classifier's mean predicted probability on held-out data should
    # track the observed positive rate (calibration in the large).
    X, y = _toy(n=1500, seed=2)
    cut = 1000
    model = build_classifier(seed=3, calibrate="isotonic", cv=3)
    model.fit(X[:cut], y[:cut])
    p = model.predict_proba(X[cut:])[:, 1]
    assert p.min() >= 0.0 and p.max() <= 1.0
    assert abs(p.mean() - y[cut:].mean()) < 0.05          # predicted rate ~ actual rate
