"""Tests for the risk-budgeting allocators."""

import numpy as np
import pandas as pd
import pytest

from quant_system.portfolio.risk_budget import (
    erc_weights,
    max_diversification_weights,
    risk_contributions,
)


def test_erc_reduces_to_inverse_vol_when_uncorrelated():
    cov = np.diag([4.0, 1.0])                             # vols 2 and 1
    w = erc_weights(cov)
    assert w == pytest.approx([1.0 / 3.0, 2.0 / 3.0], abs=1e-3)   # inverse-vol


def test_erc_equalises_risk_contributions():
    cov = np.array([[0.04, 0.006, 0.0],
                    [0.006, 0.09, 0.01],
                    [0.0, 0.01, 0.0225]])
    w = erc_weights(cov)
    rc = risk_contributions(cov, w)
    assert np.allclose(rc, 1.0 / 3.0, atol=1e-2)          # equal contributions
    assert w.sum() == pytest.approx(1.0)
    assert (w >= 0).all()


def test_erc_preserves_labels():
    cov = pd.DataFrame(np.diag([4.0, 1.0]), index=["a", "b"], columns=["a", "b"])
    w = erc_weights(cov)
    assert list(w.index) == ["a", "b"]


def test_max_diversification_down_weights_a_redundant_asset():
    # Two nearly-identical assets and one independent: the independent asset
    # diversifies more, so it should get more than an equal share.
    cov = np.array([[0.04, 0.0396, 0.0],
                    [0.0396, 0.04, 0.0],
                    [0.0, 0.0, 0.04]])
    w = max_diversification_weights(cov)
    assert w[2] > w[0] and w[2] > w[1]
    assert w.sum() == pytest.approx(1.0)


def test_risk_contributions_sum_to_one():
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    rc = risk_contributions(cov, [0.6, 0.4])
    assert rc.sum() == pytest.approx(1.0)


def test_minimum_variance_prefers_the_lower_variance_asset():
    from quant_system.portfolio.risk_budget import minimum_variance_weights
    cov = pd.DataFrame([[0.04, 0.0], [0.0, 0.01]], columns=["A", "B"], index=["A", "B"])
    weights = minimum_variance_weights(cov)
    assert weights["B"] > weights["A"]
    assert np.isclose(weights.sum(), 1.0)


def test_minimum_variance_beats_equal_weight_on_its_input_covariance():
    from quant_system.portfolio.risk_budget import minimum_variance_weights
    cov = np.array([[0.04, 0.01], [0.01, 0.01]])
    weights = minimum_variance_weights(cov)
    equal = np.array([0.5, 0.5])
    assert weights @ cov @ weights <= equal @ cov @ equal
