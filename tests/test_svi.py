"""Tests for the raw-SVI smile parameterisation."""

import numpy as np
import pytest

from quant_system.options.svi import (
    SVIParams,
    fit_svi_slice,
    svi_total_variance,
)


def _params():
    return SVIParams(a=0.02, b=0.20, rho=-0.4, m=0.05, sigma=0.15)


def test_vertex_value_at_k_equals_m():
    p = _params()
    w = svi_total_variance(p.m, p)
    assert w == pytest.approx(p.a + p.b * p.sigma)       # sqrt(sigma^2)=sigma, skew term 0


def test_wings_rise_away_from_the_vertex():
    p = _params()
    w_mid = svi_total_variance(p.m, p)
    assert svi_total_variance(p.m + 2.0, p) > w_mid
    assert svi_total_variance(p.m - 2.0, p) > w_mid


def test_wing_slopes_are_b_times_one_plus_minus_rho():
    p = _params()
    k = np.array([8.0, 9.0])                             # deep in the right wing
    slope = np.diff(svi_total_variance(k, p))[0]
    assert slope == pytest.approx(p.b * (1 + p.rho), abs=1e-3)


def test_vectorises_over_k():
    p = _params()
    out = svi_total_variance(np.linspace(-1, 1, 11), p)
    assert out.shape == (11,)
    assert (out > 0).all()


def test_fit_recovers_a_known_smile():
    p = _params()
    k = np.linspace(-0.6, 0.6, 21)
    w = svi_total_variance(k, p)
    fitted, rmse = fit_svi_slice(k, w)
    assert rmse < 1e-6                                   # the curve is recovered exactly
    assert np.allclose(svi_total_variance(k, fitted), w, atol=1e-5)


def test_fit_is_robust_to_small_noise():
    p = _params()
    k = np.linspace(-0.6, 0.6, 25)
    rng = np.random.default_rng(0)
    w = svi_total_variance(k, p) + rng.normal(0.0, 1e-4, k.size)
    _, rmse = fit_svi_slice(k, w)
    assert rmse < 5e-4                                   # fit tracks the noisy smile closely


def test_fit_enforces_parameter_bounds():
    p = _params()
    k = np.linspace(-0.5, 0.5, 15)
    w = svi_total_variance(k, p)
    fitted, _ = fit_svi_slice(k, w)
    assert fitted.b >= 0.0
    assert -1.0 < fitted.rho < 1.0
    assert fitted.sigma > 0.0


def test_fit_needs_enough_points():
    with pytest.raises(ValueError, match="at least 5 points"):
        fit_svi_slice(np.array([0.0, 0.1, 0.2]), np.array([0.04, 0.03, 0.05]))
