"""Tests for the raw-SVI smile parameterisation."""

import numpy as np
import pytest

import pandas as pd

from quant_system.options.svi import (
    SVIParams,
    durability_g,
    fit_svi_points,
    fit_svi_slice,
    is_butterfly_arbitrage_free,
    svi_derivatives,
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


def test_analytic_derivatives_match_finite_differences():
    p = _params()
    k = 0.13
    h = 1e-5
    _, w1, w2 = svi_derivatives(k, p)
    fd1 = (svi_total_variance(k + h, p) - svi_total_variance(k - h, p)) / (2 * h)
    fd2 = (svi_total_variance(k + h, p) - 2 * svi_total_variance(k, p)
           + svi_total_variance(k - h, p)) / h ** 2
    assert w1 == pytest.approx(fd1, abs=1e-5)
    assert w2 == pytest.approx(fd2, abs=1e-3)


def test_ordinary_slice_is_arbitrage_free():
    ok, min_g = is_butterfly_arbitrage_free(_params())
    assert ok
    assert min_g > 0


def test_pathological_slice_has_butterfly_arbitrage():
    # A very steep, sharp smile drives the risk-neutral density negative.
    bad = SVIParams(a=0.01, b=0.9, rho=-0.95, m=0.0, sigma=0.02)
    ok, min_g = is_butterfly_arbitrage_free(bad)
    assert not ok
    assert min_g < 0
    assert (durability_g(np.linspace(-2, 2, 401), bad) < 0).any()


def _points_two_expiries():
    near = SVIParams(a=0.010, b=0.10, rho=-0.30, m=0.02, sigma=0.10)
    far = SVIParams(a=0.030, b=0.18, rho=-0.35, m=0.03, sigma=0.14)
    k = np.linspace(-0.5, 0.5, 15)
    frames = []
    for tte, p in ((0.08, near), (0.50, far)):
        frames.append(pd.DataFrame({
            "time_to_expiry": tte,
            "log_moneyness": k,
            "total_variance": svi_total_variance(k, p),
        }))
    return pd.concat(frames, ignore_index=True)


def test_fit_points_returns_a_row_per_expiry():
    result = fit_svi_points(_points_two_expiries())
    assert len(result) == 2
    assert set(["a", "b", "rho", "m", "sigma", "rmse", "arb_free", "min_g"]).issubset(result.columns)
    assert (result["rmse"] < 1e-5).all()                # each smile is recovered
    assert result["arb_free"].all()


def test_fit_points_skips_thin_expiries():
    pts = _points_two_expiries()
    thin = pd.DataFrame({"time_to_expiry": 1.0, "log_moneyness": [0.0, 0.1, 0.2],
                         "total_variance": [0.04, 0.03, 0.05]})
    result = fit_svi_points(pd.concat([pts, thin], ignore_index=True))
    assert set(result["time_to_expiry"]) == {0.08, 0.50}   # the 3-point expiry is dropped
