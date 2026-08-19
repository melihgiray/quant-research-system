"""Tests for the SSVI arbitrage-free parameterisation."""

import numpy as np
import pytest

from quant_system.options.ssvi import (
    SSVIParams,
    fit_ssvi_slice,
    ssvi_butterfly_free,
    ssvi_to_svi_params,
    ssvi_total_variance,
)
from quant_system.options.svi import (
    SVIParams,
    is_butterfly_arbitrage_free,
    svi_total_variance,
)


def test_ssvi_reduces_to_the_equivalent_raw_svi():
    p = SSVIParams(theta=0.04, rho=-0.4, psi=1.5)
    k = np.linspace(-0.6, 0.6, 21)
    assert np.allclose(ssvi_total_variance(k, p), svi_total_variance(k, ssvi_to_svi_params(p)))


def test_atm_total_variance_is_theta():
    p = SSVIParams(theta=0.05, rho=-0.3, psi=2.0)
    assert ssvi_total_variance(0.0, p) == pytest.approx(p.theta)   # w(0) = theta


def test_ssvi_is_positive_and_vectorised():
    p = SSVIParams(theta=0.03, rho=0.2, psi=1.0)
    out = ssvi_total_variance(np.linspace(-1, 1, 15), p)
    assert out.shape == (15,)
    assert (out > 0).all()


def test_conditions_hold_implies_the_slice_is_arbitrage_free():
    p = SSVIParams(theta=0.09, rho=-0.3, psi=2.0)          # theta*psi^2*(1+|rho|)=0.47 <= 4
    assert ssvi_butterfly_free(p)
    assert is_butterfly_arbitrage_free(ssvi_to_svi_params(p))[0]   # sufficiency confirmed via g(k)


def test_conditions_flag_a_violating_slice():
    bad = SSVIParams(theta=0.09, rho=-0.9, psi=8.0)        # theta*psi^2*(1+|rho|)=10.9 > 4
    assert not ssvi_butterfly_free(bad)
    assert not is_butterfly_arbitrage_free(ssvi_to_svi_params(bad))[0]


def test_conditions_reject_degenerate_parameters():
    assert not ssvi_butterfly_free(SSVIParams(theta=-0.01, rho=0.0, psi=1.0))
    assert not ssvi_butterfly_free(SSVIParams(theta=0.04, rho=0.0, psi=-1.0))


def test_fit_recovers_a_known_ssvi_slice():
    true = SSVIParams(theta=0.05, rho=-0.35, psi=1.8)
    k = np.linspace(-0.6, 0.6, 21)
    fit, rmse = fit_ssvi_slice(k, ssvi_total_variance(k, true))
    assert rmse < 1e-6
    assert ssvi_butterfly_free(fit)
    assert fit.theta == pytest.approx(true.theta, abs=1e-3)
    assert fit.psi == pytest.approx(true.psi, abs=1e-2)


def test_fit_to_arbitrageable_data_stays_arbitrage_free():
    # SSVI cannot represent an arbitrageable smile, so fitting one yields a clean
    # slice at the cost of RMSE, rather than reproducing the violation.
    bad = SVIParams(a=0.01, b=0.9, rho=-0.95, m=0.0, sigma=0.02)
    k = np.linspace(-0.4, 0.4, 25)
    fit, rmse = fit_ssvi_slice(k, svi_total_variance(k, bad))
    assert ssvi_butterfly_free(fit)
    assert is_butterfly_arbitrage_free(ssvi_to_svi_params(fit))[0]
    assert rmse > 1e-3                                    # it could not fit the arb, as expected


def test_fit_needs_enough_points():
    with pytest.raises(ValueError, match="at least 5 points"):
        fit_ssvi_slice(np.array([0.0, 0.1, 0.2]), np.array([0.04, 0.03, 0.05]))
