"""Tests for the SSVI arbitrage-free parameterisation."""

import numpy as np
import pytest

from quant_system.options.ssvi import (
    SSVIParams,
    ssvi_to_svi_params,
    ssvi_total_variance,
)
from quant_system.options.svi import svi_total_variance


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
