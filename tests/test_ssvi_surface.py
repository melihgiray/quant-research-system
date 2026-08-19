"""Tests for the calendar-arbitrage-free SSVI surface."""

import numpy as np
import pytest

from quant_system.options.ssvi import (
    SSVISurface,
    ssvi_butterfly_free,
    ssvi_surface_arbitrage_free,
    ssvi_surface_slice,
)


def _good_surface():
    mats = np.array([0.08, 0.25, 0.5, 1.0])
    thetas = np.array([0.010, 0.028, 0.052, 0.100])          # rising with maturity
    return SSVISurface(rho=-0.3, eta=0.6, gamma=0.5, maturities=mats, thetas=thetas)


def test_slices_carry_the_shared_rho_and_curvature_law():
    s = _good_surface()
    sl0, sl1 = ssvi_surface_slice(s, 0), ssvi_surface_slice(s, 1)
    assert sl0.rho == s.rho == sl1.rho
    assert sl0.theta == pytest.approx(s.thetas[0])
    assert sl1.psi < sl0.psi                              # phi decreases with theta (gamma>0)


def test_good_surface_is_butterfly_and_calendar_free():
    s = _good_surface()
    butterfly, calendar = ssvi_surface_arbitrage_free(s)
    assert butterfly and calendar
    assert all(ssvi_butterfly_free(ssvi_surface_slice(s, i)) for i in range(len(s.thetas)))


def test_decreasing_theta_is_calendar_arbitrage():
    s = _good_surface()
    bad = SSVISurface(rho=s.rho, eta=s.eta, gamma=s.gamma, maturities=s.maturities,
                      thetas=np.array([0.10, 0.05, 0.05, 0.02]))   # total variance falls
    butterfly, calendar = ssvi_surface_arbitrage_free(bad)
    assert not calendar


def test_excessive_curvature_breaks_the_butterfly_condition():
    s = _good_surface()
    bad = SSVISurface(rho=s.rho, eta=40.0, gamma=s.gamma, maturities=s.maturities,
                      thetas=s.thetas)                     # huge eta -> steep slices
    butterfly, _ = ssvi_surface_arbitrage_free(bad)
    assert not butterfly
