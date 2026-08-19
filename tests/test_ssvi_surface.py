"""Tests for the calendar-arbitrage-free SSVI surface."""

import numpy as np
import pandas as pd
import pytest

from quant_system.options.ssvi import (
    SSVISurface,
    fit_ssvi_surface,
    ssvi_butterfly_free,
    ssvi_surface_arbitrage_free,
    ssvi_surface_slice,
    ssvi_total_variance,
)


def _points_from_surface(surface, k=None):
    k = np.linspace(-0.5, 0.5, 15) if k is None else k
    rows = []
    for i, t in enumerate(surface.maturities):
        w = ssvi_total_variance(k, ssvi_surface_slice(surface, i))
        for kk, ww in zip(k, w):
            rows.append({"time_to_expiry": float(t), "log_moneyness": float(kk),
                         "total_variance": float(ww)})
    return pd.DataFrame(rows)


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


def test_surface_fit_recovers_a_known_arbitrage_free_surface():
    true = _good_surface()
    surface, diag = fit_ssvi_surface(_points_from_surface(true))
    assert diag["rmse"].max() < 1e-6
    assert surface.rho == pytest.approx(true.rho, abs=1e-2)
    assert np.allclose(surface.thetas, true.thetas, atol=1e-3)
    assert ssvi_surface_arbitrage_free(surface) == (True, True)


def test_surface_fit_forces_a_monotone_theta_when_data_has_calendar_arbitrage():
    # Feed a term structure whose at-the-money variance falls with maturity: the
    # fit cannot honour it (theta is monotone by construction) and returns a
    # calendar-free surface at a cost in fit error.
    rows = []
    k = np.linspace(-0.4, 0.4, 12)
    for t, theta in ((0.1, 0.06), (0.5, 0.02)):            # variance DROPS with maturity
        slice_surface = SSVISurface(rho=-0.2, eta=0.5, gamma=0.5,
                                    maturities=np.array([t]), thetas=np.array([theta]))
        w = ssvi_total_variance(k, ssvi_surface_slice(slice_surface, 0))
        for kk, ww in zip(k, w):
            rows.append({"time_to_expiry": t, "log_moneyness": float(kk),
                         "total_variance": float(ww)})
    surface, _ = fit_ssvi_surface(pd.DataFrame(rows))
    assert np.all(np.diff(surface.thetas) >= -1e-9)        # theta non-decreasing
    assert ssvi_surface_arbitrage_free(surface)[1]         # calendar-free by construction


def test_surface_fit_needs_two_expiries():
    s = _good_surface()
    one = _points_from_surface(SSVISurface(rho=s.rho, eta=s.eta, gamma=s.gamma,
                                           maturities=s.maturities[:1], thetas=s.thetas[:1]))
    with pytest.raises(ValueError, match="at least two expiries"):
        fit_ssvi_surface(one)
