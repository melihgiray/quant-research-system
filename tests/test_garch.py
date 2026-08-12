"""Tests for GARCH(1,1) conditional-volatility forecasts."""

import numpy as np
import pandas as pd
import pytest

arch = pytest.importorskip("arch")   # skip cleanly where the garch extra is absent

from quant_system.risk.garch import garch_forecast_vol


def _garch_like(n=1200, omega=1e-6, alpha=0.08, beta=0.90, seed=0):
    """Simulate a return path with genuine volatility clustering."""
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    var = omega / (1 - alpha - beta)
    for t in range(1, n):
        var = omega + alpha * r[t - 1] ** 2 + beta * var
        r[t] = rng.normal(0.0, np.sqrt(var))
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.Series(r, index=idx)


def test_forecast_is_positive_and_finite():
    v = garch_forecast_vol(_garch_like())
    assert np.isfinite(v) and v > 0


def test_higher_vol_series_forecasts_higher_vol():
    calm = _garch_like(omega=1e-6, seed=1)
    wild = _garch_like(omega=1e-5, seed=1)              # ~3x the unconditional variance
    assert garch_forecast_vol(wild) > garch_forecast_vol(calm)


def test_forecast_in_a_sane_range_of_sample_vol():
    r = _garch_like(seed=2)
    v = garch_forecast_vol(r)
    sample = r.std()
    assert 0.3 * sample < v < 3.0 * sample             # same order of magnitude


def test_too_few_observations_raises():
    with pytest.raises(ValueError, match="at least"):
        garch_forecast_vol(_garch_like(n=50))
