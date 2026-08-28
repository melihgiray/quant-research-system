"""Tests for time-series statistics."""

import numpy as np

import pytest

from quant_system.stats import autocorrelation, hurst_exponent, variance_ratio


def _random_walk(n=8000, seed=0):
    return np.cumsum(np.random.default_rng(seed).normal(0, 1, n))


def _persistent(n=20000, seed=1):
    rng = np.random.default_rng(seed)
    inc = np.zeros(n)
    for i in range(1, n):
        inc[i] = 0.7 * inc[i - 1] + rng.normal(0, 1)     # positively autocorrelated increments
    return np.cumsum(inc)


def _mean_reverting(n=8000, seed=2):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.4 * x[i - 1] + rng.normal(0, 1)          # OU-like, anti-persistent
    return x


def test_random_walk_has_hurst_near_half():
    assert abs(hurst_exponent(_random_walk()) - 0.5) < 0.06


def test_persistent_series_exceeds_half():
    assert hurst_exponent(_persistent()) > 0.55


def test_mean_reverting_series_is_below_half():
    assert hurst_exponent(_mean_reverting()) < 0.45


def test_variance_ratio_near_one_for_a_random_walk():
    assert abs(variance_ratio(_random_walk(), q=4) - 1.0) < 0.1


def test_variance_ratio_above_one_when_persistent():
    assert variance_ratio(_persistent(), q=4) > 1.2


def test_variance_ratio_below_one_when_mean_reverting():
    assert variance_ratio(_mean_reverting(), q=4) < 0.8


def test_variance_ratio_rejects_small_q():
    with pytest.raises(ValueError, match="q >= 2"):
        variance_ratio(_random_walk(), q=1)


def _ar1(phi, n=20000, seed=5):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal(0, 1)
    return x


def test_white_noise_has_near_zero_autocorrelation():
    noise = np.random.default_rng(6).normal(0, 1, 20000)
    assert abs(autocorrelation(noise, lag=1)) < 0.03


def test_ar1_autocorrelation_matches_phi():
    x = _ar1(0.6)
    assert abs(autocorrelation(x, lag=1) - 0.6) < 0.03
    assert abs(autocorrelation(x, lag=2) - 0.36) < 0.04     # phi**2


def test_autocorrelation_rejects_bad_lag():
    with pytest.raises(ValueError, match="1 <= lag"):
        autocorrelation(_ar1(0.5), lag=0)


def _bubble(seed=0):
    rng = np.random.default_rng(seed)
    b = list(np.cumsum(rng.normal(0, 1, 150)))
    v = b[-1]
    for _ in range(100):
        v = 1.05 * v + rng.normal(0, 1)                  # explosive AR: coefficient > 1
        b.append(v)
    return np.array(b)


def test_sadf_stays_low_for_a_random_walk():
    from quant_system.stats import sadf
    assert np.nanmax(sadf(_random_walk(n=300))) < 2.0


def test_sadf_spikes_for_an_explosive_bubble():
    from quant_system.stats import sadf
    assert np.nanmax(sadf(_bubble())) > 5.0


def test_sadf_warmup_is_nan():
    from quant_system.stats import sadf
    out = sadf(_random_walk(n=200), min_window=40)
    assert np.isnan(out[:80]).all()
    assert len(out) == 200


def _ou(kappa, n=50000, seed=9):
    rng = np.random.default_rng(seed)
    s = np.zeros(n)
    for i in range(1, n):
        s[i] = (1 - kappa) * s[i - 1] + rng.normal(0, 1)
    return s


def test_half_life_recovers_the_ou_theory():
    from quant_system.stats import half_life
    assert abs(half_life(_ou(0.1)) - np.log(2) / 0.1) < 1.0


def test_faster_reversion_has_shorter_half_life():
    from quant_system.stats import half_life
    assert half_life(_ou(0.3)) < half_life(_ou(0.1)) < half_life(_ou(0.05))


def test_random_walk_half_life_is_large():
    from quant_system.stats import half_life
    assert half_life(_random_walk(n=50000)) > 200


def test_jarque_bera_does_not_reject_normal_data():
    from quant_system.stats import jarque_bera
    normal = np.random.default_rng(0).normal(0, 1, 5000)
    _, p = jarque_bera(normal)
    assert p > 0.05                                   # fail to reject normality


def test_jarque_bera_rejects_fat_tails():
    from quant_system.stats import jarque_bera
    fat = np.random.default_rng(1).standard_t(3, 5000)
    _, p = jarque_bera(fat)
    assert p < 0.01                                   # reject normality


def test_ljung_box_does_not_reject_white_noise():
    from quant_system.stats import ljung_box
    noise = np.random.default_rng(0).normal(0, 1, 3000)
    _, p = ljung_box(noise, lags=10)
    assert p > 0.05


def test_ljung_box_rejects_autocorrelated_series():
    from quant_system.stats import ljung_box
    _, p = ljung_box(_ar1(0.5, n=3000), lags=10)
    assert p < 0.01


def test_runs_test_passes_random_signs():
    from quant_system.stats import runs_test
    r = np.random.default_rng(0).normal(0, 1, 5000)
    _, p = runs_test(r)
    assert p > 0.05


def test_runs_test_flags_long_streaks():
    from quant_system.stats import runs_test
    trend = np.concatenate([np.ones(500), -np.ones(500), np.ones(500)])   # few long runs
    z, p = runs_test(trend)
    assert p < 0.01 and z < 0                          # far fewer runs than random
