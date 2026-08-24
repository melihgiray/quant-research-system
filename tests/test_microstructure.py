"""Tests for microstructure estimators."""

import numpy as np
import pytest

from quant_system.microstructure import amihud_illiquidity, roll_spread


def test_roll_recovers_a_known_bounce_spread():
    rng = np.random.default_rng(0)
    mid = np.cumsum(rng.normal(0, 0.1, 20000)) + 100.0
    spread = 0.5
    observed = mid + (spread / 2.0) * rng.choice([-1.0, 1.0], size=len(mid))
    assert abs(roll_spread(observed) - spread) < 0.05


def test_roll_is_near_zero_without_a_bounce():
    rng = np.random.default_rng(1)
    mid = np.cumsum(rng.normal(0, 0.1, 20000)) + 100.0
    assert roll_spread(mid) < 0.05


def test_roll_handles_short_input():
    assert np.isnan(roll_spread([100.0, 100.1]))


def test_amihud_is_higher_for_thinner_volume():
    rng = np.random.default_rng(2)
    prices = np.cumprod(1 + rng.normal(0, 0.01, 500)) * 100.0
    liquid = amihud_illiquidity(prices, np.full(500, 1e7))
    thin = amihud_illiquidity(prices, np.full(500, 1e5))    # same prices, 100x less volume
    assert thin > liquid
    assert thin == pytest.approx(liquid * 100.0, rel=1e-6)  # inverse in volume


def test_amihud_skips_zero_volume_days():
    prices = np.array([100.0, 101.0, 102.0])
    vol = np.array([0.0, 0.0, 0.0])
    assert np.isnan(amihud_illiquidity(prices, vol))
