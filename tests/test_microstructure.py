"""Tests for microstructure estimators."""

import numpy as np

from quant_system.microstructure import roll_spread


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
