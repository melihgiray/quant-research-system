"""Tests for up/down capture ratios."""

import numpy as np
import pandas as pd

from quant_system.performance.active import capture_ratios


def _bench(n=4000, seed=0):
    return pd.Series(np.random.default_rng(seed).normal(0.0, 0.01, n))


def test_leveraged_strategy_captures_proportionally():
    b = _bench()
    up, down = capture_ratios(1.5 * b, b)
    assert up == np.float64(1.5) or abs(up - 1.5) < 1e-9
    assert abs(down - 1.5) < 1e-9


def test_identical_strategy_captures_one():
    b = _bench(seed=1)
    up, down = capture_ratios(b, b)
    assert abs(up - 1.0) < 1e-9 and abs(down - 1.0) < 1e-9


def test_defensive_strategy_has_low_down_capture():
    b = _bench(seed=2)
    # Full participation up, half participation down.
    strat = b.where(b > 0, b * 0.5)
    up, down = capture_ratios(strat, b)
    assert down < up
    assert down < 0.7
