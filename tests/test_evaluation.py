"""Tests for signal-quality evaluation."""

import numpy as np

from quant_system.signals.evaluation import rank_ic


def test_perfect_predictor_has_ic_near_one():
    rng = np.random.default_rng(0)
    signal = rng.normal(0, 1, 1000)
    forward = signal * 2 + rng.normal(0, 0.01, 1000)      # monotone in signal
    assert rank_ic(signal, forward) > 0.99


def test_random_signal_has_ic_near_zero():
    rng = np.random.default_rng(1)
    assert abs(rank_ic(rng.normal(0, 1, 5000), rng.normal(0, 1, 5000))) < 0.05


def test_inverted_predictor_is_negative():
    rng = np.random.default_rng(2)
    signal = rng.normal(0, 1, 1000)
    assert rank_ic(signal, -signal) < -0.99
