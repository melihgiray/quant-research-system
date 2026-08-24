"""Tests for probability-based bet sizing."""

import numpy as np
import pytest

from quant_system.signals.bet_sizing import bet_size, discretize_bets


def test_coin_flip_probability_sizes_to_zero():
    assert bet_size(0.5) == pytest.approx(0.0)


def test_certainty_sizes_to_a_full_bet():
    assert bet_size(0.999) == pytest.approx(1.0, abs=1e-2)
    assert bet_size(0.001) == pytest.approx(-1.0, abs=1e-2)   # side +1, prob near 0 -> short


def test_side_flips_the_sign():
    assert bet_size(0.8, side=-1.0) == pytest.approx(-bet_size(0.8, side=1.0))


def test_size_increases_with_probability():
    sizes = bet_size(np.array([0.55, 0.65, 0.75, 0.85]))
    assert np.all(np.diff(sizes) > 0)
    assert (sizes > 0).all() and (sizes < 1).all()


def test_discretize_rounds_to_steps():
    assert discretize_bets(0.37, step=0.25) == pytest.approx(0.25)
    assert discretize_bets(0.13, step=0.25) == pytest.approx(0.25)
    assert discretize_bets(0.12, step=0.25) == pytest.approx(0.0)


def test_discretize_rejects_nonpositive_step():
    with pytest.raises(ValueError, match="step must be positive"):
        discretize_bets(0.5, step=0.0)
