"""Tests for the entropy estimators."""

import numpy as np
import pytest

from quant_system.entropy import plugin_entropy, returns_to_bits


def test_fair_random_bits_have_near_maximal_entropy():
    rng = np.random.default_rng(0)
    bits = "".join(rng.choice(["0", "1"], size=20000))
    assert abs(plugin_entropy(bits, word_length=1) - 1.0) < 0.02


def test_constant_message_has_zero_entropy():
    assert plugin_entropy("0" * 100, word_length=1) == 0.0


def test_structure_shows_up_at_longer_words():
    # A perfectly alternating string looks like a fair coin per single bit, but its
    # structure appears at longer words: overlapping length-2 words are only "01"
    # and "10", so the per-symbol entropy drops.
    alternating = "01" * 5000
    assert plugin_entropy(alternating, word_length=1) == pytest.approx(1.0, abs=1e-3)
    assert plugin_entropy(alternating, word_length=2) < plugin_entropy(alternating, word_length=1)
    assert plugin_entropy(alternating, word_length=2) == pytest.approx(0.5, abs=1e-2)


def test_returns_to_bits_encodes_sign():
    assert returns_to_bits([0.01, -0.02, 0.0, 0.03]) == "1001"   # >0 -> 1


def test_word_length_must_be_positive():
    with pytest.raises(ValueError, match="word_length"):
        plugin_entropy("0101", word_length=0)
