"""Tests for the sequential bootstrap over overlapping labels."""

import collections

import numpy as np

from quant_system.signals.sampling import (
    average_uniqueness,
    indicator_matrix,
    sequential_bootstrap,
)


def _overlap_case():
    # Labels A and B share the same window [0, 4]; C is disjoint at [5, 9].
    return indicator_matrix(10, [(0, 4), (0, 4), (5, 9)])


def test_indicator_matrix_marks_the_spans():
    ind = indicator_matrix(6, [(0, 2), (3, 5)])
    assert ind.shape == (6, 2)
    assert list(ind[:, 0]) == [1, 1, 1, 0, 0, 0]
    assert list(ind[:, 1]) == [0, 0, 0, 1, 1, 1]


def test_average_uniqueness_halves_overlapping_labels():
    u = average_uniqueness(_overlap_case())
    assert np.allclose(u, [0.5, 0.5, 1.0])               # A,B share; C is alone


def test_disjoint_labels_are_fully_unique():
    ind = indicator_matrix(6, [(0, 2), (3, 5)])
    assert np.allclose(average_uniqueness(ind), [1.0, 1.0])


def test_sequential_bootstrap_oversamples_the_unique_label():
    draws = sequential_bootstrap(_overlap_case(), size=3000, seed=1)
    freq = collections.Counter(draws)
    n = len(draws)
    f_c = freq[2] / n
    assert f_c > 1.0 / 3.0                                # the disjoint label is favoured
    assert f_c > freq[0] / n and f_c > freq[1] / n        # over each overlapping one


def test_sequential_bootstrap_is_deterministic_and_valid():
    ind = _overlap_case()
    a = sequential_bootstrap(ind, size=20, seed=7)
    b = sequential_bootstrap(ind, size=20, seed=7)
    assert a == b
    assert len(a) == 20
    assert set(a) <= {0, 1, 2}
