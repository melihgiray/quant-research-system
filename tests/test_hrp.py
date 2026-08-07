"""Tests for the Hierarchical Risk Parity allocator."""

import numpy as np
import pandas as pd
import pytest

from quant_system.portfolio.hrp import (
    cluster_order,
    correlation_distance,
)


def _block_corr():
    # Two tight pairs (A,B) and (C,D); the pairs are nearly uncorrelated to each
    # other. Correct clustering keeps each pair adjacent.
    names = ["A", "B", "C", "D"]
    c = np.array([
        [1.00, 0.95, 0.05, 0.02],
        [0.95, 1.00, 0.03, 0.04],
        [0.05, 0.03, 1.00, 0.92],
        [0.02, 0.04, 0.92, 1.00],
    ])
    return pd.DataFrame(c, index=names, columns=names)


def test_correlation_distance_bounds_and_diagonal():
    corr = _block_corr()
    d = correlation_distance(corr)
    assert np.allclose(np.diag(d.to_numpy()), 0.0)                # identical -> 0
    assert (d.to_numpy() >= -1e-12).all() and (d.to_numpy() <= 1.0 + 1e-12).all()
    # A pair correlated 0.95 is much closer than an uncorrelated pair.
    assert d.loc["A", "B"] < d.loc["A", "C"]


def test_perfectly_correlated_is_zero_distance():
    corr = pd.DataFrame([[1.0, 1.0], [1.0, 1.0]], index=["X", "Y"], columns=["X", "Y"])
    d = correlation_distance(corr)
    assert d.loc["X", "Y"] == pytest.approx(0.0)


def test_cluster_order_keeps_correlated_pairs_adjacent():
    order = cluster_order(_block_corr())
    assert set(order) == {"A", "B", "C", "D"}
    pos = {name: i for i, name in enumerate(order)}
    assert abs(pos["A"] - pos["B"]) == 1                         # the tight pair is adjacent
    assert abs(pos["C"] - pos["D"]) == 1
