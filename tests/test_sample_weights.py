"""Tests for label-uniqueness sample weights."""

import numpy as np
import pandas as pd
import pytest

from quant_system.signals.sample_weights import (
    concurrency_counts,
    uniqueness_weights,
)


def _dates():
    # Day A shared by 4 rows, day B by 1 row.
    a = pd.Timestamp("2021-01-04")
    b = pd.Timestamp("2021-01-05")
    return pd.DatetimeIndex([a, a, a, a, b])


def test_concurrency_counts_matches_shared_dates():
    c = concurrency_counts(_dates())
    assert list(c) == [4, 4, 4, 4, 1]


def test_crowded_days_are_down_weighted():
    w = uniqueness_weights(_dates(), normalize=False)
    assert w[0] == pytest.approx(0.25)                   # 1/4 for a crowded row
    assert w[-1] == pytest.approx(1.0)                   # lone row keeps full weight
    assert w[-1] > w[0]


def test_normalised_weights_average_one():
    w = uniqueness_weights(_dates(), normalize=True)
    assert w.mean() == pytest.approx(1.0)
    assert w[-1] > w[0]                                  # ordering preserved after scaling


def test_all_unique_dates_give_equal_weights():
    dates = pd.date_range("2021-01-04", periods=6, freq="B")
    w = uniqueness_weights(dates)
    assert np.allclose(w, 1.0)


def test_empty_input():
    assert uniqueness_weights(pd.DatetimeIndex([])).shape == (0,)
