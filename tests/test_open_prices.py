"""Tests for open-price plumbing and the next-open fill mode."""

import numpy as np
import pandas as pd
import pytest

from quant_system.data.loader import PriceData, _align, load_price_data
from quant_system.data.universe import universe


def _panel_with_open():
    idx = pd.bdate_range("2021-01-01", periods=5)
    close = pd.DataFrame({"A": [10, 11, 12, 11, 13], "B": [20, 19, 21, 22, 20]},
                         index=idx, dtype=float)
    opn = close.shift(1).fillna(close.iloc[0])
    vol = pd.DataFrame(1e6, index=idx, columns=["A", "B"])
    return PriceData(close, vol, synthetic=False, open=opn)


def test_price_data_defaults_to_no_open():
    idx = pd.bdate_range("2021-01-01", periods=3)
    pd_ = PriceData(pd.DataFrame({"A": [1.0, 2, 3]}, index=idx),
                    pd.DataFrame({"A": [1.0, 1, 1]}, index=idx))
    assert pd_.has_open is False
    assert pd_.open is None


def test_subset_preserves_open():
    panel = _panel_with_open()
    sub = panel.subset(["A"])
    assert sub.has_open
    assert list(sub.open.columns) == ["A"]
    assert sub.open.shape == sub.close.shape


def test_synthetic_panel_carries_aligned_open():
    panel = load_price_data(universe("largecaps")[:4], "2019-01-01", "2020-12-31",
                            use_synthetic=True)
    assert panel.has_open
    assert list(panel.open.columns) == list(panel.close.columns)
    assert panel.open.index.equals(panel.close.index)
    assert (panel.open.to_numpy() > 0).all()
    # Open is close-with-an-overnight-gap, so it differs from the prior close.
    assert not np.allclose(panel.open.to_numpy()[1:], panel.close.to_numpy()[:-1])


def test_align_needs_open_for_every_ticker():
    idx = pd.bdate_range("2021-01-01", periods=4)
    close = pd.DataFrame({"A": [1.0, 2, 3, 4], "B": [5.0, 6, 7, 8]}, index=idx)
    vol = pd.DataFrame(1e6, index=idx, columns=["A", "B"])
    partial = {"A": close["A"]}                          # B has no open
    assert _align(close, vol, ["A", "B"], partial).open is None
    full = {"A": close["A"], "B": close["B"]}
    assert _align(close, vol, ["A", "B"], full).has_open
