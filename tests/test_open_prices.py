"""Tests for open-price plumbing and the next-open fill mode."""

import numpy as np
import pandas as pd
import pytest

from quant_system.data.loader import PriceData


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
