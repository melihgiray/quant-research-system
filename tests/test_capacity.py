"""Tests for Day-3 capacity / cost-sensitivity analysis."""

import numpy as np
import pandas as pd

from quant_system.config import CostConfig
from quant_system.data.loader import load_price_data
from quant_system.data.universe import universe
from quant_system.backtest.capacity import sweep_capital, estimate_capacity


def _panel():
    return load_price_data(universe("sectors"), "2017-01-01", "2023-12-31",
                           use_synthetic=True, seed=1)


def _alternating_book(panel):
    # Flips between fully invested and flat every day -> guaranteed daily turnover.
    arr = np.zeros((len(panel.close.index), len(panel.tickers)))
    arr[::2, :] = 1.0 / len(panel.tickers)
    return pd.DataFrame(arr, index=panel.close.index, columns=panel.tickers)


def test_cost_drag_increases_with_capital():
    panel = _panel()
    w = _alternating_book(panel)
    caps = [1e5, 1e6, 1e7, 1e8, 1e9]
    df = sweep_capital(w, panel, CostConfig(), caps)
    assert list(df.columns) == ["net_ann_return", "net_sharpe", "cost_drag_ann", "ann_cost"]
    # Square-root impact: bigger book -> strictly more cost drag.
    assert df["cost_drag_ann"].iloc[-1] > df["cost_drag_ann"].iloc[0]
    # Net Sharpe is non-increasing in capital (allowing tiny float noise).
    assert (df["net_sharpe"].diff().dropna() <= 1e-9).all()


def test_estimate_capacity_on_profitable_book():
    panel = _panel()
    # Equal-weight long benefits from the synthetic upward drift -> gross Sharpe > 0.
    ew = pd.DataFrame(1.0 / len(panel.tickers), index=panel.close.index, columns=panel.tickers)
    df = sweep_capital(ew, panel, CostConfig(), [1e5, 1e6, 1e7, 1e8, 1e9, 1e10])
    cap = estimate_capacity(df)
    assert cap["gross_sharpe"] > 0
    assert set(["capacity", "gross_sharpe", "threshold"]).issubset(cap.keys())
    assert cap["capacity"] > 0                       # finite positive or +inf


def test_capacity_undefined_when_unprofitable():
    panel = _panel()
    short = pd.DataFrame(-1.0 / len(panel.tickers), index=panel.close.index, columns=panel.tickers)
    df = sweep_capital(short, panel, CostConfig(), [1e6, 1e8])
    cap = estimate_capacity(df)
    assert np.isnan(cap["capacity"])                 # negative frictionless Sharpe -> no capacity
    assert "note" in cap
