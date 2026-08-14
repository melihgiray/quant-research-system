"""Tests for open-price plumbing and the next-open fill mode."""

import numpy as np
import pandas as pd
import pytest

from quant_system.backtest.engine import (
    portfolio_returns,
    portfolio_returns_next_open,
    run_backtest,
)
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


def test_next_open_returns_earn_the_open_to_open_interval():
    idx = pd.bdate_range("2021-01-01", periods=4)
    opn = pd.DataFrame({"A": [10.0, 11.0, 12.0, 13.0]}, index=idx)
    w = pd.DataFrame({"A": [1.0, 1.0, 1.0, 1.0]}, index=idx)
    r = portfolio_returns_next_open(w, opn)
    assert r.iloc[0] == pytest.approx(0.0)               # weight not held yet
    assert r.iloc[1] == pytest.approx(12.0 / 11.0 - 1.0)  # open(t+1)/open(t) - 1
    assert r.iloc[2] == pytest.approx(13.0 / 12.0 - 1.0)
    assert r.iloc[3] == pytest.approx(0.0)               # no next open -> earns nothing


def test_next_open_is_causal_in_the_weights():
    idx = pd.bdate_range("2021-01-01", periods=6)
    opn = pd.DataFrame({"A": np.linspace(10, 15, 6)}, index=idx)
    w = pd.DataFrame({"A": [1.0, 1, 1, 1, 1, 1]}, index=idx)
    base = portfolio_returns_next_open(w, opn)
    w2 = w.copy()
    w2.iloc[4:] = -3.0                                    # change weights from t=4 on
    after = portfolio_returns_next_open(w2, opn)
    pd.testing.assert_series_equal(base.iloc[:4], after.iloc[:4])   # earlier P&L unchanged


def test_next_open_differs_from_close_fill_with_overnight_gaps():
    panel = load_price_data(universe("largecaps")[:3], "2019-01-01", "2020-12-31",
                            use_synthetic=True)
    close_ret = panel.close.pct_change()
    w = pd.DataFrame(1.0 / panel.close.shape[1], index=panel.close.index,
                     columns=panel.close.columns)
    close_pnl = portfolio_returns(w, close_ret)
    open_pnl = portfolio_returns_next_open(w, panel.open)
    assert not np.allclose(close_pnl.to_numpy(), open_pnl.to_numpy())


def test_run_backtest_next_open_runs_and_differs_from_close():
    panel = load_price_data(universe("largecaps")[:4], "2018-01-01", "2020-12-31",
                            use_synthetic=True)
    w = pd.DataFrame(1.0 / panel.close.shape[1], index=panel.close.index,
                     columns=panel.close.columns)
    close_run = run_backtest(w, panel, fill="close")
    open_run = run_backtest(w, panel, fill="next_open")
    assert len(open_run.returns) == len(close_run.returns)
    assert not np.allclose(close_run.returns.to_numpy(), open_run.returns.to_numpy())


def test_run_backtest_next_open_requires_open_prices():
    panel = load_price_data(universe("largecaps")[:4], "2018-01-01", "2019-12-31",
                            use_synthetic=True)
    stripped = PriceData(panel.close, panel.volume, synthetic=True)   # no open
    w = pd.DataFrame(0.0, index=panel.close.index, columns=panel.close.columns)
    with pytest.raises(ValueError, match="requires open prices"):
        run_backtest(w, stripped, fill="next_open")
