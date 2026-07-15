"""Tests for the volume-participation execution model."""

import numpy as np
import pandas as pd

from quant_system.config import CostConfig, ExecutionConfig
from quant_system.backtest.engine import run_backtest
from quant_system.backtest.execution import (
    constrained_holdings, participation_cap_weights,
)
from quant_system.data.loader import load_price_data
from quant_system.data.universe import universe


def _panel():
    return load_price_data(universe("sectors")[:6], "2018-01-01", "2022-12-31",
                           use_synthetic=True, seed=2)


def _jump_book(panel):
    # Flat for 50 days, then a full-size book overnight: the worst case for a cap.
    w = pd.DataFrame(0.0, index=panel.close.index, columns=panel.tickers)
    w.iloc[50:] = 1.0 / len(panel.tickers)
    return w


def test_constrained_holdings_ramps_toward_target():
    idx = pd.bdate_range("2020-01-01", periods=10)
    target = pd.DataFrame({"X": [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]}, index=idx, dtype=float)
    cap = pd.DataFrame({"X": 0.25}, index=idx)
    held = constrained_holdings(target, cap)
    # 0.25 per day: reaches the full position on day 5, not day 2.
    assert list(held["X"].iloc[:6]) == [0.0, 0.25, 0.5, 0.75, 1.0, 1.0]


def test_daily_trade_never_exceeds_cap():
    panel = _panel()
    w = _jump_book(panel)
    cost = CostConfig(capital=200_000_000)          # big book so the cap binds
    res = run_backtest(w, panel, cost=cost,
                       execution=ExecutionConfig(max_participation=0.05))
    trades = res.held_weights.diff()
    trades.iloc[0] = res.held_weights.iloc[0]
    adv = (panel.volume.rolling(cost.adv_lookback, min_periods=5).mean()
           .fillna(panel.volume.expanding(min_periods=1).mean()).shift(1))
    cap = participation_cap_weights(panel.close, adv, cost.capital, 0.05)
    finite = np.isfinite(cap.values)
    assert (trades.abs().values[finite] <= cap.values[finite] + 1e-12).all()


def test_no_cap_matches_previous_behaviour():
    panel = _panel()
    w = _jump_book(panel)
    base = run_backtest(w, panel, cost=CostConfig())
    uncapped = run_backtest(w, panel, cost=CostConfig(), execution=ExecutionConfig())
    pd.testing.assert_series_equal(base.returns, uncapped.returns)
    assert uncapped.fill_gap.abs().sum() == 0.0


def test_fill_gap_shrinks_as_position_completes():
    panel = _panel()
    w = _jump_book(panel)
    res = run_backtest(w, panel, cost=CostConfig(capital=500_000_000),
                       execution=ExecutionConfig(max_participation=0.02))
    gap = res.fill_gap
    assert gap.iloc[51] > 0                          # right after the jump: chasing
    assert gap.iloc[-1] < gap.iloc[51]               # later: mostly caught up
    assert res.meta["avg_fill_gap"] > 0


def test_cap_binds_harder_at_larger_capital():
    panel = _panel()
    w = _jump_book(panel)
    small = run_backtest(w, panel, cost=CostConfig(capital=1_000_000),
                         execution=ExecutionConfig(max_participation=0.05))
    big = run_backtest(w, panel, cost=CostConfig(capital=1_000_000_000),
                       execution=ExecutionConfig(max_participation=0.05))
    assert big.fill_gap.sum() > small.fill_gap.sum()


def test_cost_none_means_zero_costs():
    # The documented contract: cost=None charges nothing at all.
    panel = _panel()
    w = _jump_book(panel)
    res = run_backtest(w, panel, cost=None)
    assert res.costs.abs().sum() == 0.0
    assert res.meta["frictionless"] is True
    pd.testing.assert_series_equal(res.returns, res.gross_returns)