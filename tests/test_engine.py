"""Unit tests for the backtest engine, costs, and risk overlays.

The headline test is ``test_engine_blocks_clairvoyant_signal``: it proves the
engine's one-day lag actually defeats look-ahead, which is the property the whole
system's credibility rests on.
"""

import numpy as np
import pandas as pd
import pytest

from quant_system.config import CostConfig, RiskConfig
from quant_system.backtest.engine import (
    portfolio_returns, run_backtest, assert_no_lookahead,
)
from quant_system.backtest.costs import transaction_cost_fraction, square_root_impact
from quant_system.risk.limits import apply_drawdown_stop
from quant_system.data.loader import load_price_data


@pytest.fixture
def panel():
    return load_price_data(["AAA", "BBB", "CCC"], "2018-01-01", "2022-12-31",
                           use_synthetic=True, seed=1)


def test_portfolio_returns_applies_one_day_lag():
    # Prices where asset return on day t is known; weight set on t must earn t+1.
    idx = pd.bdate_range("2020-01-01", periods=5)
    prices = pd.DataFrame({"X": [100, 110, 121, 133.1, 146.41]}, index=idx)  # +10%/day
    returns = prices.pct_change()
    # Hold a full unit weight from the first day onward.
    weights = pd.DataFrame({"X": [1.0, 1.0, 1.0, 1.0, 1.0]}, index=idx)
    port = portfolio_returns(weights, returns)
    # Day 0: no prior weight -> 0. Day 1 onward: earns the +10%.
    assert port.iloc[0] == 0.0
    assert np.allclose(port.iloc[1:].values, 0.10)


def test_engine_blocks_clairvoyant_signal(panel):
    # A signal that "knows" today's return would be hugely profitable if the engine
    # leaked. Through shift(1) it must NOT be clairvoyant.
    rets = panel.returns()
    cheat = (rets > 0).astype(float)
    leaked = (cheat * rets).sum(axis=1)            # what a buggy engine books
    honest = run_backtest(cheat, panel, cost=None).returns
    leaked_sharpe = np.sqrt(252) * leaked.mean() / leaked.std()
    honest_sharpe = np.sqrt(252) * honest.mean() / honest.std()
    assert leaked_sharpe > 8.0                      # clairvoyant is absurdly good
    assert honest_sharpe < 2.0                      # engine removes the leak


def test_assert_no_lookahead_catches_unknown_ticker(panel):
    w = pd.DataFrame(0.0, index=panel.close.index, columns=["AAA", "ZZZ"])
    with pytest.raises(AssertionError):
        assert_no_lookahead(w, panel.close)


def test_assert_no_lookahead_catches_unsorted_index(panel):
    w = pd.DataFrame(0.0, index=panel.close.index, columns=panel.tickers)
    with pytest.raises(AssertionError):
        assert_no_lookahead(w.iloc[::-1], panel.close)


def test_square_root_impact_is_concave():
    # Doubling participation should less-than-double impact (sqrt, not linear).
    eta, sigma = 0.1, 0.02
    a = square_root_impact(np.array([0.01]), np.array([sigma]), eta)[0]
    b = square_root_impact(np.array([0.04]), np.array([sigma]), eta)[0]
    assert b > a
    assert b < 4 * a                                 # 4x size -> 2x impact, not 4x


def test_costs_increase_with_capital(panel):
    # Square-root impact makes the same trade more expensive at larger size.
    w = pd.DataFrame(0.0, index=panel.close.index, columns=panel.tickers)
    w.iloc[50:] = 1.0 / 3                             # establish a book mid-sample
    small = run_backtest(w, panel, cost=CostConfig(capital=1e6)).costs.sum()
    big = run_backtest(w, panel, cost=CostConfig(capital=5e8)).costs.sum()
    assert big > small * 1.5


def test_no_cost_when_no_trades(panel):
    w = pd.DataFrame(0.0, index=panel.close.index, columns=panel.tickers)
    res = run_backtest(w, panel, cost=CostConfig())
    assert res.costs.sum() == 0.0
    assert res.returns.abs().sum() == 0.0            # flat book -> no P&L


def test_drawdown_stop_caps_losses():
    # A steadily losing stream should be cut off by the stop.
    idx = pd.bdate_range("2020-01-01", periods=200)
    losing = pd.Series(-0.01, index=idx)             # -1%/day
    stopped = apply_drawdown_stop(losing, max_drawdown=0.10, cooldown=20)
    eq_stopped = (1 + stopped).cumprod().iloc[-1]
    eq_unstopped = (1 + losing).cumprod().iloc[-1]   # ~0.134 (-86%)
    # The stop must materially outperform no stop (it goes flat after each breach).
    assert eq_stopped > 2 * eq_unstopped
    assert stopped.eq(0.0).sum() > 0                 # some days were spent flat


def test_run_backtest_is_pure(panel):
    # Same inputs -> identical outputs, and inputs are not mutated.
    w = pd.DataFrame(0.0, index=panel.close.index, columns=panel.tickers)
    w.iloc[100:] = 0.2
    w_copy = w.copy()
    r1 = run_backtest(w, panel, cost=CostConfig()).returns
    r2 = run_backtest(w, panel, cost=CostConfig()).returns
    pd.testing.assert_series_equal(r1, r2)
    pd.testing.assert_frame_equal(w, w_copy)         # no side effects on inputs
