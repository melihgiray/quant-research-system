"""Phase 3 tests: fills, assignment, strategies, and the options look-ahead guard.

The headline is ``test_clairvoyance_needs_two_days_of_foresight``. The equity
engine proves its one-day lag by showing a cheating signal earns nothing. The
options engine has discrete contracts rather than weights, so the same idea is
expressed by the *horizon* a cheat needs: a strategy that peeks one day ahead
gains nothing, because it trades after the move it foresaw; a strategy that
peeks two days ahead does profit, because it trades before. The gap between
those two is the lag, measured.
"""

import numpy as np
import pandas as pd
import pytest

from quant_system.data.loader import load_price_data
from quant_system.options.book import (
    Book, OptionContract, Order, CONTRACT_MULTIPLIER, fill_price, spread_cost,
)
from quant_system.options.backtest import run_options_backtest
from quant_system.options.provider import SyntheticChainProvider
from quant_system.options.pricing import CALL, PUT
from quant_system.options import strategies as S


@pytest.fixture(scope="module")
def prices():
    return load_price_data(["AAPL"], "2019-01-01", "2022-12-31",
                           use_synthetic=True).close["AAPL"]


@pytest.fixture(scope="module")
def provider(prices):
    return SyntheticChainProvider(underlying=prices, vol_premium=0.15)


# --------------------------------------------------------------------------- #
# Fills cross the spread
# --------------------------------------------------------------------------- #
def test_buy_at_ask_sell_at_bid():
    assert fill_price(1.00, 1.10, quantity=+1) == 1.10      # buying pays the ask
    assert fill_price(1.00, 1.10, quantity=-1) == 1.00      # selling hits the bid


def test_fill_rejects_broken_markets():
    with pytest.raises(ValueError, match="crossed"):
        fill_price(1.20, 1.00, quantity=1)
    with pytest.raises(ValueError, match="missing market"):
        fill_price(np.nan, 1.00, quantity=1)


def test_spread_cost_is_charged_on_both_sides():
    # A round trip on a 10-cent-wide option costs the full spread, not half.
    buy = spread_cost(1.00, 1.10, quantity=+1, multiplier=CONTRACT_MULTIPLIER)
    sell = spread_cost(1.00, 1.10, quantity=-1, multiplier=CONTRACT_MULTIPLIER)
    assert buy == pytest.approx(5.0)                        # 0.05 x 100
    assert buy + sell == pytest.approx(10.0)                # 0.10 x 100 round trip


def test_book_cash_moves_the_right_way():
    book = Book(cash=10_000.0)
    c = OptionContract(pd.Timestamp("2025-06-20"), 100.0, CALL)
    book.trade_option(c, quantity=-1, bid=2.00, ask=2.10, date="2025-01-02")
    assert book.cash == pytest.approx(10_000.0 + 200.0)     # sold at bid
    assert book.position(c) == -1
    book.trade_option(c, quantity=+1, bid=2.00, ask=2.10, date="2025-01-03")
    assert book.cash == pytest.approx(10_000.0 + 200.0 - 210.0)   # bought back at ask
    assert book.position(c) == 0


# --------------------------------------------------------------------------- #
# Expiry and assignment
# --------------------------------------------------------------------------- #
def test_otm_option_expires_worthless():
    book = Book(cash=1000.0)
    c = OptionContract(pd.Timestamp("2025-06-20"), 120.0, CALL)
    book.options[c] = -1
    note = book.settle_expiry(c, spot=100.0, date="2025-06-20")
    assert "worthless" in note
    assert book.cash == 1000.0 and book.shares == 0


def test_short_call_assignment_delivers_shares():
    book = Book(cash=0.0, shares=100.0)
    c = OptionContract(pd.Timestamp("2025-06-20"), 100.0, CALL)
    book.options[c] = -1
    note = book.settle_expiry(c, spot=130.0, date="2025-06-20")
    assert "assigned" in note
    assert book.shares == 0                                  # stock called away
    assert book.cash == pytest.approx(100.0 * 100)           # sold at the strike, not 130


def test_short_put_assignment_takes_delivery():
    book = Book(cash=20_000.0)
    c = OptionContract(pd.Timestamp("2025-06-20"), 100.0, PUT)
    book.options[c] = -1
    note = book.settle_expiry(c, spot=80.0, date="2025-06-20")
    assert "assigned" in note
    assert book.shares == 100                                # put the stock
    assert book.cash == pytest.approx(20_000.0 - 10_000.0)   # paid the strike


def test_long_call_exercise_takes_delivery():
    book = Book(cash=20_000.0)
    c = OptionContract(pd.Timestamp("2025-06-20"), 100.0, CALL)
    book.options[c] = +1
    book.settle_expiry(c, spot=130.0, date="2025-06-20")
    assert book.shares == 100
    assert book.cash == pytest.approx(10_000.0)


# --------------------------------------------------------------------------- #
# Strategies behave structurally as advertised
# --------------------------------------------------------------------------- #
def test_covered_call_stays_long_stock_and_short_calls(provider):
    res = run_options_backtest(provider, S.covered_call(), initial_capital=100_000,
                               label="covered call")
    assert res.n_trades > 20
    # Long roughly one lot of stock, minus the short call's delta.
    assert 0 < res.greeks["delta"].mean() < CONTRACT_MULTIPLIER
    assert res.greeks["vega"].mean() < 0                     # short volatility
    assert res.greeks["theta"].mean() > 0                    # collecting decay
    assert res.exposures["shares"].median() == pytest.approx(CONTRACT_MULTIPLIER, abs=1)


def test_cash_secured_put_never_writes_more_than_cash_covers(provider):
    res = run_options_backtest(provider, S.cash_secured_put(), initial_capital=100_000,
                               label="csp")
    assert res.n_trades > 20
    assert res.greeks["delta"].mean() > 0                    # short put is long delta
    assert res.greeks["vega"].mean() < 0
    assert (res.exposures["cash"] > 0).all()                 # stays genuinely secured


def test_delta_hedge_actually_neutralises_direction(provider):
    hedged = run_options_backtest(provider, S.delta_hedged_short_straddle(delta_band=5.0),
                                  initial_capital=100_000, label="hedged")
    naked = run_options_backtest(provider, S.delta_hedged_short_straddle(delta_band=1e9),
                                 initial_capital=100_000, label="naked")
    # With hedging on, net delta stays near flat; with the band effectively
    # infinite it never hedges and delta wanders.
    assert hedged.greeks["delta"].abs().mean() < naked.greeks["delta"].abs().mean()
    assert hedged.greeks["gamma"].mean() < 0                 # short gamma either way


def test_variance_premium_flows_through_monotonically(prices):
    # The premium is an input to the synthetic chain, so the backtest must
    # recover it monotonically. If it did not, the P&L accounting would be wrong.
    results = []
    for premium in (0.0, 0.25, 0.5):
        prov = SyntheticChainProvider(underlying=prices, vol_premium=premium)
        res = run_options_backtest(prov, S.delta_hedged_short_straddle(),
                                   initial_capital=100_000, label="x")
        results.append(res.equity.iloc[-1] / res.initial_capital - 1)
    assert results[0] < results[1] < results[2]
    # And with no premium at all, frictions must make it a loser.
    assert results[0] < 0


def test_spread_costs_are_material_and_recorded(provider):
    res = run_options_backtest(provider, S.delta_hedged_short_straddle(),
                               initial_capital=100_000, label="x")
    assert res.spread_cost > 0
    # Sum of per-fill costs must equal the book's running total.
    assert sum(f.spread_cost for f in res.trades) == pytest.approx(res.spread_cost)


# --------------------------------------------------------------------------- #
# The look-ahead guard
# --------------------------------------------------------------------------- #
def _clairvoyant(prices: pd.Series, horizon: int, provider):
    """Buy a call or put based on a move ``horizon`` days ahead of the decision.

    Orders decided on day i execute on day i+1. So horizon=1 means acting on a
    move that has already happened by the time you trade, and horizon=2 means
    acting on one that has not.
    """
    dates = list(prices.index)
    pos = {d: i for i, d in enumerate(dates)}

    def strategy(ctx):
        orders = [Order(quantity=-q, contract=c, reason="close")
                  for c, q in ctx.book.open_contracts().items()]
        i = pos[ctx.date]
        j = i + horizon
        if j >= len(dates):
            return orders
        move = float(prices.iloc[j] - prices.iloc[j - 1])
        expiry = provider.nearest_expiry(ctx.date, 28)
        if expiry is None:
            return orders
        strike = provider.nearest_strike(ctx.date, ctx.spot)
        side = CALL if move > 0 else PUT
        orders.append(Order(quantity=1.0,
                            contract=OptionContract(pd.Timestamp(expiry), strike, side),
                            reason="clairvoyant"))
        return orders

    return strategy


def test_clairvoyance_needs_two_days_of_foresight(prices):
    # Near-zero spreads so the test measures timing, not transaction cost.
    prov = SyntheticChainProvider(underlying=prices, vol_premium=0.0,
                                  half_spread_frac=1e-5, min_half_spread=1e-4,
                                  equity_half_spread_bps=0.01)

    one_day = run_options_backtest(prov, _clairvoyant(prices, 1, prov),
                                   initial_capital=100_000, label="peek 1")
    two_day = run_options_backtest(prov, _clairvoyant(prices, 2, prov),
                                   initial_capital=100_000, label="peek 2")

    def sharpe(res):
        r = res.returns.dropna()
        return float(np.sqrt(252) * r.mean() / r.std()) if r.std() > 0 else 0.0

    # Peeking one day ahead is useless: the move is already in the price by the
    # time the order fills. Peeking two days ahead is not.
    assert sharpe(two_day) > sharpe(one_day) + 1.0
    assert two_day.equity.iloc[-1] > one_day.equity.iloc[-1]
    # And the one-day peek must not look like a real edge.
    assert sharpe(one_day) < 2.0


def test_orders_execute_on_the_following_day(provider, prices):
    # A strategy that submits exactly one order on the first date must have no
    # position on that date and a position on the next.
    first = provider.dates[0]
    expiry = provider.nearest_expiry(first, 28)
    contract = OptionContract(pd.Timestamp(expiry),
                              provider.nearest_strike(first, provider.spot(first)), CALL)
    fired = {"done": False}

    def once(ctx):
        if not fired["done"]:
            fired["done"] = True
            return [Order(quantity=-1.0, contract=contract, reason="one shot")]
        return []

    res = run_options_backtest(provider, once, initial_capital=100_000,
                               start=first, label="one shot")
    assert res.exposures["n_options"].iloc[0] == 0           # nothing on decision day
    assert res.exposures["n_options"].iloc[1] == 1           # filled the next day
