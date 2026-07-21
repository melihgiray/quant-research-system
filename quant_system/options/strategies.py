"""Three single-underlying option strategies.

All three are short-volatility in some form, which is the point: they are the
standard ways a book gets paid for selling insurance, and they fail in
characteristic ways that a backtest ought to show.

**Covered call.** Own the stock, sell an out-of-the-money call against it. You
collect premium and keep dividends, but you have capped your upside at the
strike: every large rally hands the shares away. The strategy is short a call
and long the stock, which is the payoff of a short put, so it earns the variance
premium while carrying full downside.

**Cash-secured put.** Hold cash, sell an out-of-the-money put, and keep enough
cash to buy the shares if assigned. Economically the same payoff as the covered
call, reached from the other side. When assigned you end up long stock, which
this implementation then liquidates so the cycle can repeat.

**Delta-hedged short straddle.** Sell the at-the-money call and put and hedge
the delta with stock every day. Hedging strips out the directional bet and
leaves the pure trade: you are short gamma and long theta, collecting implied
volatility and paying realised volatility. This is the cleanest expression of
the variance risk premium, and the one that blows up fastest when realised
volatility exceeds implied.

Every strategy reads position state from the book rather than keeping its own,
so there is nothing to fall out of sync when a contract is assigned away.
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from .book import CONTRACT_MULTIPLIER, OptionContract, Order
from .backtest import StrategyContext
from .pricing import CALL, PUT

logger = logging.getLogger(__name__)


def _short_options(ctx: StrategyContext, option_type: Optional[str] = None) -> dict:
    """Open short positions, optionally filtered by call/put."""
    return {c: q for c, q in ctx.book.open_contracts().items()
            if q < 0 and (option_type is None or c.option_type == option_type)}


def _long_options(ctx: StrategyContext) -> dict:
    return {c: q for c, q in ctx.book.open_contracts().items() if q > 0}


def covered_call(target_delta: float = 0.30, days_to_expiry: int = 28,
                 roll_at_days: int = 5, lots: int = 1) -> Callable:
    """Long stock, systematically short an OTM call against it.

    Parameters
    ----------
    target_delta : float
        Delta of the call to sell. 0.30 is the common choice: far enough out to
        keep most of the upside, near enough to collect real premium.
    days_to_expiry : int
        Target tenor when writing.
    roll_at_days : int
        Close and rewrite when this few days remain, rather than holding into
        expiry. Gamma rises sharply in the last week, so the position becomes
        much more sensitive to a late move than the remaining premium justifies.
    lots : int
        Number of 100-share lots to hold.
    """
    target_shares = lots * CONTRACT_MULTIPLIER

    def strategy(ctx: StrategyContext) -> List[Order]:
        orders: List[Order] = []

        # Maintain the stock position. Assignment leaves us short shares.
        share_gap = target_shares - ctx.book.shares
        if abs(share_gap) >= 1:
            orders.append(Order(quantity=float(share_gap), reason="maintain stock"))

        shorts = _short_options(ctx, CALL)
        for contract, qty in shorts.items():
            days_left = (contract.expiry - ctx.date).days
            if days_left <= roll_at_days:
                orders.append(Order(quantity=-qty, contract=contract,
                                    reason="close near-expiry call"))
        # Only write when nothing is outstanding that we are not already closing.
        keeping = [c for c in shorts if (c.expiry - ctx.date).days > roll_at_days]
        if keeping:
            return orders

        expiry = ctx.provider.nearest_expiry(ctx.date, days_to_expiry)
        if expiry is None:
            return orders
        contract = ctx.provider.contract_by_delta(ctx.date, expiry, target_delta, CALL)
        if contract is not None:
            orders.append(Order(quantity=-float(lots), contract=contract,
                                reason="write covered call"))
        return orders

    return strategy


def cash_secured_put(target_delta: float = 0.30, days_to_expiry: int = 28,
                     roll_at_days: int = 5, lots: int = 1) -> Callable:
    """Short an OTM put backed by cash; liquidate stock if assigned.

    Assignment is the interesting case. Being put the stock turns a cash
    position into a long equity position, so the strategy sells those shares
    back to the market (crossing the spread again) and resumes writing. That
    liquidation is a real cost of the strategy and is deliberately not hidden.
    """
    def strategy(ctx: StrategyContext) -> List[Order]:
        orders: List[Order] = []

        # Assigned stock goes back to the market so the cycle can repeat.
        if abs(ctx.book.shares) >= 1:
            orders.append(Order(quantity=-float(ctx.book.shares),
                                reason="liquidate assigned stock"))

        shorts = _short_options(ctx, PUT)
        for contract, qty in shorts.items():
            if (contract.expiry - ctx.date).days <= roll_at_days:
                orders.append(Order(quantity=-qty, contract=contract,
                                    reason="close near-expiry put"))
        keeping = [c for c in shorts if (c.expiry - ctx.date).days > roll_at_days]
        if keeping:
            return orders

        expiry = ctx.provider.nearest_expiry(ctx.date, days_to_expiry)
        if expiry is None:
            return orders
        contract = ctx.provider.contract_by_delta(ctx.date, expiry, target_delta, PUT)
        if contract is None:
            return orders

        # Size so the cash on hand could actually take delivery. A "cash
        # secured" put that cannot pay for the shares is just a naked put.
        affordable = int(ctx.book.cash // (contract.strike * CONTRACT_MULTIPLIER))
        size = max(min(lots, affordable), 0)
        if size > 0:
            orders.append(Order(quantity=-float(size), contract=contract,
                                reason="write cash-secured put"))
        return orders

    return strategy


def delta_hedged_short_straddle(days_to_expiry: int = 28, roll_at_days: int = 7,
                                lots: int = 1, delta_band: float = 5.0) -> Callable:
    """Short the ATM straddle, hedged back to flat delta each day.

    Parameters
    ----------
    delta_band : float
        Rehedge only when net delta drifts beyond this many shares. Hedging
        every tiny drift would pay the equity spread constantly for no risk
        reduction; a band trades a little residual delta for far fewer trades.
        This is the standard practical compromise and the band width is exactly
        the knob that decides how much of the premium survives.

    The position is short gamma: every hedge trade buys high and sells low, and
    those losses are the price of the theta being collected. Whether the trade
    makes money is entirely the gap between implied and realised volatility.
    """
    def strategy(ctx: StrategyContext) -> List[Order]:
        orders: List[Order] = []
        shorts = _short_options(ctx)

        expiring = [c for c in shorts if (c.expiry - ctx.date).days <= roll_at_days]
        for contract in expiring:
            orders.append(Order(quantity=-shorts[contract], contract=contract,
                                reason="close near-expiry straddle leg"))

        live = [c for c in shorts if (c.expiry - ctx.date).days > roll_at_days]

        if expiring and not live:
            # Closing every leg. The share hedge exists only to offset option
            # delta, so it has to come off with them: a hedge with nothing left
            # to hedge is an outright stock position, and leaving it on shows up
            # as a large delta spike across the roll.
            if abs(ctx.book.shares) >= 1:
                orders.append(Order(quantity=-float(ctx.book.shares),
                                    reason="unwind hedge on roll"))
            return orders

        if not live and not expiring:
            expiry = ctx.provider.nearest_expiry(ctx.date, days_to_expiry)
            if expiry is not None:
                strike = ctx.provider.nearest_strike(ctx.date, ctx.spot)
                for side in (CALL, PUT):
                    orders.append(Order(
                        quantity=-float(lots),
                        contract=OptionContract(pd.Timestamp(expiry), strike, side),
                        reason="open short straddle"))
            # Freshly opened straddle is close to delta neutral; hedge tomorrow.
            return orders

        # Hedge whatever delta the live position carries.
        vols = {c: ctx.provider.implied_vol(ctx.date, c)
                for c in ctx.book.open_contracts()}
        greeks = ctx.book.greeks(ctx.spot, ctx.date, ctx.provider.rate, vols,
                                 ctx.provider.dividend_yield)
        if abs(greeks.delta) > delta_band:
            orders.append(Order(quantity=-float(greeks.delta), reason="delta hedge"))
        return orders

    return strategy


def buy_and_hold_underlying(lots: int = 1) -> Callable:
    """Benchmark: just own the stock. Every option strategy should be measured
    against the thing it is supposedly improving on."""
    target = lots * CONTRACT_MULTIPLIER

    def strategy(ctx: StrategyContext) -> List[Order]:
        gap = target - ctx.book.shares
        if abs(gap) >= 1:
            return [Order(quantity=float(gap), reason="buy and hold")]
        return []

    return strategy
