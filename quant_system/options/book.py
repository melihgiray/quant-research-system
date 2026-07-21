"""Contracts, portfolio state, and the fill rules.

Two conventions here do most of the honesty work in the options backtest.

**Fills cross the spread.** You buy at the ask and sell at the bid, never at
mid. This matters far more for options than for equities: a stock might be a
basis point wide, while a single-stock option is routinely 2-5% wide. A
covered-call backtest that fills at mid is not optimistic by a rounding error,
it is optimistic by most of the premium it claims to collect.

**Marks are at mid, fills are at the touch.** Position value is marked to mid
because that is the fair value of what you hold; the spread is a real cost that
gets paid when you actually trade, not a continuous drag on the mark. Booking
marks at the bid instead would double-count the spread you already paid on the
way in.

Quantities are in contracts, signed, and every option controls
``CONTRACT_MULTIPLIER`` shares. Negative quantity is short.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .greeks import Greeks, black_scholes_greeks
from .pricing import CALL, PUT

logger = logging.getLogger(__name__)

CONTRACT_MULTIPLIER = 100
UNDERLYING = None          # an Order with contract=None trades the underlying


@dataclass(frozen=True)
class OptionContract:
    """A single listed option. Hashable, so it can key a position dict."""

    expiry: pd.Timestamp
    strike: float
    option_type: str

    def __post_init__(self) -> None:
        if self.option_type not in (CALL, PUT):
            raise ValueError(f"option_type must be call/put, got {self.option_type!r}")

    def __str__(self) -> str:
        return f"{self.expiry:%Y-%m-%d} {self.strike:g} {self.option_type[0].upper()}"

    def intrinsic(self, spot: float) -> float:
        """Per-share intrinsic value at ``spot``."""
        if self.option_type == CALL:
            return max(spot - self.strike, 0.0)
        return max(self.strike - spot, 0.0)

    def time_to_expiry(self, date, days_per_year: float = 365.0) -> float:
        return max((self.expiry - pd.Timestamp(date)).days / days_per_year, 0.0)


@dataclass(frozen=True)
class Order:
    """An instruction to trade. ``contract=None`` means the underlying."""

    quantity: float                                  # signed; contracts, or shares
    contract: Optional[OptionContract] = None
    reason: str = ""

    @property
    def is_underlying(self) -> bool:
        return self.contract is None


@dataclass
class Fill:
    """What actually happened when an order was executed."""

    order: Order
    price: float
    cash_delta: float
    spread_cost: float
    date: pd.Timestamp


def fill_price(bid: float, ask: float, quantity: float) -> float:
    """Price you actually get: ask when buying, bid when selling.

    Raises on a crossed or missing market rather than silently filling at a
    price that does not exist.
    """
    if not (np.isfinite(bid) and np.isfinite(ask)):
        raise ValueError(f"cannot fill against a missing market (bid={bid}, ask={ask})")
    if ask < bid:
        raise ValueError(f"crossed market: bid {bid} > ask {ask}")
    return ask if quantity > 0 else bid


def spread_cost(bid: float, ask: float, quantity: float, multiplier: int) -> float:
    """Dollar cost of crossing, relative to the mid, for this trade."""
    mid = 0.5 * (bid + ask)
    price = fill_price(bid, ask, quantity)
    return abs(price - mid) * abs(quantity) * multiplier


@dataclass
class Book:
    """Portfolio state: cash, underlying shares, and option positions."""

    cash: float = 0.0
    shares: float = 0.0
    options: Dict[OptionContract, float] = field(default_factory=dict)
    realised_spread_cost: float = 0.0

    def position(self, contract: OptionContract) -> float:
        return self.options.get(contract, 0.0)

    def open_contracts(self) -> Dict[OptionContract, float]:
        return {c: q for c, q in self.options.items() if abs(q) > 1e-12}

    # ------------------------------------------------------------------ #
    # Trading
    # ------------------------------------------------------------------ #
    def trade_option(self, contract: OptionContract, quantity: float,
                     bid: float, ask: float, date) -> Fill:
        """Buy (positive) or sell (negative) ``quantity`` contracts, crossing the spread."""
        if quantity == 0:
            raise ValueError("cannot trade zero contracts")
        price = fill_price(bid, ask, quantity)
        cost = spread_cost(bid, ask, quantity, CONTRACT_MULTIPLIER)
        cash_delta = -price * quantity * CONTRACT_MULTIPLIER
        self.cash += cash_delta
        self.options[contract] = self.position(contract) + quantity
        self.realised_spread_cost += cost
        return Fill(Order(quantity, contract), price, cash_delta, cost, pd.Timestamp(date))

    def trade_underlying(self, quantity: float, bid: float, ask: float, date) -> Fill:
        """Buy or sell shares of the underlying, also crossing the spread."""
        if quantity == 0:
            raise ValueError("cannot trade zero shares")
        price = fill_price(bid, ask, quantity)
        cost = spread_cost(bid, ask, quantity, 1)
        cash_delta = -price * quantity
        self.cash += cash_delta
        self.shares += quantity
        self.realised_spread_cost += cost
        return Fill(Order(quantity), price, cash_delta, cost, pd.Timestamp(date))

    # ------------------------------------------------------------------ #
    # Valuation
    # ------------------------------------------------------------------ #
    def equity(self, spot: float, option_marks: Dict[OptionContract, float]) -> float:
        """Total account value: cash + shares at spot + options at mid."""
        value = self.cash + self.shares * spot
        for contract, qty in self.options.items():
            if abs(qty) < 1e-12:
                continue
            mark = option_marks.get(contract)
            if mark is None or not np.isfinite(mark):
                raise KeyError(f"no mark available for held contract {contract}")
            value += mark * qty * CONTRACT_MULTIPLIER
        return value

    def greeks(self, spot: float, date, rate: float, vols: Dict[OptionContract, float],
               dividend_yield: float = 0.0) -> Greeks:
        """Aggregate position Greeks, in shares-equivalent units.

        Delta includes the underlying share position, which is the whole point
        when a book is delta hedged: the option delta and the share delta have
        to be looked at together or the hedge is invisible.
        """
        total = {"delta": float(self.shares), "gamma": 0.0,
                 "vega": 0.0, "theta": 0.0, "rho": 0.0}
        price_total = 0.0

        for contract, qty in self.options.items():
            if abs(qty) < 1e-12:
                continue
            vol = vols.get(contract)
            if vol is None or not np.isfinite(vol):
                continue
            tte = contract.time_to_expiry(date)
            g = black_scholes_greeks(spot, contract.strike, tte, rate, vol,
                                     contract.option_type, dividend_yield)
            scale = qty * CONTRACT_MULTIPLIER
            total["delta"] += g.delta * scale
            total["gamma"] += g.gamma * scale
            total["vega"] += g.vega * scale
            total["theta"] += g.theta * scale
            total["rho"] += g.rho * scale
            price_total += g.price * scale

        return Greeks(price=price_total, **total)

    def settle_expiry(self, contract: OptionContract, spot: float, date,
                      allow_assignment: bool = True) -> Optional[str]:
        """Settle a contract at expiry: exercise, assignment, or expire worthless.

        Physical settlement, which is what single-stock options actually do:
        an in-the-money short call delivers shares at the strike, an in-the-money
        short put takes delivery at the strike. That distinction matters because
        assignment leaves you holding (or short) stock, which is exactly how a
        covered call turns into a naked long and a cash-secured put turns into
        a stock position.

        Returns a short description of what happened, or None if nothing did.
        """
        qty = self.position(contract)
        if abs(qty) < 1e-12:
            return None

        itm = contract.intrinsic(spot) > 0
        shares = abs(qty) * CONTRACT_MULTIPLIER
        self.options[contract] = 0.0

        if not itm:
            return f"{contract} expired worthless"

        if not allow_assignment:
            # Cash settle at intrinsic instead.
            self.cash += contract.intrinsic(spot) * qty * CONTRACT_MULTIPLIER
            return f"{contract} cash settled at intrinsic"

        if contract.option_type == CALL:
            if qty < 0:                     # short call assigned: deliver shares
                self.shares -= shares
                self.cash += contract.strike * shares
                return f"{contract} assigned: delivered {shares:.0f} shares at {contract.strike:g}"
            self.shares += shares           # long call exercised: take delivery
            self.cash -= contract.strike * shares
            return f"{contract} exercised: bought {shares:.0f} shares at {contract.strike:g}"

        if qty < 0:                         # short put assigned: take delivery
            self.shares += shares
            self.cash -= contract.strike * shares
            return f"{contract} assigned: bought {shares:.0f} shares at {contract.strike:g}"
        self.shares -= shares               # long put exercised: deliver shares
        self.cash += contract.strike * shares
        return f"{contract} exercised: sold {shares:.0f} shares at {contract.strike:g}"
