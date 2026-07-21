"""Event-driven options backtest, with the same timing discipline as the equity engine.

The equity engine's rule is that a weight formed at the close of T earns the
return of T+1, enforced by a single ``shift(1)``. The options engine keeps the
same promise in a form that suits discrete contracts: **a strategy is asked for
orders at the close of T, and those orders execute against T+1's quotes.** The
loop is written so this cannot be forgotten, since orders are held in a pending
queue between iterations rather than executed where they were decided.

That timing is what the clairvoyant test exploits. A strategy that peeks at
tomorrow's underlying move still has to trade after the move has happened, so
it earns nothing. If the engine ever executed orders on the day they were
decided, the same strategy would print money, and the test would fail.

Each day, in order:
  1. Settle any contract that has reached expiry (physical assignment).
  2. Execute the orders decided yesterday, crossing the spread.
  3. Mark the book to mid and record equity, Greeks and exposures.
  4. Ask the strategy for tomorrow's orders.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import TRADING_DAYS_PER_YEAR
from .book import Book, Fill, Order, OptionContract, CONTRACT_MULTIPLIER
from .greeks import Greeks
from .provider import SyntheticChainProvider

logger = logging.getLogger(__name__)


@dataclass
class StrategyContext:
    """What a strategy may look at when deciding orders.

    Everything here is as of ``date``. Orders returned will execute against the
    *next* trading day's quotes.
    """

    date: pd.Timestamp
    spot: float
    equity: float
    book: Book
    provider: SyntheticChainProvider


Strategy = Callable[[StrategyContext], List[Order]]


@dataclass
class OptionsBacktestResult:
    """Everything the options engine produces."""

    equity: pd.Series
    returns: pd.Series
    greeks: pd.DataFrame
    exposures: pd.DataFrame
    trades: List[Fill] = field(default_factory=list)
    events: List[str] = field(default_factory=list)
    spread_cost: float = 0.0
    initial_capital: float = 0.0
    synthetic: bool = True
    label: str = "options strategy"

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    def summary(self) -> List[str]:
        from ..performance.analytics import compute_metrics

        r = self.returns.dropna()
        m = compute_metrics(r) if len(r) > 2 else {}
        tag = "SYNTHETIC" if self.synthetic else "live"
        lines = [
            f"{self.label.upper()} ({tag})",
            f" period {self.equity.index[0]:%Y-%m-%d} to {self.equity.index[-1]:%Y-%m-%d}"
            f"  ({len(self.equity)} days)",
        ]
        if m:
            lines += [
                f" total return    {self.equity.iloc[-1] / self.initial_capital - 1:+.2%}",
                f" annualised      {m['ann_return']:+.2%}   vol {m['ann_vol']:.2%}",
                f" Sharpe          {m['sharpe']:.2f}   Sortino {m['sortino']:.2f}",
                f" max drawdown    {m['max_drawdown']:.2%}",
                f" hit rate        {m['hit_rate']:.1%}",
            ]
        lines += [
            f" trades          {self.n_trades}   "
            f"spread paid ${self.spread_cost:,.0f} "
            f"({self.spread_cost / self.initial_capital:.2%} of capital)",
        ]
        if not self.greeks.empty:
            g = self.greeks
            lines.append(
                f" mean Greeks     delta {g['delta'].mean():+.1f}  "
                f"gamma {g['gamma'].mean():+.2f}  "
                f"vega {g['vega'].mean():+.1f}  "
                f"theta/day {g['theta'].mean() / 365:+.1f}")
        assignments = [e for e in self.events if "assigned" in e]
        if assignments:
            lines.append(f" assignments     {len(assignments)}")
        return lines


def run_options_backtest(
    provider: SyntheticChainProvider,
    strategy: Strategy,
    initial_capital: float = 100_000.0,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    label: str = "options strategy",
    allow_assignment: bool = True,
) -> OptionsBacktestResult:
    """Run a strategy against a chain provider.

    Parameters
    ----------
    provider : SyntheticChainProvider
        Supplies spot, quotes and the listed grid on every date.
    strategy : callable(StrategyContext) -> list[Order]
        Consulted at each close; its orders fill on the next trading day.
    initial_capital : float
        Starting cash.

    Returns
    -------
    OptionsBacktestResult
    """
    dates = provider.dates
    if start is not None:
        dates = dates[dates >= pd.Timestamp(start)]
    if end is not None:
        dates = dates[dates <= pd.Timestamp(end)]
    if len(dates) < 3:
        raise ValueError("need at least three dates to run an options backtest")

    book = Book(cash=float(initial_capital))
    pending: List[Order] = []
    trades: List[Fill] = []
    events: List[str] = []

    equity_rows, greek_rows, exposure_rows = [], [], []

    for date in dates:
        spot = provider.spot(date)

        # 1. Settle anything that has reached expiry. Weekend expiries settle on
        #    the next trading day, which is why the test is <= rather than ==.
        for contract, qty in list(book.open_contracts().items()):
            if contract.expiry <= date:
                note = book.settle_expiry(contract, spot, date,
                                          allow_assignment=allow_assignment)
                if note:
                    events.append(f"{date:%Y-%m-%d} {note}")

        # 2. Execute yesterday's decisions against today's market.
        for order in pending:
            try:
                if order.is_underlying:
                    bid, ask = provider.underlying_quote(date)
                    fill = book.trade_underlying(order.quantity, bid, ask, date)
                else:
                    if order.contract.expiry <= date:
                        continue                      # cannot open an expired contract
                    bid, ask = provider.quote(date, order.contract)
                    fill = book.trade_option(order.contract, order.quantity,
                                             bid, ask, date)
                trades.append(fill)
            except ValueError as exc:
                logger.warning("%s order rejected on %s: %s", order.reason, date, exc)
        pending = []

        # 3. Mark to market.
        marks = {c: provider.mid(date, c) for c in book.open_contracts()}
        vols = {c: provider.implied_vol(date, c) for c in book.open_contracts()}
        equity = book.equity(spot, marks)
        greeks = book.greeks(spot, date, provider.rate, vols, provider.dividend_yield)

        equity_rows.append((date, equity))
        greek_rows.append((date, greeks.delta, greeks.gamma, greeks.vega,
                           greeks.theta, greeks.rho))
        exposure_rows.append((date, book.cash, book.shares,
                              len(book.open_contracts()), spot))

        # 4. Decide tomorrow's orders. They cannot touch today's fills.
        ctx = StrategyContext(date=date, spot=spot, equity=equity,
                              book=book, provider=provider)
        pending = list(strategy(ctx) or [])

    equity_series = pd.Series([e for _, e in equity_rows],
                              index=pd.DatetimeIndex([d for d, _ in equity_rows]),
                              name="equity")
    returns = equity_series.pct_change()
    # A wiped-out account has no meaningful return series beyond that point.
    if (equity_series <= 0).any():
        first_zero = equity_series.index[equity_series <= 0][0]
        logger.warning("account value hit zero on %s", first_zero)
        returns = returns.loc[:first_zero]

    greeks_df = pd.DataFrame(
        greek_rows, columns=["date", "delta", "gamma", "vega", "theta", "rho"]
    ).set_index("date")
    exposures_df = pd.DataFrame(
        exposure_rows, columns=["date", "cash", "shares", "n_options", "spot"]
    ).set_index("date")

    return OptionsBacktestResult(
        equity=equity_series,
        returns=returns,
        greeks=greeks_df,
        exposures=exposures_df,
        trades=trades,
        events=events,
        spread_cost=book.realised_spread_cost,
        initial_capital=float(initial_capital),
        synthetic=provider.synthetic,
        label=label,
    )
