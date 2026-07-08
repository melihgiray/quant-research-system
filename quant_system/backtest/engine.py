"""Event-driven backtest engine with strict look-ahead prevention.

The execution-timing convention:

    A weight on row dated T is the target portfolio formed using information
    available *at the close of T*. The engine earns that weight on day **T+1**.

The one-day execution lag is implemented in exactly ONE place - ``held = weights.shift(1)``
inside :func:`portfolio_returns`. Signal modules are therefore free to compute a
weight from same-day data (e.g. today's close), because the engine guarantees that
weight is only ever *applied* to the next day's return. Centralising the lag means
there is a single, auditable line responsible for preventing look-ahead, rather
than ``.shift()`` calls scattered (and occasionally forgotten) across signals.

Volatility and ADV used for cost estimation are likewise lagged by one day, so a
fill on day T is priced using liquidity/vol known strictly before T.

:func:`portfolio_returns` is pure: ``(weights, returns) -> portfolio return series``,
with no I/O and no globals, so it is easy to unit-test on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from ..config import CostConfig, RiskConfig
from .costs import transaction_cost_fraction


@dataclass
class BacktestResult:
    """Everything the engine produces. No side effects; pure data."""

    gross_returns: pd.Series        # before costs, after the 1-day lag
    costs: pd.Series                # transaction cost per day (fraction)
    net_returns: pd.Series          # gross - costs (before any risk stop)
    returns: pd.Series              # canonical stream to analyse (after risk stop, if any)
    turnover: pd.Series             # one-way daily turnover (fraction of portfolio)
    held_weights: pd.DataFrame      # actual weights held each day (already lagged)
    equity: pd.Series               # cumulative growth of $1 on `returns`
    meta: dict = field(default_factory=dict)

    @property
    def annual_turnover(self) -> float:
        """Average annualised one-way turnover (x per year)."""
        from ..config import TRADING_DAYS_PER_YEAR
        return float(self.turnover.mean() * TRADING_DAYS_PER_YEAR)


def assert_no_lookahead(weights: pd.DataFrame, prices: pd.DataFrame) -> None:
    """Structural guard against accidental forward-looking signals.

    Checks that:
      * every weight date exists in the price index (no phantom dates),
      * weight columns are a subset of price columns,
      * the index is monotonic increasing (no shuffled dates that could let a
        naive ``shift`` leak the future).

    The *temporal* guarantee (a weight only earns the next day's return) is
    enforced by the ``shift(1)`` in :func:`portfolio_returns`; this assertion
    catches the structural mistakes that would undermine it.
    """
    if not weights.index.is_monotonic_increasing:
        raise AssertionError("weights index is not sorted; shift(1) would be meaningless")
    if not prices.index.is_monotonic_increasing:
        raise AssertionError("price index is not sorted")
    missing_dates = weights.index.difference(prices.index)
    if len(missing_dates) > 0:
        raise AssertionError(
            f"{len(missing_dates)} weight date(s) absent from price index, e.g. {missing_dates[:3].tolist()}"
        )
    extra_cols = set(weights.columns) - set(prices.columns)
    if extra_cols:
        raise AssertionError(f"weights reference unknown tickers: {sorted(extra_cols)}")


def portfolio_returns(weights: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    """Pure core: turn target weights + asset returns into a portfolio return series.

    This is the testable heart of the engine. It applies the one-day execution
    lag and nothing else (no costs, no stops), so its behaviour is trivial to
    verify in a unit test.

    Parameters
    ----------
    weights : pd.DataFrame
        Target weights (date x ticker). Held constant between updates (forward
        filled). Row T is the target formed at the close of T.
    returns : pd.DataFrame
        Simple asset returns (date x ticker), r_t = p_t/p_{t-1} - 1.

    Returns
    -------
    pd.Series
        Portfolio return per day, gross of costs.
    """
    w = weights.reindex(index=returns.index, columns=returns.columns).ffill()
    held = w.shift(1).fillna(0.0)            # <-- THE single source of the lag
    return (held * returns).sum(axis=1)


def run_backtest(
    weights: pd.DataFrame,
    price_data,
    cost: Optional[CostConfig] = None,
    risk: Optional[RiskConfig] = None,
    apply_drawdown_stop: bool = False,
    check_lookahead: bool = True,
) -> BacktestResult:
    """Run a full backtest: lagged P&L, realistic costs, optional drawdown stop.

    Parameters
    ----------
    weights : pd.DataFrame
        Daily target weights (date x ticker).
    price_data : PriceData
        Aligned close/volume panel.
    cost : CostConfig, optional
        Cost assumptions. If None, costs are zero (useful for isolating gross P&L).
    risk : RiskConfig, optional
        Used only when ``apply_drawdown_stop`` is True.
    apply_drawdown_stop : bool
        If True, overlay a portfolio max-drawdown stop on the net returns.
    check_lookahead : bool
        Run :func:`assert_no_lookahead` first (recommended).

    Returns
    -------
    BacktestResult
    """
    from ..config import TRADING_DAYS_PER_MONTH

    close = price_data.close
    volume = price_data.volume
    cost = cost or CostConfig()

    if check_lookahead:
        assert_no_lookahead(weights, close)

    returns = close.pct_change()
    cols = returns.columns

    # Held weights after the one-day lag (the actual book each day).
    w = weights.reindex(index=returns.index, columns=cols).ffill()
    held = w.shift(1).fillna(0.0)

    gross = (held * returns).sum(axis=1)

    # Trades executed each day = change in held book. First row establishes the book.
    trades = held.diff()
    if len(held) > 0:
        trades.iloc[0] = held.iloc[0]
    turnover = trades.abs().sum(axis=1)

    # Cost inputs, lagged by one day so a fill on T uses liquidity/vol known < T.
    # Fall back to an expanding-window estimate before the rolling window warms up,
    # so early trades are still charged a sensible (non-zero) impact.
    adv = (volume.rolling(cost.adv_lookback, min_periods=5).mean()
           .fillna(volume.expanding(min_periods=1).mean()).shift(1))
    sigma = (returns.rolling(cost.vol_lookback, min_periods=5).std()
             .fillna(returns.expanding(min_periods=2).std()).shift(1))

    # Per-day transaction cost. The loop runs once per date and is vectorised over
    # names inside; for a few thousand days this is negligible and stays readable.
    cost_vals = np.zeros(len(returns))
    trade_days = np.where(turnover.values > 0)[0]
    for i in trade_days:
        t = returns.index[i]
        cost_vals[i] = transaction_cost_fraction(
            delta_weights=trades.iloc[i],
            prices=close.loc[t],
            adv_shares=adv.loc[t],
            sigma=sigma.loc[t],
            cost=cost,
        )
    costs = pd.Series(cost_vals, index=returns.index)

    net = gross - costs

    if apply_drawdown_stop and risk is not None:
        from ..risk.limits import apply_drawdown_stop as _dd_stop
        final = _dd_stop(net, max_drawdown=risk.max_drawdown,
                         cooldown=TRADING_DAYS_PER_MONTH)
    else:
        final = net

    equity = (1.0 + final.fillna(0.0)).cumprod()

    return BacktestResult(
        gross_returns=gross,
        costs=costs,
        net_returns=net,
        returns=final,
        turnover=turnover,
        held_weights=held,
        equity=equity,
        meta={
            "synthetic": getattr(price_data, "synthetic", False),
            "n_days": int(len(returns)),
            "drawdown_stop": bool(apply_drawdown_stop and risk is not None),
        },
    )
