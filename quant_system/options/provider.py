"""A chain of option quotes through time, for backtesting.

This is the piece that historical data would replace. yfinance gives the
current chain only, so an options backtest needs either a paid EOD options
history (OptionMetrics, ORATS, CBOE DataShop, OptionsDX) or a generated one.
This module generates one, and everything built on it is labelled SYNTHETIC,
the same convention the equity side uses for its price fallback.

The generator is deliberately explicit about the one assumption that decides
the answer for every short-volatility strategy: the **variance risk premium**.
Implied volatility is set to trailing realised volatility times
``(1 + vol_premium)``. With ``vol_premium = 0.15``, options are priced 15%
above the volatility that will actually be realised, so selling them wins on
average. That is a real and well documented feature of equity index options,
but here it is an input, not a discovery. A short-straddle result produced on
this data demonstrates that the machinery correctly harvests a premium that was
put there on purpose. It is not evidence the premium exists. Setting
``vol_premium=0.0`` turns it off, and the strategies should then earn roughly
nothing before costs and lose after them, which is a useful sanity check.

Implied vol is built only from data available at the time (trailing realised
vol, lagged a day), so the surface itself never peeks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .book import OptionContract
from .greeks import black_scholes_greeks
from .pricing import CALL, PUT, black_scholes_price

logger = logging.getLogger(__name__)

DAYS_PER_YEAR = 365.0
TRADING_DAYS = 252


@dataclass
class SyntheticChainProvider:
    """Quotes any option on any date, from a given underlying path.

    Parameters
    ----------
    underlying : pd.Series
        Close price indexed by date. The backtest walks this index.
    rate, dividend_yield : float
        Carry assumptions used for pricing and for the forward.
    vol_premium : float
        Implied over trailing realised. See the module docstring: this is the
        assumption that decides short-vol results.
    vol_window : int
        Trading days of trailing realised vol feeding implied vol.
    skew, curvature : float
        Smile shape in log-forward-moneyness. Negative skew reproduces the
        equity pattern of expensive downside puts.
    half_spread_frac, min_half_spread : float
        Option market width. Options are wide; this is where covered-call
        backtests usually flatter themselves.
    equity_half_spread_bps : float
        Width of the underlying market, for delta hedging.
    expiry_interval_days, n_expiries : int
        A monthly-style expiry ladder.
    strike_step_frac : float
        Strike grid spacing as a fraction of the initial spot, rounded to a
        sensible increment, so strikes are fixed levels rather than moving
        with spot.
    """

    underlying: pd.Series
    rate: float = 0.04
    dividend_yield: float = 0.0
    vol_premium: float = 0.15
    vol_window: int = 21
    base_vol_floor: float = 0.05
    skew: float = -0.35
    curvature: float = 0.60
    half_spread_frac: float = 0.025
    min_half_spread: float = 0.02
    equity_half_spread_bps: float = 1.0
    expiry_interval_days: int = 28
    n_expiries: int = 6
    strike_step_frac: float = 0.025
    strike_range_frac: float = 0.40
    synthetic: bool = True

    _implied_atm: pd.Series = field(init=False, repr=False)
    _expiries: List[pd.Timestamp] = field(init=False, repr=False)
    _strike_step: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.underlying.index, pd.DatetimeIndex):
            raise TypeError("underlying must be indexed by date")
        if len(self.underlying) < self.vol_window + 2:
            raise ValueError("underlying path is too short to estimate volatility")

        returns = self.underlying.pct_change()
        realised = returns.rolling(self.vol_window,
                                   min_periods=max(self.vol_window // 2, 2)).std()
        realised = realised * np.sqrt(TRADING_DAYS)
        # Lag one day: today's quoted vol may only use vol known before today.
        realised = realised.shift(1).bfill()
        self._implied_atm = (realised.clip(lower=self.base_vol_floor)
                             * (1.0 + self.vol_premium))

        start = self.underlying.index[0]
        end = self.underlying.index[-1]
        expiries, d = [], start
        while d <= end + pd.Timedelta(days=self.expiry_interval_days * self.n_expiries):
            d = d + pd.Timedelta(days=self.expiry_interval_days)
            expiries.append(pd.Timestamp(d).normalize())
        self._expiries = expiries

        step = float(self.underlying.iloc[0]) * self.strike_step_frac
        # Round to a familiar increment so strikes look like listed strikes.
        for increment in (0.5, 1.0, 2.5, 5.0, 10.0, 25.0):
            if step <= increment:
                step = increment
                break
        self._strike_step = step

    # ------------------------------------------------------------------ #
    # Market state
    # ------------------------------------------------------------------ #
    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.underlying.index

    def spot(self, date) -> float:
        return float(self.underlying.loc[pd.Timestamp(date)])

    def underlying_quote(self, date) -> Tuple[float, float]:
        """(bid, ask) for the underlying, so share trades also cross a spread."""
        mid = self.spot(date)
        half = mid * self.equity_half_spread_bps / 1e4
        return mid - half, mid + half

    def atm_vol(self, date) -> float:
        return float(self._implied_atm.loc[pd.Timestamp(date)])

    def implied_vol(self, date, contract: OptionContract) -> float:
        """Implied vol for a contract: ATM level shaped by the smile."""
        tte = contract.time_to_expiry(date, DAYS_PER_YEAR)
        if tte <= 0:
            return float("nan")
        spot = self.spot(date)
        forward = spot * np.exp((self.rate - self.dividend_yield) * tte)
        k = float(np.log(contract.strike / forward))
        vol = self.atm_vol(date) * (1.0 + self.skew * k + self.curvature * k ** 2)
        return float(max(vol, 0.01))

    def theoretical_price(self, date, contract: OptionContract) -> float:
        tte = contract.time_to_expiry(date, DAYS_PER_YEAR)
        spot = self.spot(date)
        if tte <= 0:
            return contract.intrinsic(spot)
        return float(black_scholes_price(spot, contract.strike, tte, self.rate,
                                         self.implied_vol(date, contract),
                                         contract.option_type, self.dividend_yield))

    def quote(self, date, contract: OptionContract) -> Tuple[float, float]:
        """(bid, ask) for any contract, listed or not.

        Held positions must remain quotable even after spot drifts away from the
        listed grid, so this prices anything. ``list_strikes`` is what governs
        which contracts may be *opened*.
        """
        price = self.theoretical_price(date, contract)
        half = max(self.half_spread_frac * price, self.min_half_spread)
        return max(price - half, 0.0), price + half

    def mid(self, date, contract: OptionContract) -> float:
        bid, ask = self.quote(date, contract)
        return 0.5 * (bid + ask)

    # ------------------------------------------------------------------ #
    # What is listed
    # ------------------------------------------------------------------ #
    def list_expiries(self, date, min_days: int = 1,
                      max_days: int = 400) -> List[pd.Timestamp]:
        d = pd.Timestamp(date)
        return [e for e in self._expiries
                if min_days <= (e - d).days <= max_days]

    def nearest_expiry(self, date, target_days: int) -> Optional[pd.Timestamp]:
        candidates = self.list_expiries(date, min_days=1, max_days=target_days * 4)
        if not candidates:
            return None
        return min(candidates, key=lambda e: abs((e - pd.Timestamp(date)).days - target_days))

    def list_strikes(self, date) -> List[float]:
        spot = self.spot(date)
        lo = spot * (1.0 - self.strike_range_frac)
        hi = spot * (1.0 + self.strike_range_frac)
        step = self._strike_step
        first = np.ceil(lo / step) * step
        return [round(float(k), 4) for k in np.arange(first, hi + step, step)]

    def nearest_strike(self, date, target: float) -> float:
        strikes = self.list_strikes(date)
        if not strikes:
            raise RuntimeError(f"no strikes listed on {date}")
        return min(strikes, key=lambda k: abs(k - target))

    def contract_by_delta(self, date, expiry, target_delta: float,
                          option_type: str) -> Optional[OptionContract]:
        """Listed contract whose delta is closest to ``target_delta``.

        Delta is the natural way to express "sell a 30-delta call": it keeps the
        moneyness of the position stable as volatility changes, which strike
        offsets do not.
        """
        best, best_gap = None, np.inf
        for strike in self.list_strikes(date):
            contract = OptionContract(pd.Timestamp(expiry), strike, option_type)
            tte = contract.time_to_expiry(date, DAYS_PER_YEAR)
            if tte <= 0:
                continue
            vol = self.implied_vol(date, contract)
            g = black_scholes_greeks(self.spot(date), strike, tte, self.rate, vol,
                                     option_type, self.dividend_yield)
            gap = abs(abs(g.delta) - abs(target_delta))
            if gap < best_gap:
                best, best_gap = contract, gap
        return best

    def summary(self) -> List[str]:
        vols = self._implied_atm.dropna()
        return [
            f"SYNTHETIC CHAIN PROVIDER ({len(self.dates)} days, "
            f"{self.dates[0]:%Y-%m-%d} to {self.dates[-1]:%Y-%m-%d})",
            f" implied ATM vol {vols.min():.1%} to {vols.max():.1%} "
            f"(mean {vols.mean():.1%}), variance premium {self.vol_premium:+.0%}",
            f" strike step {self._strike_step:g}, "
            f"option half-spread {self.half_spread_frac:.1%} of premium",
        ]


def build_provider_from_prices(close: pd.Series, **kwargs) -> SyntheticChainProvider:
    """Convenience wrapper: chain provider from any underlying close series."""
    return SyntheticChainProvider(underlying=close.astype(float), **kwargs)
