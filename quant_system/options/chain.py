"""Option chain ingestion: live quotes from yfinance, or a synthetic fallback.

A word on data, because it determines what the rest of the options leg can
honestly claim.

yfinance serves the *current* option chain. There is no history in it: you
cannot ask it what the September 400-strike was worth last March. Backtesting
an options strategy properly needs an end-of-day history of the whole chain,
and every source of that is paid (OptionMetrics, ORATS, CBOE DataShop, and the
cheaper OptionsDX archives). This package does not pretend otherwise. Live
chains are used for surface building and the paper-book repricing; the options
backtest runs on a synthetic chain, labelled as such everywhere it surfaces,
exactly like the equity side's synthetic price fallback.

The other thing real chains teach you quickly is that most quotes are junk. On
a sample SPY chain, 44 of 95 strikes had a zero bid: nobody is buying at any
price, so there is no market to trade against and no information to extract.
Filtering those is not cosmetic, it is the difference between a smile and
noise. Every filter here reports what it removed instead of silently shrinking
the dataset.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import OptionsConfig
from .pricing import CALL, PUT, black_scholes_price

logger = logging.getLogger(__name__)

# Calendar-day count. Options expire on a date, and time value decays over
# calendar time (weekends included), so ACT/365 is the honest convention here
# even though the equity side counts trading days.
DAYS_PER_YEAR = 365.0

QUOTE_COLUMNS = [
    "expiry", "time_to_expiry", "strike", "option_type",
    "bid", "ask", "mid", "volume", "open_interest",
]


@dataclass
class OptionChain:
    """A cleaned set of option quotes for one underlying at one moment.

    Attributes
    ----------
    quotes : pd.DataFrame
        One row per contract, columns as in ``QUOTE_COLUMNS``.
    spot : float
        Underlying price at ``as_of``.
    as_of : pd.Timestamp
        Observation time. Everything downstream is dated from here.
    ticker : str
    synthetic : bool
        True when generated rather than observed. Propagates into the surface
        and any report built from it.
    dropped : dict
        Rows removed per filter reason. Kept so the loss is auditable.
    """

    quotes: pd.DataFrame
    spot: float
    as_of: pd.Timestamp
    ticker: str
    synthetic: bool = False
    dropped: Dict[str, int] = field(default_factory=dict)

    @property
    def expiries(self) -> List[pd.Timestamp]:
        return sorted(self.quotes["expiry"].unique())

    @property
    def n_quotes(self) -> int:
        return len(self.quotes)

    def slice(self, expiry) -> pd.DataFrame:
        """Quotes for a single expiry, sorted by strike."""
        return (self.quotes[self.quotes["expiry"] == pd.Timestamp(expiry)]
                .sort_values("strike").reset_index(drop=True))

    def summary(self) -> List[str]:
        label = "SYNTHETIC" if self.synthetic else "live"
        lines = [
            f"OPTION CHAIN {self.ticker} ({label}) as of {self.as_of:%Y-%m-%d}",
            f" spot {self.spot:.2f}, {self.n_quotes} usable quotes "
            f"across {len(self.expiries)} expiries",
        ]
        if self.dropped:
            total = sum(self.dropped.values())
            detail = ", ".join(f"{k}={v}" for k, v in sorted(self.dropped.items()))
            lines.append(f" dropped {total} raw quotes: {detail}")
        return lines


def _clean_quotes(raw: pd.DataFrame, cfg: OptionsConfig) -> tuple:
    """Apply the hygiene filters, returning (clean_frame, dropped_counts)."""
    dropped: Dict[str, int] = {}
    n0 = len(raw)
    df = raw.copy()

    def drop(mask: pd.Series, reason: str) -> None:
        nonlocal df
        n = int(mask.sum())
        if n:
            dropped[reason] = dropped.get(reason, 0) + n
            df = df[~mask]

    df = df[np.isfinite(df["bid"]) & np.isfinite(df["ask"])]
    dropped_nonfinite = n0 - len(df)
    if dropped_nonfinite:
        dropped["non_finite"] = dropped_nonfinite

    drop(df["bid"] < cfg.min_bid, "zero_or_low_bid")
    drop(df["ask"] <= df["bid"], "crossed_or_locked")

    mid = 0.5 * (df["bid"] + df["ask"])
    with np.errstate(divide="ignore", invalid="ignore"):
        spread_ratio = (df["ask"] - df["bid"]) / mid.replace(0.0, np.nan)
    drop(spread_ratio > cfg.max_spread_ratio, "spread_too_wide")

    drop(df["time_to_expiry"] <= 0, "expired")
    drop(df["time_to_expiry"] * DAYS_PER_YEAR < cfg.min_days_to_expiry, "too_near_expiry")

    if cfg.min_open_interest > 0:
        drop(df["open_interest"].fillna(0) < cfg.min_open_interest, "thin_open_interest")

    df = df.copy()
    df["mid"] = 0.5 * (df["bid"] + df["ask"])
    return df.reset_index(drop=True), dropped


def load_option_chain(
    ticker: str,
    cfg: Optional[OptionsConfig] = None,
    max_expiries: int = 8,
    as_of: Optional[pd.Timestamp] = None,
    expiry_selection: str = "spread",
) -> OptionChain:
    """Fetch and clean the current option chain for ``ticker`` from yfinance.

    Parameters
    ----------
    max_expiries : int
        Cap on how many expiries to pull. Each is a separate HTTP round trip,
        and the far tail of a chain is mostly untradeable anyway.
    expiry_selection : {"spread", "nearest"}
        ``"spread"`` samples evenly across the whole available range, which is
        what you want for a term structure: on an underlying with daily
        expiries, taking the nearest few gives five points all inside a week
        and no information about the term structure at all. ``"nearest"`` keeps
        the front of the chain, which is the right choice for short-dated work.

    Raises
    ------
    RuntimeError
        If the chain cannot be fetched or nothing survives cleaning. Failing
        loudly beats returning an empty surface that silently prices at NaN.
    """
    cfg = cfg or OptionsConfig()
    try:
        import yfinance as yf
    except Exception as exc:                                  # pragma: no cover
        raise RuntimeError(f"yfinance unavailable: {exc}") from exc

    tk = yf.Ticker(ticker)
    try:
        expiries = list(tk.options)
    except Exception as exc:
        raise RuntimeError(f"could not list expiries for {ticker}: {exc}") from exc
    if not expiries:
        raise RuntimeError(f"no option expiries returned for {ticker}")

    spot = _fetch_spot(tk, ticker)
    stamp = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.utcnow().normalize()
    if stamp.tzinfo is not None:
        stamp = stamp.tz_localize(None)

    if expiry_selection == "spread" and len(expiries) > max_expiries:
        idx = np.unique(np.linspace(0, len(expiries) - 1, max_expiries).astype(int))
        chosen = [expiries[i] for i in idx]
    else:
        chosen = expiries[:max_expiries]

    frames = []
    for exp in chosen:
        try:
            chain = tk.option_chain(exp)
        except Exception as exc:
            logger.warning("chain fetch failed for %s %s: %s", ticker, exp, exc)
            continue
        exp_ts = pd.Timestamp(exp)
        tte = (exp_ts - stamp).days / DAYS_PER_YEAR
        for side, frame in ((CALL, chain.calls), (PUT, chain.puts)):
            if frame is None or frame.empty:
                continue
            part = pd.DataFrame({
                "expiry": exp_ts,
                "time_to_expiry": tte,
                "strike": frame["strike"].astype(float),
                "option_type": side,
                "bid": frame["bid"].astype(float),
                "ask": frame["ask"].astype(float),
                "volume": frame.get("volume", pd.Series(index=frame.index, dtype=float)),
                "open_interest": frame.get("openInterest",
                                           pd.Series(index=frame.index, dtype=float)),
            })
            frames.append(part)

    if not frames:
        raise RuntimeError(f"no option data retrieved for {ticker}")

    raw = pd.concat(frames, ignore_index=True)
    clean, dropped = _clean_quotes(raw, cfg)
    if clean.empty:
        raise RuntimeError(
            f"every quote for {ticker} failed the hygiene filters: {dropped}")

    chain_obj = OptionChain(quotes=clean[QUOTE_COLUMNS], spot=spot, as_of=stamp,
                            ticker=ticker, synthetic=False, dropped=dropped)
    logger.info("loaded %s: %d usable quotes from %d raw",
                ticker, len(clean), len(raw))
    return chain_obj


def _fetch_spot(tk, ticker: str) -> float:
    """Last close for the underlying. Raises if it cannot be determined."""
    try:
        hist = tk.history(period="5d")
        if hist is not None and not hist.empty:
            return float(hist["Close"].dropna().iloc[-1])
    except Exception:
        pass
    raise RuntimeError(f"could not determine spot price for {ticker}")


def synthetic_option_chain(
    ticker: str = "SYNTH",
    spot: float = 100.0,
    as_of: Optional[pd.Timestamp] = None,
    expiries_days: tuple = (30, 60, 91, 182, 365),
    moneyness: tuple = (0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20),
    base_vol: float = 0.22,
    skew: float = -0.10,
    curvature: float = 0.35,
    term_slope: float = 0.02,
    rate: float = 0.04,
    dividend_yield: float = 0.0,
    half_spread_frac: float = 0.02,
    cfg: Optional[OptionsConfig] = None,
) -> OptionChain:
    """Generate a labelled SYNTHETIC chain with a realistic smile and term structure.

    Volatility follows
    ``sigma(k, T) = base_vol + term_slope*sqrt(T) + skew*k + curvature*k^2``
    in log-forward-moneyness ``k``. Negative ``skew`` reproduces the equity
    pattern where downside puts trade at higher implied vol than upside calls.
    Prices are then generated with Black-Scholes at that vol and wrapped in a
    symmetric bid/ask.

    Because prices come from a smooth, well-behaved vol function, the resulting
    chain is essentially arbitrage-free, which makes it the right baseline for
    testing that the arbitrage checker does *not* cry wolf on clean data.

    Everything downstream carries ``synthetic=True``. These are not market
    prices and must never be presented as such.
    """
    cfg = cfg or OptionsConfig()
    stamp = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp("2025-01-02")
    rows = []

    for days in expiries_days:
        tte = days / DAYS_PER_YEAR
        expiry = stamp + pd.Timedelta(days=int(days))
        forward = spot * np.exp((rate - dividend_yield) * tte)
        for m in moneyness:
            strike = round(spot * m, 2)
            k = float(np.log(strike / forward))
            vol = base_vol + term_slope * np.sqrt(tte) + skew * k + curvature * k ** 2
            vol = float(max(vol, 0.02))
            for side in (CALL, PUT):
                price = float(black_scholes_price(spot, strike, tte, rate, vol,
                                                  side, dividend_yield))
                half = max(half_spread_frac * price, 0.01)
                rows.append({
                    "expiry": expiry,
                    "time_to_expiry": tte,
                    "strike": float(strike),
                    "option_type": side,
                    "bid": max(price - half, 0.0),
                    "ask": price + half,
                    "volume": 100.0,
                    "open_interest": 500.0,
                })

    raw = pd.DataFrame(rows)
    clean, dropped = _clean_quotes(raw, cfg)
    logger.info("generated SYNTHETIC chain for %s: %d quotes", ticker, len(clean))
    return OptionChain(quotes=clean[QUOTE_COLUMNS], spot=spot, as_of=stamp,
                       ticker=ticker, synthetic=True, dropped=dropped)
