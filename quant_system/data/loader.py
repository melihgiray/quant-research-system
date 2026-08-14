"""Price/volume loader with on-disk caching and an offline synthetic fallback.

Why caching: yfinance hits Yahoo's public endpoints, which rate-limit (HTTP 429)
aggressively. Every other module depends on data, so we fetch once, cache to
Parquet, and reuse. A run should never hammer the API for data it already has.

Why a synthetic fallback: the backtest engine, analytics and signals must be
*testable in isolation*, with no network dependency. When Yahoo is unreachable
(or ``use_synthetic`` is set), we generate a deterministic, cross-correlated
panel so the whole system still runs end-to-end and produces a real tearsheet.
The synthetic data is clearly labelled so it is never mistaken for live results.
"""

from __future__ import annotations

import hashlib
import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class PriceData:
    """Container for an aligned panel of adjusted closes and volumes.

    Attributes
    ----------
    close : pd.DataFrame
        Adjusted close prices, indexed by date (DatetimeIndex), columns=tickers.
    volume : pd.DataFrame
        Share volume, same index/columns as ``close``.
    synthetic : bool
        True if this panel came from the offline generator rather than yfinance.
    open : pd.DataFrame, optional
        Adjusted open prices, same index/columns as ``close`` when present. Used
        only by the next-open fill mode; None when the source did not carry it.
    """

    close: pd.DataFrame
    volume: pd.DataFrame
    synthetic: bool = False
    open: Optional[pd.DataFrame] = None

    @property
    def tickers(self) -> List[str]:
        return list(self.close.columns)

    @property
    def has_open(self) -> bool:
        """True when open prices are available for the next-open fill mode."""
        return self.open is not None

    def returns(self, kind: str = "simple") -> pd.DataFrame:
        """Daily returns.

        Parameters
        ----------
        kind : {"simple", "log"}
            "simple" -> p_t / p_{t-1} - 1 (used for P&L aggregation).
            "log"    -> ln(p_t / p_{t-1}) (used for vol/regime estimation).
        """
        if kind == "log":
            return np.log(self.close).diff()
        # pandas 3.0 defaults pct_change(fill_method=None): no forward fill, which
        # is exactly what we want (a gap stays a NaN rather than being faked).
        return self.close.pct_change()

    def subset(self, tickers: Sequence[str]) -> "PriceData":
        """Return a new PriceData restricted to ``tickers`` (order preserved)."""
        cols = [t for t in tickers if t in self.close.columns]
        open_sub = self.open[cols].copy() if self.open is not None else None
        return PriceData(self.close[cols].copy(), self.volume[cols].copy(),
                         self.synthetic, open_sub)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def load_price_data(
    tickers: Sequence[str],
    start: str,
    end: str,
    cache_dir: str = "data_cache",
    use_synthetic: bool = False,
    allow_synthetic_fallback: bool = True,
    seed: int = 7,
    verbose: bool = True,
) -> PriceData:
    """Load adjusted close + volume for ``tickers`` over ``[start, end]``.

    Resolution order per the caching strategy:
      1. If ``use_synthetic`` -> generate offline data (no network at all).
      2. Else try the per-ticker Parquet cache; fetch any missing/short tickers
         from yfinance and update the cache.
      3. If yfinance fails for *any* required ticker and fallback is allowed,
         regenerate the **entire** panel synthetically so it stays internally
         consistent (we never silently mix live and synthetic series).

    Returns
    -------
    PriceData
        Aligned panel. ``.synthetic`` flags the data source.
    """
    tickers = list(dict.fromkeys(tickers))  # de-dup, keep order

    if use_synthetic:
        if verbose:
            print("[loader] using synthetic data (use_synthetic=True)")
        return _synthetic_panel(tickers, start, end, seed)

    try:
        close, volume, opens = _load_real(tickers, start, end, cache_dir, verbose)
        panel = _align(close, volume, tickers, opens)
        if panel.close.shape[1] == 0:
            raise RuntimeError("no tickers returned any data")
        return panel
    except Exception as exc:  # network/429/empty - degrade gracefully
        if not allow_synthetic_fallback:
            raise
        warnings.warn(
            f"[loader] live fetch failed ({exc}); falling back to SYNTHETIC data. "
            f"Results are illustrative, not live."
        )
        return _synthetic_panel(tickers, start, end, seed)


# --------------------------------------------------------------------------- #
# Real data via yfinance, cached per ticker
# --------------------------------------------------------------------------- #
def _cache_path(cache_dir: str, ticker: str) -> str:
    safe = ticker.replace("/", "_").replace("\\", "_")
    return os.path.join(cache_dir, f"{safe}.parquet")


def _load_real(tickers, start, end, cache_dir, verbose):
    """Return (close_df, volume_df, opens) assembled from cache + yfinance.

    ``opens`` maps ticker -> open series for the tickers that carried one (older
    caches predate the open column); callers get open prices only when every
    loaded ticker has them.
    """
    os.makedirs(cache_dir, exist_ok=True)
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)

    closes, volumes, opens = {}, {}, {}
    to_fetch = []
    for t in tickers:
        cached = _read_cache(cache_dir, t)
        if cached is not None and cached.index.min() <= start_ts and cached.index.max() >= end_ts:
            sl = cached.loc[start_ts:end_ts]
            closes[t], volumes[t] = sl["close"], sl["volume"]
            if "open" in sl.columns:
                opens[t] = sl["open"]
        else:
            to_fetch.append(t)

    if to_fetch:
        if verbose:
            print(f"[loader] fetching {len(to_fetch)} ticker(s) from yfinance: {', '.join(to_fetch)}")
        fetched = _yf_download(to_fetch, start, end)
        for t in to_fetch:
            df = fetched.get(t)
            if df is None or df.empty:
                continue
            _write_cache(cache_dir, t, df)
            sl = df.loc[start_ts:end_ts]
            closes[t], volumes[t] = sl["close"], sl["volume"]
            if "open" in sl.columns:
                opens[t] = sl["open"]

    if not closes:
        raise RuntimeError("yfinance returned no usable data for any ticker")

    close = pd.DataFrame(closes).sort_index()
    volume = pd.DataFrame(volumes).reindex_like(close)
    return close, volume, opens


def _read_cache(cache_dir, ticker) -> Optional[pd.DataFrame]:
    path = _cache_path(cache_dir, ticker)
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        return df.sort_index()
    except Exception:
        return None


def _write_cache(cache_dir, ticker, df) -> None:
    path = _cache_path(cache_dir, ticker)
    try:
        df.to_parquet(path)
    except Exception as exc:  # caching is best-effort, never fatal
        warnings.warn(f"[loader] could not cache {ticker}: {exc}")


def _yf_download(tickers, start, end) -> dict:
    """Download via yfinance, returning {ticker: DataFrame[close, volume]}.

    Uses ``auto_adjust=True`` so 'Close' is already split/dividend adjusted -
    the correct series for return-based backtesting.
    """
    import yfinance as yf

    out: dict = {}
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    if raw is None or len(raw) == 0:
        return out

    single = len(tickers) == 1
    for t in tickers:
        try:
            if single:
                sub = raw
            else:
                sub = raw[t]
            cols = {"close": sub["Close"], "volume": sub["Volume"]}
            if "Open" in sub:
                cols["open"] = sub["Open"]
            df = pd.DataFrame(cols).dropna(how="all")
            df.index = pd.to_datetime(df.index)
            if not df.empty:
                out[t] = df.sort_index()
        except Exception:
            continue
    return out


# --------------------------------------------------------------------------- #
# Synthetic, offline, deterministic panel
# --------------------------------------------------------------------------- #
def _ticker_seed(ticker: str, base_seed: int) -> int:
    """Stable per-ticker seed so the same ticker always gets the same series."""
    h = hashlib.sha256(f"{ticker}:{base_seed}".encode()).hexdigest()
    return int(h[:8], 16)


def _synthetic_panel(tickers, start, end, seed) -> PriceData:
    """Generate a deterministic, cross-correlated price/volume panel.

    Model: a single common market factor drives co-movement (so cross-sectional
    momentum, pairs cointegration and factor regressions all have something real
    to find), plus per-ticker drift, market beta and idiosyncratic noise derived
    deterministically from the ticker name. Prices are 100 * cumprod(1 + r).
    """
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)
    if n < 5:
        raise ValueError("date range too short to synthesise")

    market_rng = np.random.default_rng(seed)
    # Market factor: ~10% annual drift, ~14% annual vol, with mild vol clustering
    # so the regime detector has high/low vol periods to find. Drift is additive
    # and must NOT be scaled by the vol state, or it would wash out to ~0.
    daily_drift = 0.10 / 252
    vol_state = 0.14 / np.sqrt(252) * (1 + 0.4 * np.sin(np.linspace(0, 6 * np.pi, n)))
    market_noise = market_rng.normal(0.0, 1.0, n) * vol_state
    market_noise -= market_noise.mean()   # demean so realised drift == daily_drift (seed-robust)
    market = daily_drift + market_noise

    close_cols, vol_cols = {}, {}
    for t in tickers:
        rng = np.random.default_rng(_ticker_seed(t, seed))
        beta = 0.6 + rng.random() * 0.9          # market beta in [0.6, 1.5]
        # Persistent idiosyncratic trend so there is real cross-sectional dispersion
        # for momentum to rank on (some names are genuine winners/losers).
        alpha = (rng.random() - 0.45) * 0.12 / 252
        idio_vol = (0.06 + rng.random() * 0.12) / np.sqrt(252)
        idio = rng.normal(0.0, idio_vol, n)
        idio -= idio.mean()              # demean so realised alpha == alpha (seed-robust)
        # Slow multi-month idiosyncratic trend so returns have *persistence* -
        # i.e. genuine momentum (and single-name reversion) for the signals to
        # capture. Pure GBM has none, which would make momentum untestable offline.
        period = 90 + rng.random() * 250
        phase = rng.random() * 2 * np.pi
        trend = (0.08 / 252) * np.sin(2 * np.pi * np.arange(n) / period + phase)
        ret = alpha + beta * market + idio + trend
        price = 100.0 * np.cumprod(1.0 + ret)
        close_cols[t] = price

        base_vol = 10 ** (6 + rng.random() * 1.5)  # 1M-30M shares/day
        vol_noise = rng.lognormal(0.0, 0.35, n)
        vol_cols[t] = base_vol * vol_noise

    # Inject genuine cointegration into known candidate pairs so the pairs-trading
    # demo finds something even offline. Leg B is rebuilt as a linear function of
    # leg A's log-price plus a *stationary* mean-reverting (Ornstein-Uhlenbeck)
    # residual: log(B) = c0 + c1*log(A) + u_t, with u_t stationary => A,B cointegrate.
    from .universe import PAIRS_CANDIDATES
    for a, b in PAIRS_CANDIDATES:
        if a in close_cols and b in close_cols:
            rng = np.random.default_rng(_ticker_seed(a + b, seed))
            log_a = np.log(close_cols[a])
            c0, c1 = rng.uniform(-0.5, 0.5), rng.uniform(0.8, 1.2)
            kappa = 0.05                       # OU mean-reversion speed (per day)
            resid_vol = 0.02
            u = np.zeros(n)
            for i in range(1, n):
                u[i] = (1 - kappa) * u[i - 1] + rng.normal(0.0, resid_vol)
            close_cols[b] = np.exp(c0 + c1 * log_a + u)

    # Open prices: yesterday's close moved by a small overnight gap, so the
    # next-open fill mode has a real open/close spread to price against (a zero
    # gap would make next-open identical to close-fill). Derived from the final
    # closes, after the cointegration override, so opens stay consistent.
    open_cols = {}
    for t in tickers:
        c = np.asarray(close_cols[t], dtype=float)
        rng = np.random.default_rng(_ticker_seed(t + "_open", seed))
        overnight = rng.normal(0.0, 0.003, n)      # ~0.3% overnight gap
        op = np.empty(n)
        op[0] = c[0]
        op[1:] = c[:-1] * (1.0 + overnight[1:])
        open_cols[t] = op

    close = pd.DataFrame(close_cols, index=dates)
    volume = pd.DataFrame(vol_cols, index=dates)
    open_ = pd.DataFrame(open_cols, index=dates)
    return PriceData(close=close, volume=volume, synthetic=True, open=open_)


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #
def _align(close, volume, tickers, opens=None) -> PriceData:
    """Drop tickers with no data, forward-fill small gaps, align the frames."""
    close = close.reindex(columns=[t for t in tickers if t in close.columns])
    close = close.dropna(axis=1, how="all")
    # Forward-fill at most a few days of holidays/missing prints, then drop any
    # leading rows that are still NaN. We never back-fill (that is look-ahead).
    close = close.ffill(limit=3)
    volume = volume.reindex(index=close.index, columns=close.columns).ffill(limit=3)
    close = close.dropna(how="all")
    volume = volume.reindex(index=close.index)

    # Open prices only when every surviving ticker carries one; a partial open
    # panel would silently disable next-open fills for some names.
    open_df = None
    if opens and all(t in opens for t in close.columns):
        open_df = pd.DataFrame({t: opens[t] for t in close.columns})
        open_df = open_df.reindex(index=close.index, columns=close.columns).ffill(limit=3)
    return PriceData(close=close, volume=volume, synthetic=False, open=open_df)
