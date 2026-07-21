"""Implied volatility surface in (log-forward-moneyness, expiry), interpolated
in total variance.

Two coordinate choices do most of the work here.

**Log-forward-moneyness** ``k = ln(K / F)``, where ``F = S*exp((r-q)T)`` is the
forward. Using the forward rather than spot centres each expiry's smile on
zero, so slices at different maturities are directly comparable and the skew
does not drift sideways purely from carry.

**Total variance** ``w = sigma^2 * T`` rather than implied vol. This is the
natural coordinate because the no-calendar-arbitrage condition is exactly
"``w`` is non-decreasing in ``T``". Interpolating linearly in ``w`` between two
arbitrage-free expiry slices preserves that property, whereas interpolating in
implied vol does not: a flat vol term structure is a rising total variance, and
naive vol interpolation across expiries can manufacture calendar arbitrage out
of perfectly clean inputs.

The surface is built from mid prices by solving each quote for its own implied
vol with the Phase 1 solver, which reports why it failed rather than returning a
number it does not believe. Failures are counted by reason and reported.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ..config import OptionsConfig
from .arbitrage import ArbitrageReport, check_surface
from .chain import OptionChain
from .implied_vol import REASON_OK, implied_volatility_detailed
from .pricing import CALL, PUT, black_scholes_price

logger = logging.getLogger(__name__)

ArrayLike = Union[float, np.ndarray]

POINT_COLUMNS = [
    "expiry", "time_to_expiry", "strike", "option_type", "forward",
    "log_moneyness", "mid", "spread", "implied_vol", "total_variance", "call_price",
]


@dataclass
class VolSurface:
    """Fitted implied-vol surface plus the diagnostics from building it.

    Attributes
    ----------
    points : pd.DataFrame
        One row per successfully inverted quote, columns as ``POINT_COLUMNS``.
    arbitrage : ArbitrageReport
        Static no-arbitrage findings. Not repaired, just reported.
    iv_failures : dict
        Count of quotes that produced no implied vol, keyed by reason code.
    synthetic : bool
        Propagated from the chain. True means these are not market prices.
    """

    points: pd.DataFrame
    spot: float
    as_of: pd.Timestamp
    ticker: str
    rate: float
    dividend_yield: float
    arbitrage: ArbitrageReport
    iv_failures: Dict[str, int] = field(default_factory=dict)
    synthetic: bool = False

    @property
    def expiries(self) -> List[pd.Timestamp]:
        return sorted(self.points["expiry"].unique())

    @property
    def maturities(self) -> np.ndarray:
        """Sorted unique times to expiry, in years."""
        return np.array(sorted(self.points["time_to_expiry"].unique()), dtype=float)

    def slice(self, expiry) -> pd.DataFrame:
        """One expiry's smile, sorted by log-moneyness."""
        sel = self.points[self.points["expiry"] == pd.Timestamp(expiry)]
        return sel.sort_values("log_moneyness").reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Interpolation
    # ------------------------------------------------------------------ #
    def total_variance(self, log_moneyness: ArrayLike,
                       time_to_expiry: ArrayLike) -> ArrayLike:
        """Interpolated total variance at (k, T).

        Linear in ``k`` within each expiry slice, then linear in ``T`` between
        the two bracketing slices. Outside the observed range in either
        coordinate the nearest edge value is held flat: extrapolating a smile is
        how people invent wings that do not exist.
        """
        k_arr = np.atleast_1d(np.asarray(log_moneyness, dtype=float))
        t_arr = np.atleast_1d(np.asarray(time_to_expiry, dtype=float))
        k_arr, t_arr = np.broadcast_arrays(k_arr, t_arr)
        scalar = np.ndim(log_moneyness) == 0 and np.ndim(time_to_expiry) == 0

        mats = self.maturities
        if mats.size == 0:
            out = np.full(k_arr.shape, np.nan)
            return float("nan") if scalar else out

        # Total variance along k for each observed maturity.
        slices = {}
        for t in mats:
            sel = self.points[self.points["time_to_expiry"] == t].sort_values("log_moneyness")
            slices[t] = (sel["log_moneyness"].to_numpy(dtype=float),
                         sel["total_variance"].to_numpy(dtype=float))

        def w_at(t_obs: float, k: np.ndarray) -> np.ndarray:
            ks, ws = slices[t_obs]
            if ks.size == 0:
                return np.full(k.shape, np.nan)
            if ks.size == 1:
                return np.full(k.shape, ws[0])
            return np.interp(k, ks, ws)          # np.interp clamps at the edges

        t_clipped = np.clip(t_arr, mats[0], mats[-1])
        out = np.empty(k_arr.shape, dtype=float)

        for idx in np.ndindex(k_arr.shape):
            k = np.atleast_1d(k_arr[idx])
            t = float(t_clipped[idx])
            hi = int(np.searchsorted(mats, t, side="left"))
            if hi <= 0:
                out[idx] = w_at(mats[0], k)[0]
            elif hi >= mats.size:
                out[idx] = w_at(mats[-1], k)[0]
            elif np.isclose(mats[hi], t):
                out[idx] = w_at(mats[hi], k)[0]
            else:
                t_lo, t_hi = mats[hi - 1], mats[hi]
                w_lo = w_at(t_lo, k)[0]
                w_hi = w_at(t_hi, k)[0]
                frac = (t - t_lo) / (t_hi - t_lo)
                out[idx] = w_lo + frac * (w_hi - w_lo)

        # atleast_1d gave us a shape-(1,) array for scalar input, so unwrap by
        # element rather than by float(), which only accepts 0-d.
        return float(out.reshape(-1)[0]) if scalar else out

    def implied_vol(self, log_moneyness: ArrayLike,
                    time_to_expiry: ArrayLike) -> ArrayLike:
        """Interpolated implied volatility at (k, T), via total variance."""
        w = self.total_variance(log_moneyness, time_to_expiry)
        t = np.asarray(time_to_expiry, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            vol = np.sqrt(np.maximum(np.asarray(w, dtype=float), 0.0) / t)
        vol = np.where(t > 0, vol, np.nan)
        return float(vol) if np.ndim(w) == 0 else vol

    def implied_vol_for_strike(self, strike: ArrayLike,
                               time_to_expiry: ArrayLike) -> ArrayLike:
        """Implied vol quoted by strike instead of moneyness."""
        t = np.asarray(time_to_expiry, dtype=float)
        forward = self.spot * np.exp((self.rate - self.dividend_yield) * t)
        return self.implied_vol(np.log(np.asarray(strike, dtype=float) / forward), t)

    def atm_term_structure(self) -> pd.DataFrame:
        """At-the-money-forward implied vol by maturity (k = 0)."""
        mats = self.maturities
        return pd.DataFrame({
            "time_to_expiry": mats,
            "atm_vol": [self.implied_vol(0.0, float(t)) for t in mats],
        })

    def summary(self) -> List[str]:
        label = "SYNTHETIC" if self.synthetic else "live"
        lines = [
            f"VOL SURFACE {self.ticker} ({label}) as of {self.as_of:%Y-%m-%d}",
            f" spot {self.spot:.2f}, rate {self.rate:.2%}, "
            f"div yield {self.dividend_yield:.2%}",
            f" {len(self.points)} points across {len(self.expiries)} expiries",
        ]
        atm = self.atm_term_structure()
        if not atm.empty:
            shown = ", ".join(f"{row.time_to_expiry * 365:.0f}d={row.atm_vol:.1%}"
                              for row in atm.itertuples())
            lines.append(f" ATM term structure: {shown}")
        if self.iv_failures:
            detail = ", ".join(f"{k}={v}" for k, v in sorted(self.iv_failures.items()))
            lines.append(f" implied-vol failures: {detail}")
        lines.extend(" " + ln for ln in self.arbitrage.summary())
        return lines


def build_surface(
    chain: OptionChain,
    rate: float = 0.04,
    dividend_yield: float = 0.0,
    cfg: Optional[OptionsConfig] = None,
    use_otm_only: bool = True,
) -> VolSurface:
    """Invert a chain into an implied-vol surface and check it for arbitrage.

    Parameters
    ----------
    use_otm_only : bool
        Keep only the out-of-the-money side of each strike (puts below the
        forward, calls above). This is what desks quote from: OTM options are
        all time value, so their implied vols are better conditioned, and it
        avoids double-counting a strike that has both a call and a put quote
        carrying the same information by put-call parity.

    Returns
    -------
    VolSurface
        Points that could be inverted, plus failure counts and the arbitrage
        report. Raises if nothing at all could be inverted, since an empty
        surface silently pricing at NaN is worse than a loud failure.
    """
    cfg = cfg or OptionsConfig()
    quotes = chain.quotes.copy()
    if quotes.empty:
        raise RuntimeError("cannot build a surface from an empty chain")

    tte = quotes["time_to_expiry"].to_numpy(dtype=float)
    forward = chain.spot * np.exp((rate - dividend_yield) * tte)
    quotes["forward"] = forward
    quotes["log_moneyness"] = np.log(quotes["strike"].to_numpy(dtype=float) / forward)

    if use_otm_only:
        is_otm = ((quotes["option_type"] == CALL) & (quotes["strike"] >= quotes["forward"])) | \
                 ((quotes["option_type"] == PUT) & (quotes["strike"] < quotes["forward"]))
        quotes = quotes[is_otm]

    records = []
    failures: Dict[str, int] = {}

    for row in quotes.itertuples():
        res = implied_volatility_detailed(
            price=float(row.mid), spot=chain.spot, strike=float(row.strike),
            time_to_expiry=float(row.time_to_expiry), rate=rate,
            option_type=row.option_type, dividend_yield=dividend_yield,
            vol_lower=cfg.iv_lower, vol_upper=cfg.iv_upper, tolerance=cfg.iv_tolerance)

        if not res.ok:
            failures[res.reason] = failures.get(res.reason, 0) + 1
            continue

        # Price the equivalent CALL at this vol, so the butterfly check has a
        # single consistent price curve in strike regardless of which side of
        # the forward the quote came from.
        call_price = float(black_scholes_price(
            chain.spot, float(row.strike), float(row.time_to_expiry), rate,
            res.vol, CALL, dividend_yield))

        records.append({
            "expiry": row.expiry,
            "time_to_expiry": float(row.time_to_expiry),
            "strike": float(row.strike),
            "option_type": row.option_type,
            "forward": float(row.forward),
            "log_moneyness": float(row.log_moneyness),
            "mid": float(row.mid),
            "spread": float(row.ask - row.bid),
            "implied_vol": float(res.vol),
            "total_variance": float(res.vol ** 2 * row.time_to_expiry),
            "call_price": call_price,
        })

    if not records:
        raise RuntimeError(
            f"no quote in the {chain.ticker} chain could be inverted to an "
            f"implied vol; failure reasons: {failures}")

    points = pd.DataFrame(records)[POINT_COLUMNS]

    # Drop expiries too thin to describe a smile, and say so.
    counts = points.groupby("expiry").size()
    thin = counts[counts < cfg.min_points_per_slice].index
    if len(thin):
        logger.info("dropping %d expiry slice(s) with fewer than %d points",
                    len(thin), cfg.min_points_per_slice)
        points = points[~points["expiry"].isin(thin)]
    if points.empty:
        raise RuntimeError("no expiry had enough points to form a smile")

    report = check_surface(points, cfg.butterfly_tolerance, cfg.calendar_tolerance)

    surface = VolSurface(
        points=points.sort_values(["time_to_expiry", "log_moneyness"]).reset_index(drop=True),
        spot=chain.spot, as_of=chain.as_of, ticker=chain.ticker,
        rate=rate, dividend_yield=dividend_yield,
        arbitrage=report, iv_failures=failures, synthetic=chain.synthetic)

    logger.info("built %s surface: %d points, %d expiries, %d arbitrage violation(s)",
                chain.ticker, len(points), len(surface.expiries),
                len(report.violations))
    return surface


def plot_surface(surface: VolSurface, path: str, title: Optional[str] = None) -> str:
    """Smile per expiry plus the ATM term structure. Returns the saved path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    cmap = plt.get_cmap("viridis")
    expiries = surface.expiries

    for i, exp in enumerate(expiries):
        sl = surface.slice(exp)
        colour = cmap(i / max(len(expiries) - 1, 1))
        days = sl["time_to_expiry"].iloc[0] * 365
        ax1.plot(sl["log_moneyness"], sl["implied_vol"] * 100, "o-", ms=3,
                 color=colour, lw=1.2, label=f"{days:.0f}d")
    ax1.axvline(0.0, color="grey", lw=0.7, ls="--")
    ax1.set_xlabel("log forward moneyness  ln(K/F)")
    ax1.set_ylabel("implied vol (%)")
    ax1.set_title("Smile by expiry")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=7, ncol=2)

    atm = surface.atm_term_structure()
    ax2.plot(atm["time_to_expiry"] * 365, atm["atm_vol"] * 100, "o-",
             color="#1f77b4", lw=1.4)
    ax2.set_xlabel("days to expiry")
    ax2.set_ylabel("ATM implied vol (%)")
    ax2.set_title("ATM term structure")
    ax2.grid(True, alpha=0.3)

    label = "SYNTHETIC" if surface.synthetic else "live"
    fig.suptitle(title or f"{surface.ticker} vol surface ({label}), "
                          f"{surface.as_of:%Y-%m-%d}")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path
