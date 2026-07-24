"""Historical option-chain files (OptionsDX end-of-day format).

yfinance serves only the current chain, so anything that wants a *past* chain
needs a file. OptionsDX publishes free end-of-day samples in a wide CSV: one row
per (snapshot, expiry, strike) with the call and the put side written out
next to each other. This module turns that into the repo's own long quote schema
so the surface builder and everything downstream treat a historical snapshot
exactly like a live one.

What the format looks like, and the choices it forces:

  * A file can hold several intraday snapshots of the same day (the sample used
    here is 2020-03-06, the COVID crash Friday, with 31 snapshots from 09:30 on).
    Each snapshot is a full chain, so they are kept separate and addressed by
    time. The last snapshot of a day is the end-of-day stand-in.
  * Missing fields are empty strings, which become NaN. Deep in-the-money and
    zero-day rows routinely have a blank vendor IV.
  * There is no open-interest column. It is filled with NaN rather than a made-up
    number, and the docstring says so; the hygiene filters do not depend on it
    unless a positive minimum is configured.
  * Time to expiry comes from the vendor DTE column, which is fractional and
    includes the intraday portion (0.27 on an expiry-day 09:30 quote). We check
    it against EXPIRE_DATE minus QUOTE_DATE and warn if they disagree by more
    than a day, but DTE is the finer measure and is the one used.

Vendor-supplied Greeks and implied vols are carried through unchanged, but as
data to validate against, never as truth. See ``scripts/validate_optionsdx.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import OptionsConfig
from .chain import DAYS_PER_YEAR, QUOTE_COLUMNS, OptionChain, _clean_quotes
from .pricing import CALL, PUT

logger = logging.getLogger(__name__)

# Columns we read from the wide file. Anything else is ignored.
_REQUIRED = ["QUOTE_DATE", "QUOTE_TIME_HOURS", "UNDERLYING_LAST", "EXPIRE_DATE",
             "DTE", "STRIKE", "C_BID", "C_ASK", "P_BID", "P_ASK"]
_NUMERIC = ["QUOTE_TIME_HOURS", "UNDERLYING_LAST", "DTE", "STRIKE",
            "C_BID", "C_ASK", "C_IV", "C_DELTA", "C_VOLUME",
            "P_BID", "P_ASK", "P_IV", "P_DELTA", "P_VOLUME"]

# The long-format columns this loader adds on top of QUOTE_COLUMNS, kept so a
# validation pass can compare our numbers against the vendor's.
VENDOR_COLUMNS = ["snapshot", "underlying_last", "vendor_iv", "vendor_delta"]


@dataclass
class OptionsDayData:
    """One day of option chains parsed from an OptionsDX file.

    Attributes
    ----------
    long : pd.DataFrame
        Every quote across every snapshot in long format: one row per contract
        side, with QUOTE_COLUMNS plus the VENDOR_COLUMNS. Not yet hygiene
        filtered, so a validation pass can see what the vendor priced even on
        quotes we would discard.
    date : pd.Timestamp
        The trading day.
    ticker : str
    dropped_total : dict
        Hygiene drops aggregated across snapshots (populated lazily by chains()).
    """

    long: pd.DataFrame
    date: pd.Timestamp
    ticker: str
    dropped_total: Dict[str, int] = field(default_factory=dict)

    @property
    def snapshots(self) -> List[float]:
        """Snapshot times as QUOTE_TIME_HOURS values, sorted."""
        return sorted(self.long["snapshot"].unique())

    def _as_of(self, snapshot: float) -> pd.Timestamp:
        return self.date + pd.Timedelta(hours=float(snapshot))

    def vendor_at(self, snapshot: float) -> pd.DataFrame:
        """The raw long frame for one snapshot, vendor columns intact."""
        return self.long[self.long["snapshot"] == snapshot].reset_index(drop=True)

    def chain_at(self, snapshot: float, cfg: Optional[OptionsConfig] = None) -> OptionChain:
        """Cleaned OptionChain for one snapshot, filtered by the shared hygiene rules."""
        cfg = cfg or OptionsConfig()
        sub = self.vendor_at(snapshot)
        raw = sub[["expiry", "time_to_expiry", "strike", "option_type",
                   "bid", "ask", "volume", "open_interest"]].copy()
        clean, dropped = _clean_quotes(raw, cfg)
        spot = float(sub["underlying_last"].iloc[0])
        return OptionChain(quotes=clean[QUOTE_COLUMNS], spot=spot,
                           as_of=self._as_of(snapshot), ticker=self.ticker,
                           synthetic=False, dropped=dropped)

    def last_chain(self, cfg: Optional[OptionsConfig] = None) -> OptionChain:
        """End-of-day stand-in: the latest snapshot in the file."""
        return self.chain_at(self.snapshots[-1], cfg)

    def summary(self) -> List[str]:
        snaps = self.snapshots
        return [
            f"OPTIONSDX {self.ticker} {self.date:%Y-%m-%d} "
            f"({len(snaps)} snapshot(s), {len(self.long)} contract-quotes)",
            f" snapshots {min(snaps):.2f}h to {max(snaps):.2f}h; "
            f"spot {self.long['underlying_last'].iloc[0]:.2f} to "
            f"{self.long['underlying_last'].iloc[-1]:.2f}",
        ]


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """OptionsDX headers sometimes carry brackets/spaces; normalise to bare names."""
    df = df.copy()
    df.columns = [str(c).strip().strip("[]").strip().upper() for c in df.columns]
    return df


def load_optionsdx_csv(path: str, ticker: str = "SPY") -> OptionsDayData:
    """Parse an OptionsDX end-of-day CSV into an :class:`OptionsDayData`.

    Parameters
    ----------
    path : str
        Path to the CSV. The file is read whole; the samples are tens of MB.
    ticker : str
        Underlying symbol. OptionsDX single-name files do not repeat it per row.

    Raises
    ------
    RuntimeError
        If the file is missing required columns or holds more than one trading
        date (this loader is per-day on purpose; concatenating days is the
        caller's job).
    """
    wide = _normalise_columns(pd.read_csv(path))
    missing = [c for c in _REQUIRED if c not in wide.columns]
    if missing:
        raise RuntimeError(f"OptionsDX file {path} missing columns: {missing}")

    for col in _NUMERIC:
        if col in wide.columns:
            wide[col] = pd.to_numeric(wide[col], errors="coerce")

    quote_dates = pd.to_datetime(wide["QUOTE_DATE"]).dt.normalize().unique()
    if len(quote_dates) != 1:
        raise RuntimeError(
            f"expected one trading date, found {len(quote_dates)}: "
            f"{[pd.Timestamp(d).date() for d in quote_dates[:5]]}")
    date = pd.Timestamp(quote_dates[0])

    expiry = pd.to_datetime(wide["EXPIRE_DATE"]).dt.normalize()
    # DTE is the vendor's fractional day count (intraday included). Cross-check
    # against the plain calendar difference and warn on real disagreement.
    calendar_days = (expiry - date).dt.days
    disagree = (wide["DTE"] - calendar_days).abs() > 1.5
    if disagree.any():
        logger.warning("DTE and (EXPIRE_DATE - QUOTE_DATE) disagree on %d row(s); "
                       "using DTE", int(disagree.sum()))
    time_to_expiry = wide["DTE"] / DAYS_PER_YEAR

    shared = pd.DataFrame({
        "snapshot": wide["QUOTE_TIME_HOURS"],
        "underlying_last": wide["UNDERLYING_LAST"],
        "expiry": expiry,
        "time_to_expiry": time_to_expiry,
        "strike": wide["STRIKE"],
    })

    def _side(prefix: str, option_type: str) -> pd.DataFrame:
        side = shared.copy()
        side["option_type"] = option_type
        side["bid"] = wide[f"{prefix}_BID"]
        side["ask"] = wide[f"{prefix}_ASK"]
        side["volume"] = wide.get(f"{prefix}_VOLUME", np.nan)
        side["open_interest"] = np.nan            # not provided by OptionsDX
        side["vendor_iv"] = wide.get(f"{prefix}_IV", np.nan)
        side["vendor_delta"] = wide.get(f"{prefix}_DELTA", np.nan)
        return side

    long = pd.concat([_side("C", CALL), _side("P", PUT)], ignore_index=True)
    long = long.sort_values(["snapshot", "expiry", "strike", "option_type"]
                            ).reset_index(drop=True)

    logger.info("loaded OptionsDX %s %s: %d contract-quotes across %d snapshot(s)",
                ticker, date.date(), len(long), long["snapshot"].nunique())
    return OptionsDayData(long=long, date=date, ticker=ticker)
