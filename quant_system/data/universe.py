"""Asset universes.

The lists are hard-coded on purpose. Scraping the live S&P 100 introduces
survivorship and point-in-time membership problems that are *worse* than using a
fixed, transparent list for a research sandbox. A fixed list keeps the history
reproducible instead of shifting under you when a scraper breaks or the index
gets re-balanced.

All tickers are large, liquid US ETFs / single names where yfinance history is
reliable and the square-root cost model's liquidity assumptions hold.
"""

from __future__ import annotations

from typing import Dict, List


# The 11 SPDR sector ETFs - a clean cross-section of the US equity market.
# Cross-sectional momentum across sectors is a classic, well-behaved test bed.
SECTOR_ETFS: List[str] = [
    "XLB",  # Materials
    "XLE",  # Energy
    "XLF",  # Financials
    "XLI",  # Industrials
    "XLK",  # Technology
    "XLP",  # Consumer Staples
    "XLU",  # Utilities
    "XLV",  # Health Care
    "XLY",  # Consumer Discretionary
    "XLRE", # Real Estate
    "XLC",  # Communication Services
]

# A liquid large-cap single-name subset (an S&P 100 slice). Used by the ML signal
# and as an alternative cross-sectional universe.
LIQUID_LARGE_CAPS: List[str] = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B",
    "JPM", "V", "JNJ", "WMT", "PG", "MA", "HD", "CVX",
    "XOM", "KO", "PEP", "BAC", "MRK", "ABBV", "COST", "MCD",
]

# Candidate pairs with a plausible economic link, so any cointegration we find
# has a story behind it (not a data-mined coincidence).
#   GLD/SLV : gold vs silver (precious metals).
#   XLE/XOP : energy sector vs E&P sub-sector.
#   KO/PEP  : the classic consumer-staples duopoly.
#   EWA/EWC : Australia vs Canada - both commodity-driven economies.
PAIRS_CANDIDATES: List[tuple] = [
    ("GLD", "SLV"),
    ("XLE", "XOP"),
    ("KO", "PEP"),
    ("EWA", "EWC"),
]

# ETFs used to build Fama-French style factor proxies in performance/factor_decomp.
#   SPY        : the market.
#   IWM        : small caps  -> SMB proxy = IWM - SPY.
#   VTV / VUG  : value / growth -> HML proxy = VTV - VUG.
FACTOR_ETFS: Dict[str, str] = {
    "market": "SPY",
    "small": "IWM",
    "value": "VTV",
    "growth": "VUG",
}


def universe(name: str) -> List[str]:
    """Return the ticker list for a named universe.

    Parameters
    ----------
    name : str
        One of ``"sectors"``, ``"largecaps"``, or ``"all"``.

    Returns
    -------
    list[str]
        De-duplicated list of tickers, order preserved.
    """
    name = name.lower()
    if name == "sectors":
        tickers = list(SECTOR_ETFS)
    elif name in ("largecaps", "large_caps", "stocks"):
        tickers = list(LIQUID_LARGE_CAPS)
    elif name == "all":
        tickers = list(SECTOR_ETFS) + list(LIQUID_LARGE_CAPS)
    else:
        raise ValueError(
            f"unknown universe {name!r}; choose 'sectors', 'largecaps', or 'all'"
        )
    # De-duplicate while preserving order.
    seen: set = set()
    out: List[str] = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def all_tickers_for(name: str) -> List[str]:
    """Universe tickers plus the factor ETFs needed for decomposition."""
    return universe(name) + list(dict.fromkeys(FACTOR_ETFS.values()))
