"""Quote-level diagnostics for option-chain data quality."""

from __future__ import annotations

import numpy as np
import pandas as pd


def put_call_parity_residuals(chain, rate: float = 0.0, dividend_yield: float = 0.0) -> pd.DataFrame:
    """Return put-call parity residuals for call/put pairs in an option chain.

    ``C - P = S exp(-qT) - K exp(-rT)`` for European options. A residual is
    compared with the two quoted spreads, because a mismatch smaller than the
    spreads is not an executable arbitrage.
    """
    q = chain.quotes.copy()
    calls = q[q["option_type"] == "call"].set_index(["expiry", "strike"])
    puts = q[q["option_type"] == "put"].set_index(["expiry", "strike"])
    joined = calls[["mid", "bid", "ask", "time_to_expiry"]].join(
        puts[["mid", "bid", "ask"]], how="inner", lsuffix="_call", rsuffix="_put"
    ).reset_index()
    if joined.empty:
        return pd.DataFrame(columns=["expiry", "strike", "residual", "combined_spread", "exceeds_spread"])
    t = joined["time_to_expiry"].to_numpy(float)
    rhs = chain.spot * np.exp(-dividend_yield * t) - joined["strike"].to_numpy(float) * np.exp(-rate * t)
    residual = joined["mid_call"].to_numpy(float) - joined["mid_put"].to_numpy(float) - rhs
    spread = ((joined["ask_call"] - joined["bid_call"]) +
              (joined["ask_put"] - joined["bid_put"])).to_numpy(float)
    out = joined[["expiry", "strike"]].copy()
    out["residual"] = residual
    out["combined_spread"] = spread
    out["exceeds_spread"] = np.abs(residual) > spread
    return out


def liquidity_profile(chain) -> dict:
    """Summarise quoted spread and activity without treating zero as liquidity."""
    q = chain.quotes.copy()
    spread = q["ask"] - q["bid"]
    mid = q["mid"].replace(0.0, np.nan)
    relative = spread / mid
    volume = q["volume"].fillna(0.0)
    return {
        "n_quotes": int(len(q)),
        "median_spread": float(spread.median()),
        "median_relative_spread": float(relative.median()),
        "share_with_volume": float((volume > 0).mean()),
        "share_with_open_interest": float((q["open_interest"].fillna(0.0) > 0).mean()),
    }
