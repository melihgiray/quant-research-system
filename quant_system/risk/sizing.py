"""Position sizing: fixed-fractional, volatility targeting, and (fractional) Kelly.

All functions take weights/returns and return *scaled weights*. None of them peek
at the future: vol targeting uses a trailing realised-vol estimate that the caller
is responsible for lagging via the engine's shift, and the functions here only
ever read past data passed to them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import TRADING_DAYS_PER_YEAR


def fixed_fractional(signal: pd.DataFrame, fraction: float = 1.0) -> pd.DataFrame:
    """Scale a unit signal to a fixed fraction of capital.

    The simplest sizing rule: bet a constant fraction. Robust, no estimation
    error, and the baseline every fancier scheme must beat.
    """
    return signal * fraction


def vol_target_scale(
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    target_vol: float = 0.10,
    lookback: int = TRADING_DAYS_PER_YEAR,
    max_leverage: float = 3.0,
) -> pd.DataFrame:
    """Scale the whole book each day so its trailing portfolio vol ~ target_vol.

    Estimate the portfolio's realised annualised volatility over a trailing
    window using the *current* weights, then multiply by target/realised. The
    scaler is lagged one day (we size today using vol known through yesterday),
    and capped at ``max_leverage`` to avoid blowing up when realised vol is tiny.

    Parameters
    ----------
    weights : pd.DataFrame
        Pre-sizing target weights (date x ticker).
    asset_returns : pd.DataFrame
        Simple asset returns (date x ticker).
    target_vol : float
        Desired annualised portfolio volatility.
    lookback : int
        Trailing window for the realised-vol estimate.
    max_leverage : float
        Cap on the gross scaling multiplier.
    """
    aligned = asset_returns.reindex_like(weights)
    port_ret = (weights * aligned).sum(axis=1)
    realised = port_ret.rolling(lookback, min_periods=lookback // 4).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    scaler = (target_vol / realised).replace([np.inf, -np.inf], np.nan)
    scaler = scaler.clip(upper=max_leverage).shift(1)  # size today on yesterday's vol
    return weights.mul(scaler, axis=0)


def kelly_weights(
    expected_returns: pd.Series,
    cov: pd.DataFrame,
    kelly_fraction: float = 0.5,
    max_weight: float = 0.10,
) -> pd.Series:
    """Fractional-Kelly weights from expected returns and a covariance matrix.

    Full Kelly is w* = Σ⁻¹ μ. It maximises long-run log-growth but is famously
    too aggressive (it assumes μ and Σ are known exactly), so we scale by
    ``kelly_fraction`` (half-Kelly is the common practitioner choice) and clip
    per-name weights. A pseudo-inverse is used so a singular Σ degrades
    gracefully rather than throwing.

    Parameters
    ----------
    expected_returns : pd.Series
        Per-asset expected (excess) return, indexed by ticker.
    cov : pd.DataFrame
        Asset return covariance matrix (same tickers).
    kelly_fraction : float
        Fraction of full Kelly to apply (0.5 = half-Kelly).
    max_weight : float
        Per-name absolute cap.
    """
    tickers = expected_returns.index
    mu = expected_returns.values.astype(float)
    sigma = cov.reindex(index=tickers, columns=tickers).values.astype(float)
    w_full = np.linalg.pinv(sigma) @ mu
    w = kelly_fraction * w_full
    w = np.clip(w, -max_weight, max_weight)
    return pd.Series(w, index=tickers, name="kelly_weight")
