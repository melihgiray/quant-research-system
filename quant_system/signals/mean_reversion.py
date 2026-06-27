"""Mean-reversion signals: statistical pairs trading and single-asset z-score.

Pairs trading (the classic stat-arb): find two assets whose *prices* are
cointegrated — individually they wander (non-stationary), but a linear
combination (the spread) is stationary and mean-reverting. We:

  1. Select the most cointegrated candidate pair via the Engle-Granger test
     (statsmodels.coint).
  2. Estimate a rolling hedge ratio so the spread is constructed from past data
     only (no look-ahead in the spread itself).
  3. Trade the spread's z-score: enter when |z| > 2 (the spread is stretched),
     exit when it reverts through 0.
  4. Re-test cointegration on a rolling window and STOP trading the pair if the
     relationship breaks down (p-value > 0.10) — a non-stationary "spread" is
     just two correlated random walks and will bankrupt you.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from ..config import PairsConfig


def find_cointegrated_pair(
    close: pd.DataFrame,
    candidates: Sequence[Tuple[str, str]],
    max_pvalue: float = 0.10,
) -> Optional[Tuple[str, str, float]]:
    """Return the candidate pair with the lowest Engle-Granger p-value.

    Parameters
    ----------
    close : pd.DataFrame
        Close prices containing the candidate tickers.
    candidates : list[(str, str)]
        Economically motivated pairs to test (don't data-mine all pairs — that
        guarantees spurious cointegration by multiple comparisons).
    max_pvalue : float
        Reject the best pair if even it exceeds this p-value.

    Returns
    -------
    (a, b, pvalue) or None
        Best cointegrated pair, or None if none qualify / data missing.
    """
    best = None
    for a, b in candidates:
        if a not in close.columns or b not in close.columns:
            continue
        s = np.log(close[[a, b]].dropna())
        if len(s) < 252:
            continue
        try:
            _, pvalue, _ = coint(s[a], s[b])
        except Exception:
            continue
        if best is None or pvalue < best[2]:
            best = (a, b, float(pvalue))
    if best is None or best[2] > max_pvalue:
        return None
    return best


def _rolling_hedge_ratio(log_a: pd.Series, log_b: pd.Series, window: int) -> pd.Series:
    """Rolling OLS hedge ratio beta = Cov(a,b)/Var(b), lagged one day.

    Estimating beta from a trailing window (and lagging it) keeps the spread
    construction free of look-ahead.
    """
    cov = log_a.rolling(window).cov(log_b)
    var = log_b.rolling(window).var()
    return (cov / var).shift(1)


def _rolling_coint_pvalue(log_a: pd.Series, log_b: pd.Series,
                          window: int, cadence: int = 21) -> pd.Series:
    """Engle-Granger p-value re-estimated every `cadence` days on a trailing window.

    Forward-filled between recomputations (cheap, and the relationship does not
    change daily). Used as a kill-switch when cointegration breaks down.
    """
    idx = log_a.index
    pvals = pd.Series(np.nan, index=idx)
    for pos in range(window, len(idx), cadence):
        sl = slice(pos - window, pos)
        a_win, b_win = log_a.iloc[sl], log_b.iloc[sl]
        try:
            _, pv, _ = coint(a_win, b_win)
        except Exception:
            pv = 1.0
        pvals.iloc[pos] = pv
    return pvals.ffill()


def pairs_signal(price_data, pair: Tuple[str, str],
                 cfg: PairsConfig = None) -> pd.DataFrame:
    """Build daily target weights for one cointegrated pair.

    The traded book is hedge-ratio weighted and gross-normalised to 1:
        long-spread  -> long A, short B (in proportion to beta)
        short-spread -> short A, long B
    Position opens at |z| > entry_z and closes when z reverts through exit_z.
    Trading halts whenever the rolling cointegration p-value exceeds the limit.

    Parameters
    ----------
    price_data : PriceData
        Panel containing the two tickers.
    pair : (str, str)
        The (A, B) tickers to trade.
    cfg : PairsConfig
        z-score / entry / exit / cointegration parameters.
    """
    cfg = cfg or PairsConfig()
    a, b = pair
    close = price_data.close
    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)

    log_a, log_b = np.log(close[a]), np.log(close[b])
    beta = _rolling_hedge_ratio(log_a, log_b, cfg.coint_lookback)
    spread = log_a - beta * log_b
    mu = spread.rolling(cfg.zscore_lookback).mean()
    sd = spread.rolling(cfg.zscore_lookback).std()
    z = (spread - mu) / sd
    pval = _rolling_coint_pvalue(log_a, log_b, cfg.coint_lookback)

    # Stateful entry/exit with a cointegration kill-switch.
    pos = 0
    positions = np.zeros(len(close.index))
    z_vals, beta_vals, pv_vals = z.values, beta.values, pval.values
    for i in range(len(close.index)):
        zi, bi, pvi = z_vals[i], beta_vals[i], pv_vals[i]
        broken = (not np.isfinite(pvi)) or (pvi > cfg.coint_pvalue_max) or (not np.isfinite(bi)) or (bi <= 0)
        if broken or not np.isfinite(zi):
            pos = 0
        elif pos == 0:
            if zi > cfg.entry_z:
                pos = -1                      # spread rich -> short spread
            elif zi < -cfg.entry_z:
                pos = +1                      # spread cheap -> long spread
        else:                                  # manage open position
            if pos == 1 and zi >= -cfg.exit_z:
                pos = 0
            elif pos == -1 and zi <= cfg.exit_z:
                pos = 0
        positions[i] = pos

        if pos != 0 and np.isfinite(bi) and bi > 0:
            denom = 1.0 + bi
            weights.iat[i, weights.columns.get_loc(a)] = pos * 1.0 / denom
            weights.iat[i, weights.columns.get_loc(b)] = -pos * bi / denom

    return weights


def single_asset_reversion(price_data, lookback: int = 21,
                           entry_z: float = 1.0, max_weight: float = 0.10) -> pd.DataFrame:
    """Single-name mean reversion: fade short-term z-score extremes.

    weight_i ∝ -zscore_i (buy what fell, sell what rose), clipped per name and
    gross-normalised. This is the counterpart to momentum — it profits when
    prices over-react and snap back, which is why it tends to do well precisely
    when momentum does badly (the two are natural diversifiers).
    """
    close = price_data.close
    mean = close.rolling(lookback).mean()
    std = close.rolling(lookback).std()
    z = (close - mean) / std
    raw = (-z).clip(lower=-max_weight, upper=max_weight)
    gross = raw.abs().sum(axis=1).replace(0.0, np.nan)
    scale = (1.0 / gross).clip(upper=1.0)
    return raw.mul(scale, axis=0).fillna(0.0)
