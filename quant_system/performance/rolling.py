"""Rolling performance series for the HTML report.

A single headline Sharpe hides how a strategy actually behaved through time: a
2.0 that came entirely from one good year is not the same bet as a steady 0.8.
These rolling views, plus the per-year table, are what turn a single number into
an honest picture.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import TRADING_DAYS_PER_YEAR
from .analytics import annualized_vol, max_drawdown, sharpe_ratio


def rolling_sharpe(returns: pd.Series,
                   window: int = 126,
                   periods: int = TRADING_DAYS_PER_YEAR,
                   rf_annual: float = 0.0) -> pd.Series:
    """Annualised Sharpe over a trailing window, one value per day after warm-up.

    Mean excess return over the window divided by its volatility, scaled to a
    year. The first ``window - 1`` days are NaN (not enough history yet).
    """
    excess = returns - rf_annual / periods
    mean = excess.rolling(window).mean()
    vol = returns.rolling(window).std()
    return (mean / vol) * np.sqrt(periods)


def rolling_beta(returns: pd.Series,
                 benchmark: pd.Series,
                 window: int = 126) -> pd.Series:
    """Trailing beta of the strategy to a benchmark: cov(r, b) / var(b).

    Aligned on the dates the two share; a beta near 0 is the market-neutral claim
    a long/short book is supposed to keep, and this shows whether it actually
    held through time. Result is reindexed to the strategy's own dates.
    """
    joined = pd.concat([returns.rename("r"), benchmark.rename("b")], axis=1).dropna()
    cov = joined["r"].rolling(window).cov(joined["b"])
    var = joined["b"].rolling(window).var()
    return (cov / var).reindex(returns.index)


def per_year_table(returns: pd.Series,
                   periods: int = TRADING_DAYS_PER_YEAR,
                   rf_annual: float = 0.0) -> pd.DataFrame:
    """Per-calendar-year return, volatility, Sharpe and worst drawdown.

    ``return`` is the year's actual compound return, not an annualised figure, so
    a partial first or last year is reported as what it was rather than
    extrapolated. Volatility and Sharpe are annualised in the usual way.
    """
    rows = []
    for year, r in returns.groupby(returns.index.year):
        r = r.dropna()
        if r.empty:
            continue
        rows.append({
            "year": int(year),
            "return": float((1.0 + r).prod() - 1.0),
            "vol": annualized_vol(r, periods),
            "sharpe": sharpe_ratio(r, rf_annual, periods),
            "max_drawdown": max_drawdown(r),
            "days": int(len(r)),
        })
    return pd.DataFrame(rows).set_index("year")
