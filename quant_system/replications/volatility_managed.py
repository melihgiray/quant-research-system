"""Causal replication tools for Moreira and Muir (2017).

The paper, *Volatility-Managed Portfolios*, scales a factor by the inverse of
its recently realised variance.  This module implements that rule on daily
excess-return series.  It deliberately calibrates the scale multiplier inside
each expanding training window rather than over the full sample.  The paper
uses a full-sample equal-volatility normalization for descriptive comparisons;
using that normalization in an out-of-sample test would leak future data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO, StringIO
from typing import Dict, List
from zipfile import ZipFile

import numpy as np
import pandas as pd
import requests

from ..config import TRADING_DAYS_PER_YEAR


KEN_FRENCH_FTP = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"


@dataclass
class VolatilityManagedResult:
    """Out-of-sample results from the expanding-window replication.

    Attributes
    ----------
    unmanaged_returns:
        The original daily excess factor return in each retained test window.
    managed_returns:
        The inverse-variance scaled return.  It is gross of implementation
        costs because the source factor series is not itself a tradable asset.
    exposure:
        Causally known factor exposure used for each test day.
    turnover:
        Daily absolute exposure change, a transparent proxy for rebalancing.
    folds:
        Training and test boundaries and the multiplier fitted in each fold.
    """

    unmanaged_returns: pd.Series
    managed_returns: pd.Series
    exposure: pd.Series
    turnover: pd.Series
    folds: List[Dict[str, object]] = field(default_factory=list)


def download_ken_french_daily(dataset: str = "F-F_Research_Data_Factors_daily", timeout: int = 30) -> pd.DataFrame:
    """Download and parse a daily Ken French Data Library CSV ZIP file.

    Parameters
    ----------
    dataset:
        File stem used by the library, for example
        ``F-F_Research_Data_Factors_daily``.
    timeout:
        Network timeout in seconds.

    Returns
    -------
    pd.DataFrame
        Daily decimal returns indexed by timestamp.  Ken French publishes the
        source files in percent, so values are divided by 100 here.

    Raises
    ------
    ValueError
        If the downloaded archive does not contain a recognisable daily table.
    requests.HTTPError
        If the official source cannot be retrieved.
    """
    url = f"{KEN_FRENCH_FTP}/{dataset}_CSV.zip"
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    with ZipFile(BytesIO(response.content)) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not names:
            raise ValueError("Ken French archive contains no CSV file")
        text = archive.read(names[0]).decode("latin-1")

    lines = text.splitlines()
    header = next((i for i, line in enumerate(lines) if "Mkt-RF" in line), None)
    if header is None:
        raise ValueError("could not find a daily return header in Ken French data")
    end = header + 1
    while end < len(lines) and lines[end].split(",", 1)[0].strip().isdigit():
        end += 1

    frame = pd.read_csv(StringIO("\n".join(lines[header:end])), index_col=0)
    frame.index = pd.to_datetime(frame.index.astype(str).str.strip(), format="%Y%m%d")
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame.apply(pd.to_numeric, errors="coerce").div(100.0)


def inverse_variance_exposure(returns: pd.Series, lookback: int = 21) -> pd.Series:
    """Return the causal inverse-realised-variance exposure for a factor.

    The value assigned to date ``t`` is calculated from returns through
    ``t - 1``.  In particular, changing the return on date ``t`` cannot change
    that day's exposure.  This is the central timing requirement of the
    replication.
    """
    if lookback < 2:
        raise ValueError("lookback must be at least two observations")
    clean = returns.astype(float).copy()
    variance = clean.rolling(lookback, min_periods=lookback).var(ddof=1).shift(1)
    return (1.0 / variance).replace([np.inf, -np.inf], np.nan).rename("raw_exposure")


def _normalization_multiplier(returns: pd.Series, raw_exposure: pd.Series) -> float:
    """Match managed and unmanaged training volatility without future data."""
    aligned = pd.concat([returns.rename("return"), raw_exposure.rename("exposure")], axis=1).dropna()
    if len(aligned) < 2:
        raise ValueError("training window has too little realised-volatility history")
    unmanaged_vol = aligned["return"].std(ddof=1)
    managed_vol = (aligned["return"] * aligned["exposure"]).std(ddof=1)
    if unmanaged_vol <= 0 or managed_vol <= 0:
        raise ValueError("training returns must have non-zero variance")
    return float(unmanaged_vol / managed_vol)


def walk_forward_volatility_managed(
    returns: pd.Series,
    train_days: int = 252 * 5,
    test_days: int = 252,
    vol_lookback: int = 21,
) -> VolatilityManagedResult:
    """Run an expanding, causal replication of the volatility-managed rule.

    A scale multiplier is fitted only on each fold's training data, then held
    fixed over its subsequent test window.  Test windows do not overlap and the
    returned series contains only those out-of-sample observations.

    Parameters
    ----------
    returns:
        Daily *excess* returns for one factor or portfolio.
    train_days:
        Initial expanding in-sample length.
    test_days:
        Length of each following out-of-sample segment.
    vol_lookback:
        Number of prior daily returns used for realised variance.
    """
    series = returns.dropna().astype(float).sort_index()
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("returns must use a DatetimeIndex")
    if train_days < vol_lookback + 2 or test_days < 1:
        raise ValueError("training and test windows are too short")

    raw = inverse_variance_exposure(series, vol_lookback)
    unmanaged_parts, managed_parts, exposure_parts, turnover_parts = [], [], [], []
    folds: List[Dict[str, object]] = []
    train_end = train_days

    while train_end + test_days <= len(series):
        train_index = series.index[:train_end]
        test_index = series.index[train_end:train_end + test_days]
        multiplier = _normalization_multiplier(series.loc[train_index], raw.loc[train_index])
        exposure = (multiplier * raw.loc[test_index]).rename("exposure")
        managed = (exposure * series.loc[test_index]).rename("managed")
        turnover = exposure.diff().abs().fillna(0.0).rename("turnover")

        unmanaged_parts.append(series.loc[test_index])
        managed_parts.append(managed)
        exposure_parts.append(exposure)
        turnover_parts.append(turnover)
        folds.append({
            "train_start": train_index[0],
            "train_end": train_index[-1],
            "test_start": test_index[0],
            "test_end": test_index[-1],
            "multiplier": multiplier,
        })
        train_end += test_days

    if not folds:
        raise ValueError("not enough returns for one complete walk-forward fold")
    unmanaged = pd.concat(unmanaged_parts)
    managed = pd.concat(managed_parts)
    exposure = pd.concat(exposure_parts)
    turnover = pd.concat(turnover_parts)
    turnover.iloc[0] = exposure.iloc[0]
    return VolatilityManagedResult(unmanaged, managed, exposure, turnover, folds)
