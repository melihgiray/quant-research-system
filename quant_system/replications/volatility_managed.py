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
from hashlib import sha256
from io import BytesIO, StringIO
from typing import Dict, List
from zipfile import ZipFile

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm

from ..config import TRADING_DAYS_PER_YEAR
from ..performance.analytics import compute_metrics


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


@dataclass(frozen=True)
class KenFrenchDataset:
    """Daily Ken French returns together with immutable source provenance."""

    returns: pd.DataFrame
    source_url: str
    sha256: str


@dataclass(frozen=True)
class TimingRegression:
    """HAC regression of managed returns on their unmanaged counterpart."""

    alpha_daily: float
    alpha_annual: float
    alpha_tstat: float
    beta: float
    r_squared: float
    n_obs: int


def download_ken_french_daily_with_metadata(
    dataset: str = "F-F_Research_Data_Factors_daily", timeout: int = 30
) -> KenFrenchDataset:
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
    KenFrenchDataset
        Daily decimal returns plus the source URL and the SHA-256 digest of the
        exact ZIP payload. Ken French publishes the source files in percent, so
        values are divided by 100 here.

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
    payload = response.content
    with ZipFile(BytesIO(payload)) as archive:
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
    returns = validate_factor_returns(frame.apply(pd.to_numeric, errors="coerce").div(100.0))
    return KenFrenchDataset(returns=returns, source_url=url, sha256=sha256(payload).hexdigest())


def download_ken_french_daily(dataset: str = "F-F_Research_Data_Factors_daily", timeout: int = 30) -> pd.DataFrame:
    """Download daily Ken French returns without the optional provenance wrapper."""
    return download_ken_french_daily_with_metadata(dataset, timeout).returns


def validate_factor_returns(
    returns: pd.DataFrame, max_missing_fraction: float = 0.01
) -> pd.DataFrame:
    """Validate a daily factor-return table before using it in a study.

    The function does not fill values or reorder observations. Any duplicate or
    unsorted dates, empty table, non-datetime index, or materially incomplete
    column is rejected so a data problem cannot masquerade as a research result.
    """
    if not 0.0 <= max_missing_fraction < 1.0:
        raise ValueError("max_missing_fraction must be in [0, 1)")
    if returns.empty or returns.shape[1] == 0:
        raise ValueError("factor returns must contain at least one row and column")
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("factor returns must use a DatetimeIndex")
    if not returns.index.is_monotonic_increasing or not returns.index.is_unique:
        raise ValueError("factor-return dates must be sorted and unique")
    missing = returns.isna().mean()
    bad = missing[missing > max_missing_fraction]
    if not bad.empty:
        detail = ", ".join(f"{column}={fraction:.1%}" for column, fraction in bad.items())
        raise ValueError(f"factor-return columns exceed missing-data limit: {detail}")
    return returns


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
    max_exposure: float | None = None,
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
    max_exposure:
        Optional absolute cap on the normalized factor exposure. ``None`` keeps
        the paper-style uncapped rule. A cap is useful for an implementation
        stress test, not for replacing the baseline replication.
    """
    series = returns.dropna().astype(float).sort_index()
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("returns must use a DatetimeIndex")
    if train_days < vol_lookback + 2 or test_days < 1:
        raise ValueError("training and test windows are too short")
    if max_exposure is not None and max_exposure <= 0:
        raise ValueError("max_exposure must be positive when provided")

    raw = inverse_variance_exposure(series, vol_lookback)
    unmanaged_parts, managed_parts, exposure_parts = [], [], []
    folds: List[Dict[str, object]] = []
    train_end = train_days

    while train_end + test_days <= len(series):
        train_index = series.index[:train_end]
        test_index = series.index[train_end:train_end + test_days]
        multiplier = _normalization_multiplier(series.loc[train_index], raw.loc[train_index])
        exposure = (multiplier * raw.loc[test_index]).rename("exposure")
        if max_exposure is not None:
            exposure = exposure.clip(lower=-max_exposure, upper=max_exposure)
        managed = (exposure * series.loc[test_index]).rename("managed")

        unmanaged_parts.append(series.loc[test_index])
        managed_parts.append(managed)
        exposure_parts.append(exposure)
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
    turnover = exposure.diff().abs().rename("turnover")
    turnover.iloc[0] = abs(exposure.iloc[0])
    return VolatilityManagedResult(unmanaged, managed, exposure, turnover, folds)


def fold_statistics(result: VolatilityManagedResult, periods: int = TRADING_DAYS_PER_YEAR) -> pd.DataFrame:
    """Return gross performance diagnostics for each out-of-sample fold.

    The rows are not independent observations; they are a descriptive check on
    concentration of the aggregate result.  In particular, the table makes it
    clear when an apparent long-history improvement comes from a small number
    of test years.
    """
    rows = []
    for number, fold in enumerate(result.folds, start=1):
        managed = result.managed_returns.loc[fold["test_start"]:fold["test_end"]]
        unmanaged = result.unmanaged_returns.loc[managed.index]
        managed_std = managed.std(ddof=1)
        rows.append({
            "fold": number,
            "test_start": fold["test_start"],
            "test_end": fold["test_end"],
            "multiplier": fold["multiplier"],
            "unmanaged_return": float((1.0 + unmanaged).prod() - 1.0),
            "managed_return": float((1.0 + managed).prod() - 1.0),
            "managed_sharpe": float(np.sqrt(periods) * managed.mean() / managed_std)
            if managed_std > 0 else float("nan"),
        })
    return pd.DataFrame(rows).set_index("fold")


def subperiod_statistics(
    result: VolatilityManagedResult, years: int = 10
) -> pd.DataFrame:
    """Compare managed and unmanaged returns in fixed calendar-year blocks.

    This descriptive split is not another model-selection exercise. It makes
    the variation hidden by the full-sample result visible and uses only the
    already out-of-sample return stream.
    """
    if years < 1:
        raise ValueError("years must be positive")
    first_year = int(result.managed_returns.index.min().year)
    last_year = int(result.managed_returns.index.max().year)
    rows = []
    for start in range(first_year, last_year + 1, years):
        end = min(start + years - 1, last_year)
        mask = (result.managed_returns.index.year >= start) & (result.managed_returns.index.year <= end)
        managed = result.managed_returns.loc[mask]
        unmanaged = result.unmanaged_returns.loc[managed.index]
        if managed.empty:
            continue
        m = compute_metrics(managed)
        u = compute_metrics(unmanaged)
        rows.append({
            "period": f"{start}-{end}",
            "n_days": len(managed),
            "unmanaged_sharpe": u["sharpe"],
            "managed_sharpe": m["sharpe"],
            "unmanaged_return": u["ann_return"],
            "managed_return": m["ann_return"],
        })
    return pd.DataFrame(rows).set_index("period")


def timing_regression(
    managed_returns: pd.Series,
    unmanaged_returns: pd.Series,
    hac_lags: int = 5,
    periods: int = TRADING_DAYS_PER_YEAR,
) -> TimingRegression:
    """Regress managed returns on the unmanaged factor with HAC errors.

    The intercept is a timing alpha conditional on the base factor, not evidence
    of a standalone tradable alpha. HAC standard errors allow for serial
    dependence induced by volatility scaling.
    """
    if hac_lags < 0:
        raise ValueError("hac_lags must be non-negative")
    data = pd.concat([managed_returns.rename("managed"), unmanaged_returns.rename("unmanaged")], axis=1).dropna()
    if len(data) < max(30, hac_lags + 5):
        raise ValueError("too few paired observations for timing regression")
    model = sm.OLS(data["managed"], sm.add_constant(data["unmanaged"])).fit(
        cov_type="HAC", cov_kwds={"maxlags": hac_lags}
    )
    alpha = float(model.params["const"])
    return TimingRegression(
        alpha_daily=alpha,
        alpha_annual=alpha * periods,
        alpha_tstat=float(model.tvalues["const"]),
        beta=float(model.params["unmanaged"]),
        r_squared=float(model.rsquared),
        n_obs=int(model.nobs),
    )
