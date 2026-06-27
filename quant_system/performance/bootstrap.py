"""Bootstrap confidence intervals for performance statistics.

A point estimate of Sharpe or CAGR hides its own sampling error. We attach error
bars with the **stationary bootstrap** (Politis & Romano, 1994): resample the
return series in blocks of random (geometric) length so that serial correlation —
which a naive i.i.d. bootstrap would destroy — is preserved on average. The
average block length is the one knob; it should span the dependence horizon
(~2 trading weeks here).

This is the empirical complement to Day 1's analytic significance: PSR/DSR give a
parametric p-value, the bootstrap gives a distribution you can actually see.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import numpy as np
import pandas as pd

from ..config import TRADING_DAYS_PER_YEAR


@dataclass
class ConfidenceInterval:
    """A point estimate with a percentile-bootstrap interval."""
    point: float
    low: float
    high: float
    level: float

    def __str__(self) -> str:
        return f"{self.point:.2f}  [{self.low:.2f}, {self.high:.2f}] ({self.level:.0%})"


def _stationary_indices(n: int, avg_block: float, n_boot: int, rng) -> np.ndarray:
    """Resampling indices for the stationary bootstrap.

    At each step we either advance to the next observation (prob 1 - 1/avg_block)
    or jump to a fresh random one (prob 1/avg_block), wrapping circularly. This
    yields blocks of geometric length with mean ``avg_block``. Vectorised across
    the ``n_boot`` replications; the only Python loop is over the ``n`` time steps.
    """
    p = 1.0 / max(avg_block, 1.0)
    idx = np.empty((n_boot, n), dtype=np.int64)
    cur = rng.integers(0, n, size=n_boot)
    fresh = rng.integers(0, n, size=(n_boot, n))
    restart = rng.random((n_boot, n)) < p
    idx[:, 0] = cur
    for t in range(1, n):
        cur = np.where(restart[:, t], fresh[:, t], (cur + 1) % n)
        idx[:, t] = cur
    return idx


def bootstrap_distribution(returns: pd.Series,
                           metric: Callable[[np.ndarray], float],
                           n_boot: int = 1000, avg_block: int = 10,
                           seed: int = 7) -> np.ndarray:
    """Return ``n_boot`` resampled values of ``metric`` applied to the returns.

    Parameters
    ----------
    returns : pd.Series
        Periodic returns.
    metric : callable(np.ndarray) -> float
        Statistic computed on each resampled return path (1-D array).
    n_boot, avg_block, seed : bootstrap controls.
    """
    r = returns.dropna().to_numpy(dtype=float)
    n = len(r)
    if n < 20:
        return np.array([])
    rng = np.random.default_rng(seed)
    idx = _stationary_indices(n, avg_block, n_boot, rng)
    paths = r[idx]                                   # (n_boot, n)
    return np.array([metric(paths[b]) for b in range(n_boot)])


# --- default metrics on a 1-D return array ----------------------------------- #
def _ann_sharpe(arr: np.ndarray, periods: int = TRADING_DAYS_PER_YEAR) -> float:
    sd = arr.std(ddof=1)
    return float(np.sqrt(periods) * arr.mean() / sd) if sd > 0 else float("nan")


def _cagr(arr: np.ndarray, periods: int = TRADING_DAYS_PER_YEAR) -> float:
    growth = float(np.prod(1.0 + arr))
    years = len(arr) / periods
    return growth ** (1.0 / years) - 1.0 if growth > 0 and years > 0 else float("nan")


def _percentile_ci(samples: np.ndarray, point: float, level: float) -> ConfidenceInterval:
    if samples.size == 0:
        nan = float("nan")
        return ConfidenceInterval(point, nan, nan, level)
    a = (1.0 - level) / 2.0
    lo, hi = np.nanpercentile(samples, [a * 100, (1 - a) * 100])
    return ConfidenceInterval(point, float(lo), float(hi), level)


def sharpe_confidence_interval(returns: pd.Series, level: float = 0.95,
                               n_boot: int = 1000, avg_block: int = 10,
                               periods: int = TRADING_DAYS_PER_YEAR,
                               seed: int = 7) -> ConfidenceInterval:
    """Bootstrap CI for the annualised Sharpe ratio."""
    point = _ann_sharpe(returns.dropna().to_numpy(float), periods)
    dist = bootstrap_distribution(returns, lambda a: _ann_sharpe(a, periods),
                                  n_boot, avg_block, seed)
    return _percentile_ci(dist, point, level)


def cagr_confidence_interval(returns: pd.Series, level: float = 0.95,
                             n_boot: int = 1000, avg_block: int = 10,
                             periods: int = TRADING_DAYS_PER_YEAR,
                             seed: int = 7) -> ConfidenceInterval:
    """Bootstrap CI for the compound annual growth rate."""
    point = _cagr(returns.dropna().to_numpy(float), periods)
    dist = bootstrap_distribution(returns, lambda a: _cagr(a, periods),
                                  n_boot, avg_block, seed)
    return _percentile_ci(dist, point, level)


def bootstrap_summary(returns: pd.Series, level: float = 0.95,
                      n_boot: int = 1000, avg_block: int = 10,
                      periods: int = TRADING_DAYS_PER_YEAR, seed: int = 7) -> List[str]:
    """Tearsheet-ready lines with bootstrap CIs and P(Sharpe>0)."""
    sr = sharpe_confidence_interval(returns, level, n_boot, avg_block, periods, seed)
    cg = cagr_confidence_interval(returns, level, n_boot, avg_block, periods, seed)
    dist = bootstrap_distribution(returns, lambda a: _ann_sharpe(a, periods),
                                  n_boot, avg_block, seed)
    p_pos = float(np.mean(dist > 0)) if dist.size else float("nan")
    return [
        f"BOOTSTRAP {level:.0%} CI (stationary, block~{avg_block}d, {n_boot} reps)",
        f" Sharpe   {sr}",
        f" CAGR     {cg.point:+.1%}  [{cg.low:+.1%}, {cg.high:+.1%}]",
        f" P(Sharpe>0) bootstrap   {p_pos:.0%}",
    ]
