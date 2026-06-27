"""Statistical significance of a Sharpe ratio.

A point-estimate Sharpe is almost useless on its own: it ignores (a) that returns
are skewed and fat-tailed, which changes the estimator's variance, and (b) that
you probably tried many strategies and reported the best, which inflates the
maximum Sharpe you'd see by chance. This module implements the three Bailey &
López de Prado tools that address exactly this:

  * Probabilistic Sharpe Ratio (PSR) — P(true SR > benchmark SR), corrected for
    skew/kurtosis.  (Bailey & López de Prado, 2012)
  * Deflated Sharpe Ratio (DSR) — PSR where the benchmark is the *expected maximum*
    Sharpe across N independent trials, so a Sharpe found by searching is
    discounted.  (Bailey & López de Prado, 2014)
  * Minimum Track Record Length (minTRL) — the sample length needed for the Sharpe
    to be significant at a chosen confidence.

All maths is done on the *per-period* (e.g. daily) Sharpe, which is the correct
frequency for the estimator's sampling distribution; we annualise only for display.

References:
  Bailey, D. & López de Prado, M. (2012) "The Sharpe Ratio Efficient Frontier",
  Journal of Risk 15(2). (2014) "The Deflated Sharpe Ratio", JPM 40(5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from ..config import TRADING_DAYS_PER_YEAR

# Euler–Mascheroni constant, used in the expected-maximum-of-N-Gaussians formula.
_EULER_GAMMA = 0.5772156649015329


@dataclass
class SharpeMoments:
    """The four numbers PSR/DSR depend on, at the return frequency."""
    sr: float          # per-period Sharpe (mean/std, NOT annualised)
    skew: float        # γ3 (0 for normal)
    kurtosis: float    # γ4, Pearson (3 for normal)
    n: int             # number of observations


def sharpe_moments(returns: pd.Series, rf_per_period: float = 0.0) -> SharpeMoments:
    """Compute the per-period Sharpe and the higher moments of the return series.

    Parameters
    ----------
    returns : pd.Series
        Periodic returns.
    rf_per_period : float
        Per-period risk-free rate to subtract before computing the Sharpe.
    """
    r = returns.dropna().astype(float) - rf_per_period
    n = int(len(r))
    if n < 3 or r.std(ddof=1) == 0:
        return SharpeMoments(sr=float("nan"), skew=0.0, kurtosis=3.0, n=n)
    sr = float(r.mean() / r.std(ddof=1))
    sk = float(stats.skew(r.values, bias=False))
    ku = float(stats.kurtosis(r.values, fisher=False, bias=False))  # Pearson: normal=3
    return SharpeMoments(sr=sr, skew=sk, kurtosis=ku, n=n)


def _sr_estimator_variance(m: SharpeMoments) -> float:
    """Variance of the (per-period) Sharpe estimator under non-normal returns.

    Var(SR_hat) = (1 - γ3·SR + ((γ4-1)/4)·SR²) / (n-1)

    For normal returns (γ3=0, γ4=3) this reduces to (1 + SR²/2)/(n-1), the classic
    Lo (2002) result. Fatter tails / negative skew inflate it — i.e. make the same
    Sharpe less significant.
    """
    denom = 1.0 - m.skew * m.sr + ((m.kurtosis - 1.0) / 4.0) * m.sr ** 2
    return max(denom, 1e-12) / max(m.n - 1, 1)


def probabilistic_sharpe_ratio(returns: pd.Series, benchmark_sr_annual: float = 0.0,
                               periods: int = TRADING_DAYS_PER_YEAR,
                               rf_annual: float = 0.0) -> float:
    """P(true Sharpe > benchmark), accounting for skew, kurtosis and sample size.

    Parameters
    ----------
    returns : pd.Series
        Periodic returns.
    benchmark_sr_annual : float
        The annualised Sharpe to test against (0 = "better than nothing").
    periods : int
        Periods per year (to convert the annual benchmark to per-period).
    rf_annual : float
        Annual risk-free rate.

    Returns
    -------
    float
        PSR in [0, 1]. > 0.95 is the usual "significant" bar.
    """
    m = sharpe_moments(returns, rf_per_period=rf_annual / periods)
    if math.isnan(m.sr):
        return float("nan")
    sr_star = benchmark_sr_annual / math.sqrt(periods)     # de-annualise the benchmark
    se = math.sqrt(_sr_estimator_variance(m))
    if se == 0:
        return float("nan")
    return float(stats.norm.cdf((m.sr - sr_star) / se))


def expected_max_sharpe(sr_variance_across_trials: float, n_trials: int) -> float:
    """Expected maximum (per-period) Sharpe from N independent random trials.

    E[max SR] ≈ √V · [ (1-γ)·Φ⁻¹(1 - 1/N) + γ·Φ⁻¹(1 - 1/(N·e)) ]

    where V is the variance of the Sharpe ratios across the trials and γ is the
    Euler–Mascheroni constant. This is the benchmark a real strategy must beat:
    even pure noise, searched N times, produces a best-of-N Sharpe this large.
    """
    if n_trials <= 1 or sr_variance_across_trials <= 0:
        return 0.0
    n = float(n_trials)
    term = ((1 - _EULER_GAMMA) * stats.norm.ppf(1 - 1.0 / n)
            + _EULER_GAMMA * stats.norm.ppf(1 - 1.0 / (n * math.e)))
    return float(math.sqrt(sr_variance_across_trials) * term)


def deflated_sharpe_ratio(returns: pd.Series, n_trials: int,
                          sr_variance_across_trials: Optional[float] = None,
                          periods: int = TRADING_DAYS_PER_YEAR,
                          rf_annual: float = 0.0) -> dict:
    """Deflated Sharpe Ratio: PSR against the expected best-of-N-trials Sharpe.

    Parameters
    ----------
    returns : pd.Series
        Periodic returns of the *selected* strategy.
    n_trials : int
        Number of independent configurations effectively searched. This is a
        judgement call you must be able to justify; more trials -> lower DSR.
    sr_variance_across_trials : float, optional
        Variance of the per-period Sharpe across trials. If omitted, we use the
        Sharpe estimator's own variance as a documented, conservative proxy (the
        right value is the empirical variance of all trial Sharpes, when retained).
    periods, rf_annual : see probabilistic_sharpe_ratio.

    Returns
    -------
    dict
        {dsr, sr_star_annual, n_trials, psr_vs_zero}.
    """
    m = sharpe_moments(returns, rf_per_period=rf_annual / periods)
    if math.isnan(m.sr):
        return {"dsr": float("nan"), "sr_star_annual": float("nan"),
                "n_trials": n_trials, "psr_vs_zero": float("nan")}

    v = sr_variance_across_trials if sr_variance_across_trials is not None \
        else _sr_estimator_variance(m)
    sr_star = expected_max_sharpe(v, n_trials)             # per-period threshold
    se = math.sqrt(_sr_estimator_variance(m))
    dsr = float(stats.norm.cdf((m.sr - sr_star) / se)) if se > 0 else float("nan")
    return {
        "dsr": dsr,
        "sr_star_annual": sr_star * math.sqrt(periods),
        "n_trials": n_trials,
        "psr_vs_zero": probabilistic_sharpe_ratio(returns, 0.0, periods, rf_annual),
    }


def min_track_record_length(returns: pd.Series, benchmark_sr_annual: float = 0.0,
                            confidence: float = 0.95,
                            periods: int = TRADING_DAYS_PER_YEAR,
                            rf_annual: float = 0.0) -> dict:
    """Minimum number of observations for the Sharpe to be significant.

    minTRL = 1 + (1 - γ3·SR + ((γ4-1)/4)·SR²) · ( Φ⁻¹(confidence) / (SR - SR*) )²

    Returns the length in periods and in years. Infinite if the Sharpe does not
    exceed the benchmark (you can never prove significance against a higher bar).
    """
    m = sharpe_moments(returns, rf_per_period=rf_annual / periods)
    sr_star = benchmark_sr_annual / math.sqrt(periods)
    if math.isnan(m.sr) or m.sr <= sr_star:
        return {"min_periods": float("inf"), "min_years": float("inf"),
                "have_periods": m.n, "significant_now": False}
    denom = 1.0 - m.skew * m.sr + ((m.kurtosis - 1.0) / 4.0) * m.sr ** 2
    z = stats.norm.ppf(confidence)
    min_periods = 1.0 + denom * (z / (m.sr - sr_star)) ** 2
    return {
        "min_periods": float(min_periods),
        "min_years": float(min_periods / periods),
        "have_periods": m.n,
        "significant_now": bool(m.n >= min_periods),
    }


def significance_summary(returns: pd.Series, n_trials: int = 1,
                         periods: int = TRADING_DAYS_PER_YEAR,
                         rf_annual: float = 0.0,
                         confidence: float = 0.95) -> List[str]:
    """Tearsheet-ready lines summarising PSR, DSR and minTRL."""
    psr = probabilistic_sharpe_ratio(returns, 0.0, periods, rf_annual)
    dsr = deflated_sharpe_ratio(returns, n_trials, periods=periods, rf_annual=rf_annual)
    trl = min_track_record_length(returns, 0.0, confidence, periods, rf_annual)

    def pct(x):
        return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:6.1%}"

    trl_txt = ("already met" if trl["significant_now"]
               else (f"{trl['min_years']:.1f} yrs" if math.isfinite(trl["min_years"]) else "∞"))
    return [
        "STATISTICAL SIGNIFICANCE (Bailey & López de Prado)",
        f" PSR  P(SR>0)            {pct(psr)}",
        f" DSR  (deflated, N={dsr['n_trials']:<3d}) {pct(dsr['dsr'])}   "
        f"benchmark SR*={dsr['sr_star_annual']:+.2f}",
        f" minTRL @ {confidence:.0%}           {trl_txt}  (have {trl['have_periods']} obs)",
    ]
