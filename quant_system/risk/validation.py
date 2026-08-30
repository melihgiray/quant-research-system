"""Backtests for Value-at-Risk forecasts.

The Kupiec test checks whether a VaR model produces the right *number* of
exceptions.  It does not check whether they arrive independently.  A model
that has the right exception rate but fails repeatedly during a sell-off is
not calibrated in the way a risk limit needs it to be.
"""

from __future__ import annotations

from math import log

import numpy as np
import pandas as pd
from scipy.stats import chi2
from scipy.stats import binom


def christoffersen_independence_test(
    returns: pd.Series,
    var: float | pd.Series,
    level: float = 0.95,
) -> dict:
    """Test whether VaR exceptions are independent over time.

    ``var`` is a positive loss threshold.  The returned likelihood-ratio
    statistic is asymptotically chi-squared with one degree of freedom.
    """
    r = pd.Series(returns, dtype=float)
    threshold = pd.Series(var, index=r.index, dtype=float)
    hits = (r < -threshold).dropna().astype(int)
    if len(hits) < 2:
        return {"lr_ind": np.nan, "pvalue": np.nan, "reject": False,
                "n_exceptions": int(hits.sum()), "n_obs": int(len(hits))}

    prev, curr = hits.iloc[:-1].to_numpy(), hits.iloc[1:].to_numpy()
    n00 = int(((prev == 0) & (curr == 0)).sum())
    n01 = int(((prev == 0) & (curr == 1)).sum())
    n10 = int(((prev == 1) & (curr == 0)).sum())
    n11 = int(((prev == 1) & (curr == 1)).sum())

    def _log_binom(success: int, failure: int) -> float:
        total = success + failure
        if total == 0:
            return 0.0
        p = np.clip(success / total, 1e-12, 1 - 1e-12)
        return success * log(p) + failure * log(1 - p)

    unrestricted = _log_binom(n01, n00) + _log_binom(n11, n10)
    restricted = _log_binom(n01 + n11, n00 + n10)
    lr = max(0.0, -2.0 * (restricted - unrestricted))
    pvalue = float(chi2.sf(lr, df=1))
    return {
        "lr_ind": float(lr), "pvalue": pvalue, "reject": bool(pvalue < 1 - level),
        "n_exceptions": int(hits.sum()), "n_obs": int(len(hits)),
        "transition_counts": {"n00": n00, "n01": n01, "n10": n10, "n11": n11},
    }


def christoffersen_conditional_coverage_test(
    returns: pd.Series,
    var: float | pd.Series,
    level: float = 0.95,
) -> dict:
    """Combine Kupiec coverage and Christoffersen independence tests.

    The conditional-coverage statistic has two degrees of freedom and rejects
    forecasts with either the wrong exception frequency or clustered failures.
    """
    from .metrics import kupiec_pof_test

    r = pd.Series(returns, dtype=float).dropna()
    if isinstance(var, pd.Series):
        aligned = pd.concat([r, var.rename("var")], axis=1).dropna()
        r, threshold = aligned.iloc[:, 0], aligned["var"]
    else:
        threshold = float(var)
    lr_pof, kupiec_pvalue = kupiec_pof_test(r, threshold, level=level)
    kupiec = {"lr_pof": lr_pof, "pvalue": kupiec_pvalue,
              "reject": bool(kupiec_pvalue < 1 - level)}
    independence = christoffersen_independence_test(r, threshold, level=level)
    lr_cc = float(kupiec["lr_pof"] + independence["lr_ind"])
    pvalue = float(chi2.sf(lr_cc, df=2))
    return {
        "lr_cc": lr_cc, "pvalue": pvalue, "reject": bool(pvalue < 1 - level),
        "kupiec": kupiec, "independence": independence,
    }


def basel_traffic_light(
    returns: pd.Series,
    var: float | pd.Series,
    level: float = 0.99,
) -> dict:
    """Classify a VaR model using Basel's binomial exception zones.

    The familiar 250-day, 99% model has green at four or fewer exceptions,
    yellow at five through nine, and red at ten or more.  The cutoffs here are
    calculated from the binomial distribution, so the same rule works for a
    different sample size or confidence level without hard-coded thresholds.
    """
    r = pd.Series(returns, dtype=float)
    threshold = pd.Series(var, index=r.index, dtype=float)
    aligned = pd.concat([r.rename("return"), threshold.rename("var")], axis=1).dropna()
    n = len(aligned)
    if n == 0:
        return {"zone": "unknown", "exceptions": 0, "n_obs": 0,
                "green_limit": np.nan, "yellow_limit": np.nan}
    exceptions = int((aligned["return"] < -aligned["var"].abs()).sum())
    p = 1.0 - level
    if n == 250 and np.isclose(level, 0.99):
        # Basel's published traffic-light table has fixed canonical boundaries.
        green_limit, yellow_limit = 4, 9
    else:
        green_limit = int(binom.ppf(0.95, n, p))
        yellow_limit = int(binom.ppf(0.9999, n, p))
    zone = "green" if exceptions <= green_limit else "yellow" if exceptions <= yellow_limit else "red"
    return {"zone": zone, "exceptions": exceptions, "n_obs": n,
            "green_limit": green_limit, "yellow_limit": yellow_limit}


def dynamic_quantile_test(
    returns: pd.Series,
    var: float | pd.Series,
    level: float = 0.95,
    lags: int = 4,
) -> dict:
    """Engle-Manganelli dynamic-quantile test for VaR forecast adequacy.

    The regression asks whether centred exception hits can be predicted by their
    own recent history or by the forecast threshold. A calibrated VaR forecast
    should leave neither relationship behind. The DQ statistic is ``n x R^2``.
    """
    if lags < 1:
        raise ValueError("lags must be positive")
    r = pd.Series(returns, dtype=float)
    threshold = pd.Series(var, index=r.index, dtype=float)
    df = pd.concat([r.rename("return"), threshold.rename("var")], axis=1).dropna()
    hits = (df["return"] < -df["var"].abs()).astype(float) - (1.0 - level)
    if len(hits) <= lags + 5:
        return {"dq": np.nan, "pvalue": np.nan, "reject": False, "n_obs": 0}
    design = [np.ones(len(hits) - lags), df["var"].iloc[lags:].to_numpy()]
    for lag in range(1, lags + 1):
        design.append(hits.shift(lag).iloc[lags:].to_numpy())
    x = np.column_stack(design)
    y = hits.iloc[lags:].to_numpy()
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ beta
    ss_total = float(np.sum((y - y.mean()) ** 2))
    r2 = 0.0 if ss_total == 0 else max(0.0, 1.0 - float(np.sum((y - fitted) ** 2)) / ss_total)
    dq = float(len(y) * r2)
    pvalue = float(chi2.sf(dq, df=x.shape[1]))
    return {"dq": dq, "pvalue": pvalue, "reject": bool(pvalue < 1 - level),
            "n_obs": int(len(y)), "n_regressors": int(x.shape[1])}
