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
