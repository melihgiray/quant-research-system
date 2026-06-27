"""Fama-French style 3-factor decomposition of strategy returns.

This is the step that separates "I ran a backtest" from "I understand my
backtest". We regress the strategy's daily excess return on three factor
returns and read off:

    alpha (intercept)  — return NOT explained by the factors. Annualised and
                         tested for significance. This is the part you can
                         honestly call skill/edge.
    betas              — exposure to each factor. A "market-neutral" strategy
                         should have market beta ~ 0.
    R^2                — how much of the variance the factors explain. High R^2
                         with ~0 alpha means you are just selling factor beta.

Factors are built from liquid ETF proxies (no external data dependency):
    Market (MKT) = SPY - rf
    Size   (SMB) = IWM - SPY     (small-minus-big proxy)
    Value  (HML) = VTV - VUG     (value-minus-growth proxy)

Standard errors are Newey-West (HAC): daily strategy returns are mildly
autocorrelated and heteroskedastic, which deflates naive OLS standard errors and
inflates t-stats. HAC corrects for that, so a t-stat > 2 here is trustworthy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ..config import TRADING_DAYS_PER_YEAR
from ..data.universe import FACTOR_ETFS


@dataclass
class FactorDecompResult:
    """Result of the 3-factor regression."""

    alpha_daily: float
    alpha_annual: float
    alpha_tstat: float
    alpha_pvalue: float
    alpha_significant: bool
    betas: Dict[str, float]
    tstats: Dict[str, float]
    r_squared: float
    n_obs: int
    market_excess: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    def summary(self) -> List[str]:
        """Compact lines suitable for appending to a tearsheet."""
        sig = "YES (|t|>2)" if self.alpha_significant else "no"
        lines = [
            "FAMA-FRENCH 3-FACTOR DECOMPOSITION",
            f" Alpha (ann.)  {self.alpha_annual:+.2%}   t={self.alpha_tstat:+.2f}   sig: {sig}",
            f" Beta  MKT={self.betas.get('MKT', float('nan')):+.2f} "
            f"SMB={self.betas.get('SMB', float('nan')):+.2f} "
            f"HML={self.betas.get('HML', float('nan')):+.2f}",
            f" R^2 = {self.r_squared:.3f}   (n={self.n_obs})",
        ]
        return lines


def _factor_returns(factor_close: pd.DataFrame, rf_daily: float,
                    factor_etfs: Dict[str, str]) -> Optional[pd.DataFrame]:
    """Build MKT/SMB/HML daily factor returns from ETF closes. None if missing."""
    need = list(dict.fromkeys(factor_etfs.values()))
    if not set(need).issubset(set(factor_close.columns)):
        return None
    rets = factor_close[need].pct_change()
    spy = factor_etfs["market"]
    iwm = factor_etfs["small"]
    vtv = factor_etfs["value"]
    vug = factor_etfs["growth"]
    return pd.DataFrame({
        "MKT": rets[spy] - rf_daily,    # market excess
        "SMB": rets[iwm] - rets[spy],   # small minus big (proxy)
        "HML": rets[vtv] - rets[vug],   # value minus growth (proxy)
    })


def factor_decomposition(
    strategy_returns: pd.Series,
    factor_close: pd.DataFrame,
    rf_annual: float = 0.0,
    periods: int = TRADING_DAYS_PER_YEAR,
    hac_lags: int = 5,
    factor_etfs: Optional[Dict[str, str]] = None,
) -> Optional[FactorDecompResult]:
    """Regress strategy excess returns on the 3 factors with HAC standard errors.

    Parameters
    ----------
    strategy_returns : pd.Series
        Daily strategy returns (the OOS stream).
    factor_close : pd.DataFrame
        Close prices including the factor ETFs (SPY, IWM, VTV, VUG).
    rf_annual : float
        Annual risk-free rate (converted to daily internally).
    periods : int
        Periods per year for annualising alpha.
    hac_lags : int
        Newey-West lag length for the robust covariance.
    factor_etfs : dict, optional
        Override the default ETF->role mapping.

    Returns
    -------
    FactorDecompResult or None
        None if the required factor ETFs are not present (e.g. not loaded).
    """
    factor_etfs = factor_etfs or FACTOR_ETFS
    rf_daily = rf_annual / periods

    factors = _factor_returns(factor_close, rf_daily, factor_etfs)
    if factors is None:
        return None

    y = (strategy_returns - rf_daily).rename("excess")
    data = pd.concat([y, factors], axis=1).dropna()
    if len(data) < 30:  # too few points for a meaningful regression
        return None

    X = sm.add_constant(data[["MKT", "SMB", "HML"]])
    model = sm.OLS(data["excess"], X).fit(
        cov_type="HAC", cov_kwds={"maxlags": hac_lags}
    )

    alpha_daily = float(model.params["const"])
    alpha_t = float(model.tvalues["const"])
    return FactorDecompResult(
        alpha_daily=alpha_daily,
        alpha_annual=alpha_daily * periods,
        alpha_tstat=alpha_t,
        alpha_pvalue=float(model.pvalues["const"]),
        alpha_significant=abs(alpha_t) > 2.0,
        betas={k: float(model.params[k]) for k in ("MKT", "SMB", "HML")},
        tstats={k: float(model.tvalues[k]) for k in ("MKT", "SMB", "HML")},
        r_squared=float(model.rsquared),
        n_obs=int(model.nobs),
        market_excess=data["MKT"],
    )
