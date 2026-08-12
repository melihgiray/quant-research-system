"""GARCH(1,1) conditional-volatility forecasts for sizing and regime detection.

Trailing realised volatility looks backward: it tells you how bumpy the last N
days were, and reacts slowly. A GARCH(1,1) model instead forecasts *tomorrow's*
variance from today's, capturing the volatility clustering that realised vol lags
behind (a shock today raises the forecast immediately, then decays). Used as a
sizing input it de-risks faster going into turbulence and re-risks faster coming
out.

``arch`` is an optional dependency (the ``garch`` extra). It is imported lazily so
the rest of the package works without it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import TRADING_DAYS_PER_YEAR


def garch_forecast_vol(returns,
                       p: int = 1,
                       q: int = 1,
                       scale: float = 100.0,
                       min_obs: int = 100) -> float:
    """One-step-ahead conditional volatility from a GARCH(p, q) fit.

    Fits on the whole series passed in and forecasts the next period's
    volatility, in the same units as ``returns`` (daily simple returns in, daily
    vol out). Returns are rescaled by ``scale`` for the fit, which keeps the
    optimiser numerically well behaved on tiny daily numbers, then the forecast
    is scaled back.
    """
    r = pd.Series(returns).dropna()
    if len(r) < min_obs:
        raise ValueError(f"need at least {min_obs} observations, got {len(r)}")

    from arch import arch_model                     # lazy: optional dependency
    model = arch_model(r.to_numpy() * scale, vol="GARCH", p=p, q=q,
                       mean="Constant", dist="normal", rescale=False)
    res = model.fit(disp="off")
    forecast = res.forecast(horizon=1, reindex=False)
    variance = float(forecast.variance.to_numpy()[-1, 0])
    return float(np.sqrt(variance) / scale)


def garch_vol_series(returns,
                     refit_every: int = 21,
                     min_obs: int = 252,
                     scale: float = 100.0,
                     p: int = 1,
                     q: int = 1) -> pd.Series:
    """A causal one-step conditional-vol forecast for each day of a return path.

    Refitting GARCH every day is expensive, so the parameters are refit on
    expanding history every ``refit_every`` days and the last one-step forecast
    is carried forward between refits. Every value is strictly causal: the
    forecast placed on day t is fit on returns before t, so this can be used to
    size a position on day t without look-ahead. The first ``min_obs`` days are
    NaN (not enough history to fit).
    """
    r = pd.Series(returns).dropna()
    out = pd.Series(np.nan, index=r.index)
    last = np.nan
    for pos in range(len(r)):
        if pos < min_obs:
            continue
        if (pos - min_obs) % refit_every == 0:
            try:
                last = garch_forecast_vol(r.iloc[:pos], p=p, q=q,
                                          scale=scale, min_obs=min_obs)
            except Exception:
                pass                                   # keep the previous forecast on a fit failure
        out.iloc[pos] = last
    return out.reindex(pd.Series(returns).index)


def garch_vol_target(returns,
                     target_vol: float = 0.10,
                     max_leverage: float = 3.0,
                     refit_every: int = 21,
                     min_obs: int = 252,
                     scale: float = 100.0) -> pd.Series:
    """Scale a return stream toward ``target_vol`` using the GARCH forecast.

    The GARCH analogue of ``portfolio.volatility_target``. Because the GARCH
    series is a one-step-ahead *forecast* of day t's volatility, the scaler for
    day t is already known before day t's return is earned, so no extra one-day
    lag is applied here (the realised-vol version needs one because it looks
    backward). The multiplier is capped at ``max_leverage``.
    """
    r = pd.Series(returns)
    gvol = garch_vol_series(r, refit_every=refit_every, min_obs=min_obs, scale=scale)
    annualised = gvol * np.sqrt(TRADING_DAYS_PER_YEAR)
    scaler = (target_vol / annualised).replace([np.inf, -np.inf], np.nan)
    scaler = scaler.clip(upper=max_leverage)
    return r.mul(scaler)
