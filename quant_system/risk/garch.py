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
