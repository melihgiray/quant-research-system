"""Market-regime detection.

There are two detectors here, and they are used for different things.

``vol_ratio_regime`` is causal. It calls a defensive regime when short-horizon
realised vol (21d) runs hot relative to the long-horizon baseline (252d). It only
uses trailing windows and is lagged a day, so it is safe to size positions with.
This is the one the switcher uses.

``hmm_regime`` fits a 2-state Gaussian HMM on daily returns. It reads nicely on a
chart, but ``predict`` runs Viterbi over the whole series, so it peeks at the
future and its labels are not causal. For that reason the HMM only feeds the
overlay and the regime stats. Sizing stays on the vol-ratio.

Both label 1 = defensive/high-vol, 0 = risk-on/low-vol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..config import RegimeConfig, TRADING_DAYS_PER_YEAR


@dataclass
class RegimeResult:
    """Regime labels plus the metadata needed to explain/plot them."""

    labels: pd.Series          # display labels (HMM if used, else vol-ratio)
    causal_labels: pd.Series   # always the causal vol-ratio labels (for sizing)
    method: str                # "hmm" or "vol_ratio"
    details: Dict[str, object] = field(default_factory=dict)

    @property
    def defensive_fraction(self) -> float:
        return float((self.labels == 1).mean())


def market_proxy_returns(price_data, benchmark: Optional[str] = None) -> pd.Series:
    """Daily return of the 'market' the regime is measured on.

    Uses ``benchmark`` (e.g. 'SPY') if present, else the equal-weight average of
    the panel - a reasonable broad-market proxy when no index is loaded.
    """
    rets = price_data.returns()
    if benchmark and benchmark in rets.columns:
        return rets[benchmark].rename("market")
    return rets.mean(axis=1).rename("market")


def vol_ratio_regime(price_data, cfg: RegimeConfig = None,
                     benchmark: Optional[str] = None) -> pd.Series:
    """Causal high/low-vol regime from a fast/slow realised-vol ratio.

    regime = 1 (defensive) when std_21 / std_252 > high_vol_ratio, else 0.
    Lagged one day so day-T sizing uses only volatility known through T-1.
    """
    cfg = cfg or RegimeConfig()
    mkt = market_proxy_returns(price_data, benchmark)
    fast = mkt.rolling(cfg.fast_vol, min_periods=cfg.fast_vol // 2).std()
    slow = mkt.rolling(cfg.slow_vol, min_periods=cfg.slow_vol // 2).std()
    ratio = fast / slow
    labels = (ratio > cfg.high_vol_ratio).astype(float)
    labels[ratio.isna()] = np.nan
    return labels.shift(1).rename("regime_vol_ratio")


def garch_regime_labels(returns: pd.Series,
                        high_vol_ratio: float = 1.30,
                        baseline_window: int = TRADING_DAYS_PER_YEAR,
                        refit_every: int = 21,
                        min_obs: int = TRADING_DAYS_PER_YEAR) -> pd.Series:
    """Causal high/low-vol regime from a GARCH forecast running hot vs its baseline.

    regime = 1 (defensive) when the one-step GARCH volatility forecast exceeds
    ``high_vol_ratio`` times its own trailing-median forecast, else 0. The GARCH
    series is already a forecast made from data before day t, so the label is
    causal without the extra one-day shift the realised-vol version needs. The
    ``arch`` extra is required.
    """
    from ..risk.garch import garch_vol_series          # lazy: pulls in the arch extra

    gvol = garch_vol_series(returns, refit_every=refit_every, min_obs=min_obs)
    baseline = gvol.rolling(baseline_window, min_periods=baseline_window // 4).median()
    ratio = gvol / baseline
    labels = (ratio > high_vol_ratio).astype(float)
    labels[ratio.isna()] = np.nan
    return labels.rename("regime_garch")


def garch_regime(price_data, cfg: RegimeConfig = None,
                 benchmark: Optional[str] = None) -> pd.Series:
    """GARCH-forecast regime on the market proxy: a third, causal regime definition."""
    cfg = cfg or RegimeConfig()
    mkt = market_proxy_returns(price_data, benchmark)
    return garch_regime_labels(mkt, high_vol_ratio=cfg.high_vol_ratio)


def hmm_regime(price_data, cfg: RegimeConfig = None, seed: int = 7,
               benchmark: Optional[str] = None):
    """Fit a 2-state Gaussian HMM on daily returns; label the high-variance state 1.

    Returns (labels, details). Falls back to (None, {...}) if hmmlearn is not
    installed so callers can degrade to the vol-ratio detector.

    NOTE: these labels are smoothed (non-causal) and intended for analysis/plots,
    not for live position sizing - see the module docstring.
    """
    cfg = cfg or RegimeConfig()
    try:
        from hmmlearn.hmm import GaussianHMM
    except Exception as exc:  # not installed / failed to import
        return None, {"error": f"hmmlearn unavailable: {exc}"}

    mkt = market_proxy_returns(price_data, benchmark).dropna()
    X = mkt.values.reshape(-1, 1)
    model = GaussianHMM(n_components=2, covariance_type="full",
                        n_iter=200, random_state=seed)
    try:
        model.fit(X)
        states = model.predict(X)
    except Exception as exc:
        return None, {"error": f"HMM fit failed: {exc}"}

    variances = model.covars_.reshape(2)
    means = model.means_.reshape(2)
    high_vol_state = int(np.argmax(variances))           # defensive = noisier state
    labels = pd.Series((states == high_vol_state).astype(float), index=mkt.index,
                       name="regime_hmm")
    details = {
        "state_means": {int(s): float(means[s]) for s in range(2)},
        "state_daily_vol": {int(s): float(np.sqrt(variances[s])) for s in range(2)},
        "high_vol_state": high_vol_state,
        "transition_matrix": model.transmat_.tolist(),
    }
    return labels, details


def detect_regime(price_data, cfg: RegimeConfig = None,
                  seed: int = 7, benchmark: Optional[str] = None) -> RegimeResult:
    """Run the configured detector and return labels + causal labels + metadata.

    The causal vol-ratio labels are always computed (and used by the switcher for
    sizing). If ``cfg.use_hmm`` and hmmlearn is available, the HMM labels become
    the *display* labels for the overlay; otherwise display == causal.
    """
    cfg = cfg or RegimeConfig()
    causal = vol_ratio_regime(price_data, cfg, benchmark)

    if cfg.use_hmm:
        hmm_labels, details = hmm_regime(price_data, cfg, seed, benchmark)
        if hmm_labels is not None:
            return RegimeResult(
                labels=hmm_labels.reindex(causal.index),
                causal_labels=causal,
                method="hmm",
                details=details,
            )

    return RegimeResult(
        labels=causal.rename("regime"),
        causal_labels=causal,
        method="vol_ratio",
        details={"high_vol_ratio": cfg.high_vol_ratio,
                 "fast_vol": cfg.fast_vol, "slow_vol": cfg.slow_vol},
    )
