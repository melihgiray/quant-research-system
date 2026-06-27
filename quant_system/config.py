"""Central configuration.

Every tunable number in the system lives here so that nothing important is
buried as a magic constant deep in the code. The config is an *immutable*
dataclass: the CLI builds one instance and threads it through the call graph.
That keeps us honest about the "no global mutable state" rule — code reads
parameters from a value that was passed in, never from module-level globals.

All windows are expressed in *trading days* (~252/year, ~21/month, ~63/quarter).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List


# Trading-day calendar constants (approximate, the market convention).
TRADING_DAYS_PER_YEAR = 252
TRADING_DAYS_PER_MONTH = 21
TRADING_DAYS_PER_QUARTER = 63


@dataclass(frozen=True)
class CostConfig:
    """Transaction-cost assumptions.

    Costs have two components, mirroring how a real desk thinks about them:

    half_spread_bps:
        Half the bid/ask spread, paid on every share traded regardless of size.
        1-2 bps is realistic for liquid large-caps / ETFs. Expressed in basis
        points of traded notional (1 bp = 0.01%).
    impact_eta:
        The dimensionless constant in the square-root market-impact model
        ``impact = eta * sigma * sqrt(Q / V)``. Empirically eta is O(0.1) for
        equities (Almgren et al. 2005), hence the default.
    capital:
        Notional portfolio size in dollars. Needed to convert a *weight change*
        into a *share count* so the participation rate Q/V is meaningful.
    adv_lookback:
        Window (days) used to estimate Average Daily Volume V.
    vol_lookback:
        Window (days) used to estimate the daily volatility sigma that scales
        market impact.
    """

    half_spread_bps: float = 1.5
    impact_eta: float = 0.1
    capital: float = 1_000_000.0
    adv_lookback: int = TRADING_DAYS_PER_MONTH
    vol_lookback: int = TRADING_DAYS_PER_MONTH
    # Floor for the daily-vol input to market impact. When the rolling estimate is
    # unavailable (e.g. an establishment trade before the window warms up) we use
    # this instead of 0, so impact is never silently understated. ~1.5%/day ~ 24%/yr.
    default_daily_vol: float = 0.015


@dataclass(frozen=True)
class WalkForwardConfig:
    """Walk-forward validation windows (expanding by default).

    in_sample:   initial training/look-back window (1 year).
    out_sample:  out-of-sample test window per fold (1 quarter).
    step:        how far the window advances each fold (1 quarter => no overlap
                 between consecutive OOS segments, so concatenating them yields a
                 clean, non-overlapping OOS equity curve).
    expanding:   True  -> in-sample grows each fold (anchored start).
                 False -> rolling fixed-length in-sample window.
    """

    in_sample: int = TRADING_DAYS_PER_YEAR
    out_sample: int = TRADING_DAYS_PER_QUARTER
    step: int = TRADING_DAYS_PER_QUARTER
    expanding: bool = True


@dataclass(frozen=True)
class MomentumConfig:
    """Cross-sectional momentum (Jegadeesh-Titman 1993).

    lookback:   formation window (12 months ~ 252 days).
    skip:       most-recent days skipped to avoid 1-month reversal (the "12-1").
    quantile:   fraction in each leg (0.2 => long top quintile, short bottom).
    rebalance:  trading days between rebalances (21 ~ monthly).
    """

    lookback: int = TRADING_DAYS_PER_YEAR
    skip: int = TRADING_DAYS_PER_MONTH
    quantile: float = 0.2
    rebalance: int = TRADING_DAYS_PER_MONTH


@dataclass(frozen=True)
class PairsConfig:
    """Statistical pairs trading (Engle-Granger cointegration).

    zscore_lookback:  window for the rolling spread z-score.
    entry_z:          |z| at which we open a position.
    exit_z:           |z| at which we close (revert to flat).
    coint_pvalue_max: if the rolling Engle-Granger p-value exceeds this, the
                      relationship is deemed broken and we stop trading.
    coint_lookback:   window over which cointegration is re-tested.
    """

    zscore_lookback: int = 60
    entry_z: float = 2.0
    exit_z: float = 0.0
    coint_pvalue_max: float = 0.10
    coint_lookback: int = TRADING_DAYS_PER_YEAR


@dataclass(frozen=True)
class MLConfig:
    """ML directional signal (gradient boosting on engineered features).

    train_window:   days of history used to fit each model (2 years).
    momentum_spans: lookbacks for the momentum features.
    vol_span:       window for realised-vol / z-score features.
    rsi_span:       RSI window.
    prob_deadband:  predicted P(up) within +/- this of 0.5 => no position
                    (avoid trading on coin-flip predictions).
    """

    train_window: int = 2 * TRADING_DAYS_PER_YEAR
    momentum_spans: tuple = (5, 10, 21)
    vol_span: int = TRADING_DAYS_PER_MONTH
    rsi_span: int = 14
    prob_deadband: float = 0.05


@dataclass(frozen=True)
class RegimeConfig:
    """Volatility-regime detector.

    fast_vol:        short realised-vol window (~1 month).
    slow_vol:        long realised-vol window (~1 year baseline).
    high_vol_ratio:  fast/slow vol above this => "defensive" (high-vol) regime.
    defensive_scale: position-size multiplier applied in the defensive regime.
    use_hmm:         if True and hmmlearn is installed, use a 2-state Gaussian
                     HMM on returns instead of the vol-ratio rule.
    """

    fast_vol: int = TRADING_DAYS_PER_MONTH
    slow_vol: int = TRADING_DAYS_PER_YEAR
    high_vol_ratio: float = 1.30
    defensive_scale: float = 0.50
    use_hmm: bool = True


@dataclass(frozen=True)
class RiskConfig:
    """Portfolio risk controls and sizing.

    target_vol:      annualised volatility target for vol-targeting sizing.
    max_drawdown:    hard stop — if breached, the engine flattens to cash.
    max_weight:      per-name concentration cap (absolute weight).
    kelly_fraction:  fraction of full-Kelly to use (full Kelly is too aggressive).
    """

    target_vol: float = 0.10
    max_drawdown: float = 0.20
    max_weight: float = 0.10
    kelly_fraction: float = 0.50


@dataclass(frozen=True)
class Config:
    """Top-level configuration object threaded through the system."""

    start: str = "2015-01-01"          # data start (inclusive)
    end: str = "2024-12-31"            # data end (inclusive)
    risk_free_rate: float = 0.0        # annualised; 0 is the honest default if FRED is unavailable
    cache_dir: str = "data_cache"      # on-disk price cache
    reports_dir: str = "reports"       # where plots are written
    use_synthetic: bool = False        # force the offline synthetic data generator
    random_seed: int = 7               # reproducibility for synthetic data / any sampling

    cost: CostConfig = field(default_factory=CostConfig)
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    momentum: MomentumConfig = field(default_factory=MomentumConfig)
    pairs: PairsConfig = field(default_factory=PairsConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)

    def with_(self, **changes) -> "Config":
        """Return a copy of this config with top-level fields overridden.

        Keeps the object immutable while still being ergonomic for the CLI,
        e.g. ``cfg.with_(use_synthetic=True)``.
        """
        return replace(self, **changes)


def default_config() -> Config:
    """Build the default configuration. The single entry point used by the CLI."""
    return Config()
