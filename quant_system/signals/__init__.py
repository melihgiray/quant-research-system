"""Trading signals. Each returns *target weights* (date x ticker) for the engine.

Convention: a signal may use data up to and including each row's date. The
backtest engine owns the one-day execution lag (``shift(1)``), so signals never
need their own - and must never add a second - lag.
"""

from .momentum import cross_sectional_momentum, time_series_momentum
from .mean_reversion import (
    find_cointegrated_pair,
    scan_candidate_pairs,
    pairs_signal,
    single_asset_reversion,
)

__all__ = [
    "cross_sectional_momentum",
    "time_series_momentum",
    "find_cointegrated_pair",
    "scan_candidate_pairs",
    "pairs_signal",
    "single_asset_reversion",
]
