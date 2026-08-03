"""Trading signals. Each returns *target weights* (date x ticker) for the engine.

Convention: a signal may use data up to and including each row's date. The
backtest engine owns the one-day execution lag (``shift(1)``), so signals never
need their own - and must never add a second - lag.
"""

from .momentum import cross_sectional_momentum, time_series_momentum
from .mean_reversion import (
    find_cointegrated_pair,
    scan_candidate_pairs,
    causal_pairs_weights,
    pairs_signal,
    single_asset_reversion,
)
from .cv import PurgedKFold, purged_cv_scores, pooled_frame
from .ml_signal import permutation_importance_pvalues, ml_feature_significance
from .feature_selection import fdr_control_features, feature_fdr_summary

__all__ = [
    "cross_sectional_momentum",
    "time_series_momentum",
    "find_cointegrated_pair",
    "scan_candidate_pairs",
    "causal_pairs_weights",
    "pairs_signal",
    "single_asset_reversion",
    "PurgedKFold",
    "purged_cv_scores",
    "pooled_frame",
    "permutation_importance_pvalues",
    "ml_feature_significance",
    "fdr_control_features",
    "feature_fdr_summary",
]
