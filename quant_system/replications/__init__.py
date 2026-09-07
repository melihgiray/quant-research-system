"""Reproducible implementations of published quantitative-finance studies.

Each module states where it follows a paper and where practical data or
out-of-sample constraints require a documented deviation.
"""

from .volatility_managed import (
    VolatilityManagedResult,
    KenFrenchDataset,
    download_ken_french_daily,
    download_ken_french_daily_with_metadata,
    fold_statistics,
    subperiod_statistics,
    inverse_variance_exposure,
    walk_forward_volatility_managed,
)

__all__ = [
    "VolatilityManagedResult",
    "KenFrenchDataset",
    "download_ken_french_daily",
    "download_ken_french_daily_with_metadata",
    "fold_statistics",
    "subperiod_statistics",
    "inverse_variance_exposure",
    "walk_forward_volatility_managed",
]
