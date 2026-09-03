"""Reproducible implementations of published quantitative-finance studies.

Each module states where it follows a paper and where practical data or
out-of-sample constraints require a documented deviation.
"""

from .volatility_managed import (
    VolatilityManagedResult,
    download_ken_french_daily,
    inverse_variance_exposure,
    walk_forward_volatility_managed,
)

__all__ = [
    "VolatilityManagedResult",
    "download_ken_french_daily",
    "inverse_variance_exposure",
    "walk_forward_volatility_managed",
]
