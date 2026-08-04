"""Combine strategy sleeves into one book."""

from .allocator import (
    blend_returns,
    combine_weights,
    inverse_vol_allocations,
    volatility_target,
)

__all__ = [
    "inverse_vol_allocations",
    "combine_weights",
    "blend_returns",
    "volatility_target",
]
