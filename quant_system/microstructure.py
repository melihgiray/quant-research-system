"""Microstructure estimators from daily close and volume.

Effective spreads and price impact are usually measured from tick data, but
useful proxies can be recovered from daily bars alone. These estimators need only
the series this project already carries, so they can annotate liquidity without a
tick feed.
"""

from __future__ import annotations

import numpy as np


def roll_spread(prices) -> float:
    """Roll's (1984) effective spread from the serial covariance of price changes.

    Bid-ask bounce makes consecutive price changes negatively autocovaried; under
    Roll's model that first-order autocovariance is ``-s**2 / 4``, so the implied
    spread is ``2 * sqrt(-cov)``. Returned in the price units of ``prices``; the
    model only applies when the autocovariance is negative, otherwise 0.
    """
    changes = np.diff(np.asarray(prices, dtype=float))
    if len(changes) < 3:
        return float("nan")
    cov = np.cov(changes[:-1], changes[1:])[0, 1]
    return float(2.0 * np.sqrt(-cov)) if cov < 0 else 0.0
