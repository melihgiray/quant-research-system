"""Shared building blocks for the research scripts.

The ``scripts/build_*_results.py`` runners all assemble the same thing: the
standard ticker universe, and the three canonical strategy sleeves as
walk-forward weight callbacks. That setup was copy-pasted across seven scripts;
this module is the single definition they share, so a change to what "the three
sleeves" means happens in one place.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import pandas as pd

from .config import Config
from .data.universe import FACTOR_ETFS, PAIRS_CANDIDATES, universe
from .regime.switcher import apply_regime_sizing
from .signals.mean_reversion import causal_pairs_weights
from .signals.momentum import cross_sectional_momentum
from .signals.ml_signal import train_predict

MakeWeights = Callable[[object, object], pd.DataFrame]


def research_universe() -> List[str]:
    """The standard research ticker set: sectors, factor ETFs, large caps and pairs.

    Sorted and de-duplicated, so a panel loaded from it has a stable column order.
    """
    return sorted(set(
        universe("all")
        + list(dict.fromkeys(FACTOR_ETFS.values()))
        + [t for pair in PAIRS_CANDIDATES for t in pair]
    ))


def sleeve_makers(cfg: Config, regime_labels: Optional[pd.Series] = None,
                  defensive_scale: Optional[float] = None) -> Dict[str, MakeWeights]:
    """The three canonical sleeves as ``make(price_data, fit_end)`` callbacks.

    Momentum and the ML directional signal are scaled down in the defensive
    regime when ``regime_labels`` is given; the pairs sleeve is already
    market-neutral and is left unscaled. Passing ``regime_labels=None`` returns
    the unscaled sleeves (useful for a no-regime baseline).
    """
    scale = cfg.regime.defensive_scale if defensive_scale is None else defensive_scale

    def _sized(weights: pd.DataFrame) -> pd.DataFrame:
        if regime_labels is None:
            return weights
        return apply_regime_sizing(weights, regime_labels, scale)

    def momentum(price_data, fit_end):
        return _sized(cross_sectional_momentum(price_data, cfg.momentum))

    def pairs(price_data, fit_end):
        return causal_pairs_weights(price_data, fit_end, PAIRS_CANDIDATES, cfg.pairs)

    def ml(price_data, fit_end):
        return _sized(train_predict(price_data, fit_end, cfg.ml,
                                    max_weight=cfg.risk.max_weight))

    return {"momentum": momentum, "pairs": pairs, "ml": ml}
