"""Regime-aware position routing.

Two ways to use a regime:

* ``apply_regime_sizing`` - keep the same strategy but cut gross exposure in the
  defensive regime (de-risk when vol is high). This is the conservative default
  and answers the "what happens when volatility spikes?" question directly:
  positions are scaled by ``defensive_scale`` (e.g. halved).

* ``switch_strategies`` - actually route to a different strategy per regime
  (e.g. momentum in calm markets, mean-reversion or cash when stressed). More
  aggressive; provided for completeness.

Both consume the CAUSAL regime labels (already lagged in the detector), so no
future information leaks into today's book.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd


def apply_regime_sizing(weights: pd.DataFrame, regime_labels: pd.Series,
                        defensive_scale: float = 0.5) -> pd.DataFrame:
    """Scale gross exposure down in the defensive (label==1) regime.

    Parameters
    ----------
    weights : pd.DataFrame
        Target weights (date x ticker).
    regime_labels : pd.Series
        Causal 0/1 labels (1 = defensive). Aligned/ffilled to the weight index.
    defensive_scale : float
        Multiplier applied on defensive days (0.5 = halve risk). Risk-on days are
        left at 1.0.

    Returns
    -------
    pd.DataFrame
        Regime-scaled weights.
    """
    reg = regime_labels.reindex(weights.index).ffill().fillna(0.0)
    scale = reg.map(lambda v: defensive_scale if v == 1 else 1.0)
    return weights.mul(scale, axis=0)


def switch_strategies(regime_labels: pd.Series,
                      strategy_weights: Dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Select each day's book from a per-regime dictionary of weight matrices.

    Parameters
    ----------
    regime_labels : pd.Series
        Causal 0/1 labels.
    strategy_weights : dict[int, pd.DataFrame]
        Maps regime label -> weight matrix to use in that regime. Missing regimes
        default to flat (cash).

    Returns
    -------
    pd.DataFrame
        Daily weights assembled by routing each date to its regime's strategy.
    """
    if not strategy_weights:
        raise ValueError("strategy_weights must contain at least one regime")
    template = next(iter(strategy_weights.values()))
    reg = regime_labels.reindex(template.index).ffill().fillna(0.0)

    out = pd.DataFrame(0.0, index=template.index, columns=template.columns)
    for label, w in strategy_weights.items():
        mask = (reg == label)
        aligned = w.reindex(index=template.index, columns=template.columns).ffill().fillna(0.0)
        out = out.where(~mask, aligned)
    return out
