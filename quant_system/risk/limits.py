"""Portfolio-level risk controls: drawdown stop and concentration limits.

These operate on *outputs* (a realised return stream, or a weight matrix) so they
compose cleanly with the engine without entangling its core P&L math.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def apply_drawdown_stop(
    returns: pd.Series,
    max_drawdown: float = 0.20,
    cooldown: int = 21,
) -> pd.Series:
    """Overlay a portfolio max-drawdown stop on a realised return stream.

    Rule: track equity from a running peak. The day the drawdown breaches
    ``-max_drawdown`` we cut risk to zero (returns become 0) for ``cooldown``
    trading days, then re-enter with the peak reset to the current equity (a
    fresh start, so we are not instantly re-stopped by the old high-water mark).

    This is a deliberately simple, defensible rule. In live trading you would add
    a smarter re-entry (e.g. wait for the signal's shadow equity to recover), but
    the parameterisation here makes the behaviour explicit and inspectable.

    Returns
    -------
    pd.Series
        The post-stop return stream (0 on days spent flat).
    """
    r = returns.fillna(0.0).values
    out = np.zeros_like(r)
    equity, peak = 1.0, 1.0
    in_market, cooldown_left = True, 0

    for i, ri in enumerate(r):
        if in_market:
            out[i] = ri
            equity *= (1.0 + ri)
            peak = max(peak, equity)
            if equity / peak - 1.0 <= -max_drawdown:
                in_market = False
                cooldown_left = cooldown
        else:
            out[i] = 0.0
            cooldown_left -= 1
            if cooldown_left <= 0:
                in_market = True
                peak = equity  # fresh high-water mark on re-entry
    return pd.Series(out, index=returns.index, name="returns_after_stop")


def cap_concentration(weights: pd.DataFrame, max_weight: float = 0.10) -> pd.DataFrame:
    """Clip every per-name absolute weight to ``max_weight`` (long or short).

    We clip rather than renormalise so the cap genuinely reduces gross exposure
    on over-concentrated days instead of just reshuffling it onto other names.
    """
    return weights.clip(lower=-max_weight, upper=max_weight)


def check_limits(weights: pd.DataFrame, max_weight: float = 0.10,
                 max_gross: float = 2.0) -> Dict[str, object]:
    """Report (not enforce) limit violations for diagnostics/logging.

    Returns a dict with the worst single-name weight, the worst gross exposure,
    and boolean breach flags. Useful to surface in a tearsheet without silently
    mutating the strategy.
    """
    abs_w = weights.abs()
    worst_name = float(abs_w.to_numpy().max()) if weights.size else 0.0
    gross = abs_w.sum(axis=1)
    worst_gross = float(gross.max()) if len(gross) else 0.0
    return {
        "worst_name_weight": worst_name,
        "worst_gross_exposure": worst_gross,
        "name_limit_breached": worst_name > max_weight + 1e-9,
        "gross_limit_breached": worst_gross > max_gross + 1e-9,
    }
