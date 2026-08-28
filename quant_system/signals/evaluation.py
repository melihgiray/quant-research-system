"""Signal-quality evaluation."""

from __future__ import annotations

import numpy as np


def rank_ic(signal, forward_returns) -> float:
    """Rank information coefficient: Spearman correlation of a signal with the
    returns that follow it.

    The standard read on whether a signal has predictive power: +1 means it ranks
    outcomes perfectly, 0 means no rank relationship, negative means it points the
    wrong way. Rank-based, so it is robust to outliers and monotone transforms."""
    from scipy.stats import spearmanr
    s = np.asarray(signal, dtype=float)
    f = np.asarray(forward_returns, dtype=float)
    mask = np.isfinite(s) & np.isfinite(f)
    if mask.sum() < 3:
        return float("nan")
    corr = spearmanr(s[mask], f[mask]).correlation
    return float(corr) if np.isfinite(corr) else float("nan")
