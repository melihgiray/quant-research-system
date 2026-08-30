"""Risk-budgeting allocators: equal risk contribution and maximum diversification.

Inverse-volatility weighting equalises risk only when assets are uncorrelated.
Equal risk contribution (ERC) solves for weights whose *risk contributions*
``w_i * (Sigma w)_i`` are equal even under correlation, the honest form of "risk
parity". Maximum diversification instead maximises the ratio of the weighted
average volatility to the portfolio volatility, tilting toward assets that
diversify rather than merely toward low-vol ones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _as_matrix(cov):
    if isinstance(cov, pd.DataFrame):
        return cov.to_numpy(dtype=float), list(cov.columns)
    return np.asarray(cov, dtype=float), None


def _labelled(weights: np.ndarray, labels):
    return pd.Series(weights, index=labels) if labels is not None else weights


def erc_weights(cov):
    """Long-only weights whose risk contributions are equal (risk parity).

    Minimises the dispersion of the per-asset risk contributions subject to being
    long-only and fully invested. On a diagonal covariance this reduces to
    inverse-volatility weighting.
    """
    matrix, labels = _as_matrix(cov)
    n = len(matrix)

    def dispersion(w):
        marginal = matrix @ w
        rc = w * marginal
        total = rc.sum()
        if total <= 0:
            return 1.0
        share = rc / total                                # scale-free: shares sum to 1
        return float(np.sum((share - 1.0 / n) ** 2))

    result = minimize(dispersion, np.full(n, 1.0 / n), method="SLSQP",
                      bounds=[(1e-6, 1.0)] * n,
                      constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
                      options={"ftol": 1e-12, "maxiter": 1000})
    weights = np.clip(result.x, 0.0, None)
    weights = weights / weights.sum()
    return _labelled(weights, labels)


def risk_contributions(cov, weights) -> np.ndarray:
    """Per-asset share of total portfolio variance under ``weights`` (sums to 1)."""
    matrix, _ = _as_matrix(cov)
    w = np.asarray(weights, dtype=float)
    marginal = matrix @ w
    rc = w * marginal
    total = rc.sum()
    return rc / total if total != 0 else rc


def max_diversification_weights(cov):
    """Long-only weights maximising the diversification ratio.

    The diversification ratio is the weighted average asset volatility over the
    portfolio volatility; maximising it favours assets that hedge each other, not
    just quiet ones. Reduces to putting everything in one asset only if that asset
    dominates the diversification.
    """
    matrix, labels = _as_matrix(cov)
    vols = np.sqrt(np.diag(matrix))
    n = len(matrix)

    def neg_diversification(w):
        port_vol = np.sqrt(w @ matrix @ w)
        if port_vol == 0:
            return 0.0
        return -float((w @ vols) / port_vol)

    result = minimize(neg_diversification, np.full(n, 1.0 / n), method="SLSQP",
                      bounds=[(0.0, 1.0)] * n,
                      constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}])
    weights = np.clip(result.x, 0.0, None)
    weights = weights / weights.sum()
    return _labelled(weights, labels)


def minimum_variance_weights(cov):
    """Long-only fully invested weights with minimum estimated variance.

    This is a baseline for HRP and ERC. It makes no return forecast: it asks
    only which feasible mix has the lowest risk under the supplied covariance.
    """
    matrix, labels = _as_matrix(cov)
    n = len(matrix)
    result = minimize(lambda w: float(w @ matrix @ w), np.full(n, 1.0 / n),
                      method="SLSQP", bounds=[(0.0, 1.0)] * n,
                      constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
                      options={"ftol": 1e-12, "maxiter": 1_000})
    if not result.success:
        raise RuntimeError(f"minimum-variance optimisation failed: {result.message}")
    weights = np.clip(result.x, 0.0, None)
    return _labelled(weights / weights.sum(), labels)
