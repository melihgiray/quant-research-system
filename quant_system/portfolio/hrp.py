"""Hierarchical Risk Parity (Lopez de Prado, 2016).

Inverse-variance weighting treats every asset as if it lived on its own: it
ignores that two near-identical assets should share one asset's worth of risk
budget, not two. HRP fixes that without inverting a covariance matrix (which is
what makes mean-variance weights blow up on noisy estimates). It does three
things: build a tree from the correlation structure, reorder the assets so
similar ones sit next to each other, then split the risk budget down that tree
so tightly-correlated clusters are treated as one before their members compete.

The pieces are separated so each is testable on its own: the correlation
distance, the clustering order, the inverse-variance building block, and the
recursive bisection that ties them together.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform


def correlation_distance(corr: pd.DataFrame) -> pd.DataFrame:
    """Lopez de Prado's correlation distance, d_ij = sqrt(0.5 * (1 - corr_ij)).

    Maps correlation in [-1, 1] to a proper distance in [0, 1]: identical assets
    (corr 1) sit at distance 0, uncorrelated at ~0.71, perfectly opposed at 1.
    """
    d = np.sqrt(0.5 * (1.0 - corr.to_numpy()))
    np.fill_diagonal(d, 0.0)                              # kill tiny float noise on the diagonal
    return pd.DataFrame(d, index=corr.index, columns=corr.columns)


def cluster_order(corr: pd.DataFrame, method: str = "single") -> List[str]:
    """Quasi-diagonalisation order: leaves of the correlation dendrogram.

    Hierarchically cluster the assets on their correlation distance, then read
    the leaves left to right. Similar assets end up adjacent, so reordering the
    covariance matrix by this list pushes the large entries toward the diagonal.
    """
    dist = correlation_distance(corr)
    condensed = squareform(dist.to_numpy(), checks=False)
    link = linkage(condensed, method=method)
    order = leaves_list(link)
    return [corr.index[i] for i in order]


def inverse_variance_weights(cov: pd.DataFrame) -> pd.Series:
    """Inverse-variance (naive risk parity) weights: w_i proportional to 1/var_i.

    The baseline HRP is measured against. It ignores correlation entirely, so two
    near-duplicate assets each get their full 1/var share and the pair ends up
    over-weighted.
    """
    ivar = 1.0 / np.diag(cov.to_numpy())
    w = ivar / ivar.sum()
    return pd.Series(w, index=cov.index)


def _cluster_variance(cov: pd.DataFrame, items: List[str]) -> float:
    """Variance of the inverse-variance sub-portfolio over ``items``."""
    sub = cov.loc[items, items]
    w = inverse_variance_weights(sub).to_numpy()
    return float(w @ sub.to_numpy() @ w)


def recursive_bisection(cov: pd.DataFrame, order: List[str]) -> pd.Series:
    """Split the risk budget down the clustered order (Lopez de Prado, 2016).

    Walk the ordered assets top down. At each step split the current block in
    half and give each half a share of its parent's budget inversely to the
    half's inverse-variance-portfolio variance, so the riskier half is trimmed.
    Recurse until every block is a single asset. Weights are long-only and sum
    to 1.
    """
    weights = pd.Series(1.0, index=order)
    clusters = [order]
    while clusters:
        clusters = [block[half:stop]
                    for block in clusters
                    for half, stop in ((0, len(block) // 2), (len(block) // 2, len(block)))
                    if len(block) > 1]
        for i in range(0, len(clusters), 2):
            left, right = clusters[i], clusters[i + 1]
            var_left = _cluster_variance(cov, left)
            var_right = _cluster_variance(cov, right)
            alpha = 1.0 - var_left / (var_left + var_right)      # more budget to the calmer half
            weights[left] *= alpha
            weights[right] *= 1.0 - alpha
    return weights


def hrp_weights(returns: Optional[pd.DataFrame] = None,
                *,
                cov: Optional[pd.DataFrame] = None,
                corr: Optional[pd.DataFrame] = None,
                method: str = "single") -> pd.Series:
    """Full Hierarchical Risk Parity weights.

    Pass a returns frame (cov and corr are estimated from it), or pass ``cov``
    and ``corr`` directly. Clusters on the correlation distance, reorders so
    similar assets are adjacent, then splits the risk budget by recursive
    bisection. Returns long-only weights summing to 1, indexed like ``cov``.
    """
    if cov is None or corr is None:
        if returns is None:
            raise ValueError("pass returns, or both cov and corr")
        cov = returns.cov()
        corr = returns.corr()
    order = cluster_order(corr, method=method)
    return recursive_bisection(cov, order).reindex(cov.index)
