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
