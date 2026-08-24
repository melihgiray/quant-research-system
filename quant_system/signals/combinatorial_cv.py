"""Combinatorial purged cross-validation (Lopez de Prado, ch. 12).

Ordinary k-fold leaves one test block at a time; combinatorial purged CV instead
partitions the data into N contiguous groups and tests every choice of k of them,
giving C(N, k) train/test splits and many more backtest paths from the same data.
Around each test group an embargo drops the neighbouring training samples, so
information does not leak across the boundary from overlapping labels.
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterator, Tuple

import numpy as np


def combinatorial_purged_splits(n_samples: int, n_groups: int = 6, k_test: int = 2,
                                embargo: int = 0) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) for every choice of ``k_test`` of ``n_groups``.

    The samples are partitioned into ``n_groups`` contiguous groups. For each
    combination of ``k_test`` groups, those samples are the test set and the rest
    are training, minus an ``embargo`` of samples on each side of every test group
    to purge boundary leakage. Yields C(n_groups, k_test) splits.
    """
    if not 1 <= k_test < n_groups:
        raise ValueError("need 1 <= k_test < n_groups")
    if n_samples < n_groups:
        raise ValueError("need at least as many samples as groups")

    bounds = np.linspace(0, n_samples, n_groups + 1).astype(int)
    groups = [np.arange(bounds[i], bounds[i + 1]) for i in range(n_groups)]

    for test_ids in combinations(range(n_groups), k_test):
        test_idx = np.concatenate([groups[i] for i in test_ids])
        blocked = set(test_idx.tolist())
        for i in test_ids:
            g = groups[i]
            for e in range(1, embargo + 1):
                if g[0] - e >= 0:
                    blocked.add(int(g[0] - e))
                if g[-1] + e < n_samples:
                    blocked.add(int(g[-1] + e))
        train_idx = np.array([j for j in range(n_samples) if j not in blocked], dtype=int)
        yield train_idx, np.sort(test_idx)
