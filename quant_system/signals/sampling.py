"""Sequential bootstrap for overlapping labels (Lopez de Prado, ch. 4).

Labels built from paths (triple-barrier) overlap in time: two events a day apart
share most of their outcome window, so a plain bootstrap draws near-duplicates
and overstates how much independent data there is. The sequential bootstrap draws
samples one at a time, each time favouring the labels that overlap least with what
has already been drawn, so the resampled set is closer to independent.

Overlap is described by an indicator matrix (bars x labels): a 1 where a label's
outcome window is open on that bar. Concurrency is how many labels are open at a
bar, and a label's uniqueness is the average of ``1 / concurrency`` over the bars
it spans.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def indicator_matrix(n_bars: int, spans: Sequence[Tuple[int, int]]) -> np.ndarray:
    """Bars x labels indicator: 1 where label j's window (inclusive) is open on bar i."""
    matrix = np.zeros((n_bars, len(spans)), dtype=float)
    for j, (start, end) in enumerate(spans):
        matrix[start:end + 1, j] = 1.0
    return matrix


def average_uniqueness(ind_matrix: np.ndarray) -> np.ndarray:
    """Average uniqueness per label: mean of 1/concurrency over the bars it spans."""
    concurrency = ind_matrix.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        contribution = ind_matrix / concurrency[:, None]     # 1/c where active, nan where c=0
    active = ind_matrix.sum(axis=0)
    out = np.divide(np.nansum(contribution, axis=0), active,
                    out=np.zeros(ind_matrix.shape[1]), where=active > 0)
    return out


def sequential_bootstrap(ind_matrix: np.ndarray, size: int = None,
                         seed: int = 0) -> List[int]:
    """Draw ``size`` label indices, each pick weighted by its uniqueness so far.

    At each step, for every candidate label, form the indicator of the
    already-drawn labels plus that candidate and take the candidate's average
    uniqueness within it; sample the next draw with probability proportional to
    those uniqueness values. Draws are with replacement. Returns the drawn label
    indices in order.
    """
    n_labels = ind_matrix.shape[1]
    size = n_labels if size is None else size
    rng = np.random.default_rng(seed)
    drawn: List[int] = []
    while len(drawn) < size:
        avg_u = np.zeros(n_labels)
        for j in range(n_labels):
            cols = drawn + [j]
            avg_u[j] = average_uniqueness(ind_matrix[:, cols])[-1]   # uniqueness of candidate j
        prob = avg_u / avg_u.sum()
        drawn.append(int(rng.choice(n_labels, p=prob)))
    return drawn
