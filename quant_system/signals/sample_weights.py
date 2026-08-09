"""Sample weights by label uniqueness (Lopez de Prado, Advances in ML, ch. 4).

The pooled ML training set stacks every asset on every day, and the label is the
sign of the next day's return. On a big up-day for the market most assets are
labelled "up" together, so those rows are not independent observations: the model
sees the same market-wide move many times and can overfit to whichever days had
the most names trading. De Prado's fix is to weight each row by how *unique* its
label is.

For a label that spans a single bar (as here: day t predicts the return over the
one day t -> t+1), average uniqueness reduces exactly to one over the number of
labels concurrent at that bar, which in this pooled cross-section is the number
of assets sharing that label date. Rows on crowded days are down-weighted; a row
on a day when few names traded counts for more. This is deliberately the
single-bar special case, not the overlapping triple-barrier version, because the
labels here do not overlap in time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def concurrency_counts(label_dates) -> np.ndarray:
    """For each row, how many rows share its label date (the concurrency at that bar)."""
    dates = pd.Index(label_dates)
    counts = dates.value_counts()
    return counts.reindex(dates).to_numpy()


def uniqueness_weights(label_dates, normalize: bool = True) -> np.ndarray:
    """Per-row sample weights inversely proportional to label concurrency.

    A row's weight is 1 / (number of rows sharing its label date). With
    ``normalize`` the weights are rescaled to average 1, so passing them to a
    classifier reweights the rows without changing the effective sample size.
    """
    dates = pd.Index(label_dates)
    if len(dates) == 0:
        return np.empty(0)
    counts = concurrency_counts(dates)
    w = 1.0 / counts
    if normalize:
        w = w * (len(w) / w.sum())
    return w
