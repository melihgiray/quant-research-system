"""Purged and embargoed cross-validation for financial ML.

Plain K-fold is broken for this kind of data, in two ways. First, a label here
spans an interval: the row dated t is labelled with the return over (t, t+1].
If a training row's label interval overlaps the test window, the model has
effectively seen part of the test answer. Second, financial series are serially
correlated, so a training row dated right after the test window still carries
an echo of it.

The fix (Lopez de Prado, "Advances in Financial Machine Learning", ch. 7):

  * Split by time, never by shuffling rows. Test folds are contiguous blocks.
  * PURGE: drop any training row whose label interval overlaps the test window.
  * EMBARGO: additionally drop training rows for a few days after the test
    window ends.

The result is a score you can take at face value. It usually comes out lower
than the naive K-fold score, and the gap between the two is a direct measure of
how much leakage the naive procedure was rewarding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import MLConfig


class PurgedKFold:
    """Time-ordered K-fold with purging and an embargo.

    Parameters
    ----------
    n_splits : int
        Number of contiguous test blocks.
    embargo : int
        Trading days dropped from training immediately after each test block.

    The splitter works on row timestamps plus each row's label-end time, so it
    handles pooled multi-asset frames where many rows share a date: rows from
    the same date always land on the same side of a split.
    """

    def __init__(self, n_splits: int = 5, embargo: int = 5):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        self.n_splits = n_splits
        self.embargo = embargo

    def split(self, dates: pd.Series,
              t1: pd.Series) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield (train_idx, test_idx) positional indices.

        Parameters
        ----------
        dates : pd.Series of Timestamps
            Observation time of each row (the feature date).
        t1 : pd.Series of Timestamps
            When each row's label is fully known (for a next-day label, the
            following trading day).
        """
        dates = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
        t1 = pd.to_datetime(pd.Series(t1).reset_index(drop=True))
        if len(dates) != len(t1):
            raise ValueError("dates and t1 must have the same length")

        unique_days = np.array(sorted(dates.unique()))
        if len(unique_days) < self.n_splits:
            raise ValueError("fewer unique dates than splits")
        blocks = np.array_split(unique_days, self.n_splits)

        date_vals = dates.values
        t1_vals = t1.values
        for block in blocks:
            test_start, test_end = block[0], block[-1]
            test_mask = (date_vals >= test_start) & (date_vals <= test_end)
            test_idx = np.where(test_mask)[0]

            # Latest time at which any test label resolves.
            test_t1_max = t1_vals[test_mask].max()

            # Embargo horizon: skip this many trading days after the test block.
            days_after = unique_days[unique_days > test_t1_max]
            if self.embargo > 0 and len(days_after) > 0:
                embargo_end = days_after[: self.embargo][-1]
            else:
                embargo_end = test_t1_max

            # PURGE: a training row leaks if its label interval [date, t1]
            # overlaps the test window [test_start, test_t1_max].
            overlaps = (date_vals <= test_t1_max) & (t1_vals >= test_start)
            # EMBARGO: rows starting inside (test_t1_max, embargo_end].
            embargoed = (date_vals > test_t1_max) & (date_vals <= embargo_end)

            train_idx = np.where(~test_mask & ~overlaps & ~embargoed)[0]
            yield train_idx, test_idx


def pooled_frame(price_data, cfg: MLConfig = None) -> pd.DataFrame:
    """All assets stacked into one labelled frame, dates kept as a column.

    Columns: the feature set, plus `_y` (next-day up/down), `_date` (feature
    date) and `_t1` (the next trading day, when the label is known). Rows with
    incomplete features or an unknown label are dropped.
    """
    from .ml_signal import compute_features, FEATURE_NAMES

    cfg = cfg or MLConfig()
    features = compute_features(price_data, cfg)
    rets = price_data.returns()
    index = price_data.close.index
    next_day = pd.Series(index[1:].append(pd.DatetimeIndex([pd.NaT])), index=index)

    parts = []
    for t, feats in features.items():
        next_ret = rets[t].shift(-1)
        df = feats.copy()
        df["_y"] = (next_ret > 0).astype(float).where(next_ret.notna())
        df["_date"] = df.index
        df["_t1"] = next_day.reindex(df.index).values
        parts.append(df.dropna())
    pooled = pd.concat(parts, ignore_index=True)
    return pooled[FEATURE_NAMES + ["_y", "_date", "_t1"]]


@dataclass
class CVResult:
    """Per-fold scores plus the settings that produced them."""

    scores: pd.DataFrame        # fold, n_train, n_test, accuracy, auc
    n_splits: int
    embargo: int

    def summary(self) -> List[str]:
        lines = [f"PURGED {self.n_splits}-FOLD CV (embargo {self.embargo}d)"]
        for _, row in self.scores.iterrows():
            lines.append(f" fold {int(row['fold'])}  acc={row['accuracy']:.3f}  "
                         f"auc={row['auc']:.3f}  "
                         f"(train {int(row['n_train'])}, test {int(row['n_test'])})")
        lines.append(f" mean  acc={self.scores['accuracy'].mean():.3f}  "
                     f"auc={self.scores['auc'].mean():.3f}")
        return lines


def purged_cv_scores(price_data, cfg: MLConfig = None, n_splits: int = 5,
                     embargo: int = 5, seed: int = 7,
                     splitter: Optional[PurgedKFold] = None) -> Optional[CVResult]:
    """Score the ML signal's classifier with purged, embargoed K-fold CV.

    Accuracy and ROC AUC per fold. AUC is the more honest of the two here:
    with a roughly balanced up/down label, accuracy hovers near 0.5 and hides
    small amounts of skill that AUC picks up.

    Returns None when there is not enough data to build the folds.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score
    from .ml_signal import FEATURE_NAMES

    cfg = cfg or MLConfig()
    pooled = pooled_frame(price_data, cfg)
    if pooled.empty:
        return None

    X = pooled[FEATURE_NAMES].values
    y = pooled["_y"].values
    splitter = splitter or PurgedKFold(n_splits=n_splits, embargo=embargo)

    rows = []
    for k, (train_idx, test_idx) in enumerate(
            splitter.split(pooled["_date"], pooled["_t1"])):
        if len(train_idx) < 100 or len(np.unique(y[train_idx])) < 2:
            continue
        model = HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=300,
            l2_regularization=1.0, random_state=seed,
        )
        model.fit(X[train_idx], y[train_idx])
        proba = model.predict_proba(X[test_idx])[:, 1]
        pred = (proba > 0.5).astype(float)
        try:
            auc = roc_auc_score(y[test_idx], proba)
        except ValueError:            # single-class test fold
            auc = float("nan")
        rows.append({
            "fold": k,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "accuracy": accuracy_score(y[test_idx], pred),
            "auc": auc,
        })
    if not rows:
        return None
    return CVResult(scores=pd.DataFrame(rows),
                    n_splits=splitter.n_splits, embargo=splitter.embargo)
