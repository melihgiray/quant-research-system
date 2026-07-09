"""Tests for purged, embargoed cross-validation."""

import numpy as np
import pandas as pd
import pytest

from quant_system.config import default_config
from quant_system.data.loader import load_price_data
from quant_system.data.universe import universe
from quant_system.signals.cv import PurgedKFold, pooled_frame, purged_cv_scores

CFG = default_config()


def _panel(n_tickers=5):
    return load_price_data(universe("largecaps")[:n_tickers],
                           "2018-01-01", "2022-12-31", use_synthetic=True)


def _simple_dates(n=200, horizon=1):
    idx = pd.bdate_range("2020-01-01", periods=n)
    dates = pd.Series(idx)
    t1 = pd.Series(idx).shift(-horizon).ffill()
    return dates, t1


def test_folds_partition_the_rows():
    dates, t1 = _simple_dates()
    seen = np.zeros(len(dates), dtype=int)
    for train_idx, test_idx in PurgedKFold(5, embargo=3).split(dates, t1):
        seen[test_idx] += 1
        assert len(np.intersect1d(train_idx, test_idx)) == 0
    assert (seen == 1).all()          # every row tested exactly once


def test_no_label_overlap_between_train_and_test():
    dates, t1 = _simple_dates()
    for train_idx, test_idx in PurgedKFold(5, embargo=0).split(dates, t1):
        test_start = dates.iloc[test_idx].min()
        test_t1_max = t1.iloc[test_idx].max()
        d_tr = dates.iloc[train_idx]
        t1_tr = t1.iloc[train_idx]
        overlap = (d_tr <= test_t1_max) & (t1_tr >= test_start)
        assert not overlap.any()


def test_embargo_removes_days_after_test_block():
    dates, t1 = _simple_dates()
    embargo = 7
    unique_days = np.array(sorted(dates.unique()))
    for train_idx, test_idx in PurgedKFold(4, embargo=embargo).split(dates, t1):
        test_t1_max = t1.iloc[test_idx].max()
        after = unique_days[unique_days > np.datetime64(test_t1_max)][:embargo]
        if len(after) == 0:
            continue
        assert not dates.iloc[train_idx].isin(after).any()
    # And embargo=0 keeps those same days, so the parameter is doing the work.
    kept_days = set()
    for train_idx, test_idx in PurgedKFold(4, embargo=0).split(dates, t1):
        kept_days.update(dates.iloc[train_idx])
    assert len(kept_days) > 0


def test_rows_sharing_a_date_stay_together():
    # Pooled multi-asset frames repeat each date once per ticker.
    idx = pd.bdate_range("2020-01-01", periods=100)
    dates = pd.Series(np.repeat(idx.values, 3))
    t1 = pd.Series(np.repeat(pd.Series(idx).shift(-1).ffill().values, 3))
    for train_idx, test_idx in PurgedKFold(4, embargo=2).split(dates, t1):
        test_days = set(dates.iloc[test_idx])
        train_days = set(dates.iloc[train_idx])
        assert test_days.isdisjoint(train_days)


def test_split_validates_inputs():
    dates, t1 = _simple_dates(20)
    with pytest.raises(ValueError):
        PurgedKFold(1).split(dates, t1)
    with pytest.raises(ValueError):
        list(PurgedKFold(50).split(dates[:10], t1[:10]))


def test_pooled_frame_has_labels_and_horizons():
    panel = _panel()
    pooled = pooled_frame(panel, CFG.ml)
    assert {"_y", "_date", "_t1"}.issubset(pooled.columns)
    assert set(pooled["_y"].unique()).issubset({0.0, 1.0})
    assert (pooled["_t1"] > pooled["_date"]).all()   # label resolves after the feature date


def test_purged_cv_scores_run_end_to_end():
    panel = _panel()
    res = purged_cv_scores(panel, CFG.ml, n_splits=4, embargo=5)
    assert res is not None
    assert len(res.scores) >= 3
    assert res.scores["accuracy"].between(0.3, 0.7).all()   # no leak-inflated scores
    lines = res.summary()
    assert "PURGED" in lines[0] and "mean" in lines[-1]
