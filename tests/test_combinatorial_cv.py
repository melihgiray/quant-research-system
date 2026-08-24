"""Tests for combinatorial purged cross-validation."""

import math

import numpy as np
import pytest

from quant_system.signals.combinatorial_cv import combinatorial_purged_splits


def test_number_of_splits_is_n_choose_k():
    splits = list(combinatorial_purged_splits(600, n_groups=6, k_test=2))
    assert len(splits) == math.comb(6, 2)                # 15 paths


def test_train_and_test_are_disjoint():
    for train, test in combinatorial_purged_splits(600, n_groups=6, k_test=2):
        assert set(train).isdisjoint(set(test))


def test_embargo_removes_neighbours_of_test_groups():
    embargo = 5
    for train, test in combinatorial_purged_splits(600, n_groups=6, k_test=2, embargo=embargo):
        train_set = set(train.tolist())
        for t in test:
            for e in range(1, embargo + 1):
                assert (t + e) not in train_set or (t + e) in test
                assert (t - e) not in train_set or (t - e) in test


def test_no_embargo_partitions_all_samples():
    for train, test in combinatorial_purged_splits(600, n_groups=6, k_test=2, embargo=0):
        assert len(train) + len(test) == 600             # every sample is train or test


def test_rejects_bad_k():
    with pytest.raises(ValueError, match="k_test"):
        list(combinatorial_purged_splits(100, n_groups=5, k_test=5))
