"""Tests for meta-labeling."""

import numpy as np
import pandas as pd
import pytest

from quant_system.signals.meta_labeling import primary_side


def _proba():
    idx = pd.bdate_range("2021-01-01", periods=3)
    return pd.DataFrame({
        "A": [0.70, 0.30, 0.52],     # long, short, near-flat
        "B": [0.48, np.nan, 0.90],   # near-flat, missing, long
    }, index=idx)


def test_side_is_long_short_or_flat():
    side = primary_side(_proba(), deadband=0.05)
    assert side.loc[side.index[0], "A"] == 1.0      # 0.70 -> long
    assert side.loc[side.index[1], "A"] == -1.0     # 0.30 -> short
    assert side.loc[side.index[2], "A"] == 0.0      # 0.52 within deadband -> flat
    assert side.loc[side.index[0], "B"] == 0.0      # 0.48 within deadband -> flat


def test_missing_prediction_stays_missing():
    side = primary_side(_proba(), deadband=0.05)
    assert np.isnan(side.loc[side.index[1], "B"])   # no P(up) -> no side


def test_zero_deadband_takes_a_side_everywhere_defined():
    side = primary_side(_proba(), deadband=0.0)
    assert side.loc[side.index[2], "A"] == 1.0      # 0.52 > 0.5 -> long with no deadband
    assert set(np.unique(side.dropna().to_numpy())) <= {-1.0, 0.0, 1.0}
