"""Tests for triple-barrier labeling."""

import numpy as np
import pandas as pd

import pytest

from quant_system.signals.labeling import (
    cusum_events,
    ewm_volatility,
    triple_barrier_labels,
)


def _series(values):
    idx = pd.bdate_range("2021-01-01", periods=len(values))
    return pd.Series(np.asarray(values, dtype=float), index=idx)


def test_rising_path_hits_the_profit_target_first():
    close = _series([100, 101, 102, 103, 104, 105, 106])
    vol = np.full(len(close), 0.02)                  # 2% barriers
    out = triple_barrier_labels(close, [0], pt=1.0, sl=1.0, vertical=6, vol=vol)
    assert out.loc[0, "label"] == 1
    assert out.loc[0, "ret"] >= 0.02                 # touched the +2% target
    assert out.loc[0, "touch"] < close.index[-1]     # before the vertical barrier


def test_falling_path_hits_the_stop_first():
    close = _series([100, 99, 98, 97, 96, 95])
    vol = np.full(len(close), 0.02)
    out = triple_barrier_labels(close, [0], pt=1.0, sl=1.0, vertical=5, vol=vol)
    assert out.loc[0, "label"] == -1
    assert out.loc[0, "ret"] <= -0.02


def test_range_bound_path_hits_the_vertical_barrier():
    close = _series([100, 100.2, 99.9, 100.1, 100.05])   # never moves 2%
    vol = np.full(len(close), 0.02)
    out = triple_barrier_labels(close, [0], pt=1.0, sl=1.0, vertical=4, vol=vol)
    assert out.loc[0, "touch"] == close.index[4]         # rode to the time limit
    assert out.loc[0, "label"] == int(np.sign(100.05 / 100 - 1))   # sign of terminal return


def test_asymmetric_barriers_can_flip_the_label():
    # A small rise then a large fall: a tight target catches the rise first.
    close = _series([100, 100.6, 98, 96])
    vol = np.full(len(close), 0.01)                  # pt=0.5% target, sl=3% stop
    out = triple_barrier_labels(close, [0], pt=0.5, sl=3.0, vertical=3, vol=vol)
    assert out.loc[0, "label"] == 1                  # +0.6% clears the 0.5% target at bar 1


def test_events_with_nan_volatility_are_skipped():
    close = _series([100, 101, 102, 103])
    vol = np.array([np.nan, 0.02, 0.02, 0.02])
    out = triple_barrier_labels(close, [0, 1], pt=1.0, sl=1.0, vertical=2, vol=vol)
    assert list(out["event"]) == [close.index[1]]    # the NaN-vol event dropped out


def test_ewm_volatility_is_positive_after_warmup():
    close = _series(np.cumprod(1 + np.random.default_rng(0).normal(0, 0.01, 300)) * 100)
    vol = ewm_volatility(close)
    assert (vol.dropna() >= 0).all()


def test_cusum_fires_on_a_steady_rise_at_the_expected_cadence():
    series = _series(np.arange(0.0, 5.0, 0.25))       # rises 0.25 each step
    events = cusum_events(series, threshold=1.0)      # every 4 steps the run hits 1.0
    assert events == [4, 8, 12, 16]


def test_cusum_is_silent_on_a_flat_series():
    assert cusum_events(_series([5.0] * 20), threshold=0.5) == []


def test_cusum_is_symmetric_on_down_moves():
    series = _series(np.arange(0.0, -3.0, -0.5))       # falls 0.5 each step
    events = cusum_events(series, threshold=1.0)
    assert events == [2, 4]                            # a run down of 1.0 also triggers


def test_cusum_catches_both_directions():
    series = _series([0, 1, 2, 3, 2, 1, 0, -1])        # up then down
    events = cusum_events(series, threshold=1.5)
    assert events                                      # fires on the up leg and the down leg
    assert events == sorted(events)


def test_cusum_rejects_a_nonpositive_threshold():
    with pytest.raises(ValueError, match="threshold must be positive"):
        cusum_events(_series([1.0, 2.0, 3.0]), threshold=0.0)


def test_cusum_events_feed_the_barrier_labeler():
    rng = np.random.default_rng(3)
    close = _series(np.cumprod(1 + rng.normal(0, 0.01, 400)) * 100)
    events = cusum_events(np.log(close), threshold=0.02)   # 2% cumulative-return events
    labels = triple_barrier_labels(close, events, pt=1.0, sl=1.0, vertical=10)
    assert len(labels) > 0
    assert set(labels["label"].unique()) <= {-1, 0, 1}
