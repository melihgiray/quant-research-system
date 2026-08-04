"""Tests for combining strategy sleeves into one book."""

import numpy as np
import pandas as pd
import pytest

from quant_system.portfolio.allocator import (
    combine_weights,
    inverse_vol_allocations,
)


def _streams(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    calm = pd.Series(rng.normal(0.0003, 0.004, n), index=idx)     # low vol
    jumpy = pd.Series(rng.normal(0.0003, 0.020, n), index=idx)    # 5x the vol
    return {"calm": calm, "jumpy": jumpy}


def test_lower_vol_sleeve_gets_more_capital():
    alloc = inverse_vol_allocations(_streams(), lookback=40, min_periods=20)
    warm = alloc.iloc[100]                                        # well past warm-up
    assert warm["calm"] > warm["jumpy"]
    assert warm.sum() == pytest.approx(1.0)


def test_rows_sum_to_one_after_warmup():
    alloc = inverse_vol_allocations(_streams(), lookback=40, min_periods=20)
    warm = alloc.iloc[60:]
    assert np.allclose(warm.sum(axis=1).to_numpy(), 1.0)


def test_no_allocation_before_warmup():
    alloc = inverse_vol_allocations(_streams(), lookback=40, min_periods=20)
    assert alloc.iloc[0].sum() == 0.0                             # nothing estimable yet


def test_allocation_is_causal():
    # The allocation on day t must not change when returns after t are altered.
    streams = _streams()
    base = inverse_vol_allocations(streams, lookback=40, min_periods=20)
    tampered = {k: v.copy() for k, v in streams.items()}
    tampered["jumpy"].iloc[200:] *= 10                           # blow up the tail
    after = inverse_vol_allocations(tampered, lookback=40, min_periods=20)
    pd.testing.assert_frame_equal(base.iloc[:200], after.iloc[:200])


def test_empty_input_raises():
    with pytest.raises(ValueError, match="at least one sleeve"):
        inverse_vol_allocations({})


def test_combine_weights_is_allocation_weighted_sum():
    idx = pd.bdate_range("2021-01-01", periods=3)
    alloc = pd.DataFrame({"a": [0.25, 0.5, 0.5], "b": [0.75, 0.5, 0.5]}, index=idx)
    wa = pd.DataFrame({"AAPL": [1.0, 1.0, 1.0]}, index=idx)          # a holds AAPL
    wb = pd.DataFrame({"MSFT": [1.0, 1.0, 1.0]}, index=idx)          # b holds MSFT
    out = combine_weights({"a": wa, "b": wb}, alloc)
    assert list(out.columns) == ["AAPL", "MSFT"]                     # union, sorted
    assert out["AAPL"].tolist() == pytest.approx([0.25, 0.5, 0.5])
    assert out["MSFT"].tolist() == pytest.approx([0.75, 0.5, 0.5])


def test_combine_weights_overlapping_ticker_adds():
    idx = pd.bdate_range("2021-01-01", periods=2)
    alloc = pd.DataFrame({"a": [0.4, 0.4], "b": [0.6, 0.6]}, index=idx)
    wa = pd.DataFrame({"SPY": [1.0, 1.0]}, index=idx)
    wb = pd.DataFrame({"SPY": [-1.0, -1.0]}, index=idx)              # opposite side
    out = combine_weights({"a": wa, "b": wb}, alloc)
    assert out["SPY"].tolist() == pytest.approx([-0.2, -0.2])        # 0.4 - 0.6


def test_combine_weights_requires_allocation_column():
    idx = pd.bdate_range("2021-01-01", periods=2)
    alloc = pd.DataFrame({"a": [1.0, 1.0]}, index=idx)
    wb = pd.DataFrame({"SPY": [1.0, 1.0]}, index=idx)
    with pytest.raises(ValueError, match="no allocation column"):
        combine_weights({"b": wb}, alloc)
