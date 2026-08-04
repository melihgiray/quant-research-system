"""Tests for combining strategy sleeves into one book."""

import numpy as np
import pandas as pd
import pytest

from quant_system.portfolio.allocator import inverse_vol_allocations


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
