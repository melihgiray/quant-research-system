"""Tests for combining strategy sleeves into one book."""

import numpy as np
import pandas as pd
import pytest

from quant_system.portfolio.allocator import (
    blend_returns,
    combine_weights,
    erc_allocations,
    hrp_allocations,
    inverse_vol_allocations,
    volatility_target,
)


def _streams(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    calm = pd.Series(rng.normal(0.0003, 0.004, n), index=idx)     # low vol
    jumpy = pd.Series(rng.normal(0.0003, 0.020, n), index=idx)    # 5x the vol
    return {"calm": calm, "jumpy": jumpy}


def _three_streams(n=500, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    return {
        "a": pd.Series(rng.normal(0.0003, 0.006, n), index=idx),
        "b": pd.Series(rng.normal(0.0002, 0.012, n), index=idx),
        "c": pd.Series(rng.normal(0.0004, 0.009, n), index=idx),
    }


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


def test_hrp_allocations_warmup_is_zero_then_sums_to_one():
    alloc = hrp_allocations(_three_streams(), lookback=60, refit_every=21)
    assert (alloc.iloc[:60].to_numpy() == 0.0).all()             # nothing before warm-up
    warm = alloc.iloc[60:]
    assert np.allclose(warm.sum(axis=1).to_numpy(), 1.0)
    assert (warm.to_numpy() >= 0.0).all()                        # HRP is long-only


def test_hrp_allocations_are_causal():
    streams = _three_streams(seed=1)
    base = hrp_allocations(streams, lookback=60, refit_every=21)
    tampered = {k: v.copy() for k, v in streams.items()}
    tampered["b"].iloc[300:] *= 8                                # blow up a sleeve's tail
    after = hrp_allocations(tampered, lookback=60, refit_every=21)
    pd.testing.assert_frame_equal(base.iloc[:300], after.iloc[:300])


def test_erc_allocations_warmup_zero_sum_to_one_and_causal():
    streams = _three_streams(seed=3)
    alloc = erc_allocations(streams, lookback=60, refit_every=21)
    assert (alloc.iloc[:60].to_numpy() == 0.0).all()
    warm = alloc.iloc[60:]
    assert np.allclose(warm.sum(axis=1).to_numpy(), 1.0)
    assert (warm.to_numpy() >= 0.0).all()
    tampered = {k: v.copy() for k, v in streams.items()}
    tampered["b"].iloc[300:] *= 8
    after = erc_allocations(tampered, lookback=60, refit_every=21)
    pd.testing.assert_frame_equal(alloc.iloc[:300], after.iloc[:300])


def test_hrp_allocations_blend_is_a_valid_return_stream():
    streams = _three_streams(seed=2)
    alloc = hrp_allocations(streams, lookback=60, refit_every=21)
    blended = blend_returns(streams, allocations=alloc)
    assert len(blended) == len(next(iter(streams.values())))
    assert blended.iloc[:60].abs().sum() == 0.0                  # no allocation, no P&L


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


def test_blend_with_given_allocations_is_weighted_sum():
    idx = pd.bdate_range("2021-01-01", periods=3)
    streams = {"a": pd.Series([0.01, 0.02, -0.01], index=idx),
               "b": pd.Series([0.00, -0.01, 0.03], index=idx)}
    alloc = pd.DataFrame({"a": [0.5, 0.5, 0.5], "b": [0.5, 0.5, 0.5]}, index=idx)
    out = blend_returns(streams, allocations=alloc)
    expected = [0.005, 0.005, 0.01]                                  # simple mean each day
    assert out.tolist() == pytest.approx(expected)


def test_blend_defaults_to_inverse_vol_and_is_causal():
    streams = _streams()
    out = blend_returns(streams, lookback=40, min_periods=20)
    # Early rows have no allocation, so the blended return is 0 there.
    assert out.iloc[0] == 0.0
    # Tampering with the far tail must not change early blended returns.
    tampered = {k: v.copy() for k, v in streams.items()}
    tampered["jumpy"].iloc[300:] *= 5
    out2 = blend_returns(tampered, lookback=40, min_periods=20)
    pd.testing.assert_series_equal(out.iloc[:300], out2.iloc[:300])


def test_volatility_target_moves_realised_vol_toward_target():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2018-01-01", periods=1000)
    raw = pd.Series(rng.normal(0.0, 0.03, 1000), index=idx)         # ~48% annual vol
    scaled = volatility_target(raw, target_vol=0.10, lookback=126, max_leverage=5.0)
    tail = scaled.iloc[300:]
    realised = tail.std() * np.sqrt(252)
    assert 0.06 < realised < 0.14                                   # near the 10% target
    assert realised < raw.iloc[300:].std() * np.sqrt(252)          # scaled the book down


def test_volatility_target_is_causal():
    rng = np.random.default_rng(4)
    idx = pd.bdate_range("2018-01-01", periods=600)
    raw = pd.Series(rng.normal(0.0, 0.02, 600), index=idx)
    base = volatility_target(raw, lookback=100)
    tampered = raw.copy()
    tampered.iloc[400:] *= 8
    after = volatility_target(tampered, lookback=100)
    # Scaler is lagged, so day t depends only on returns through t-1; days up to
    # 400 use the untouched history and the scaled return equals raw*scaler there.
    pd.testing.assert_series_equal(base.iloc[:400], after.iloc[:400])
