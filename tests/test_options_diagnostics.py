import pandas as pd
import pytest

from quant_system.options.chain import OptionChain
from quant_system.options.diagnostics import put_call_parity_residuals
from quant_system.options.diagnostics import liquidity_profile


def _chain(call_mid=5.0, put_mid=5.0):
    expiry = pd.Timestamp("2027-01-01")
    quotes = pd.DataFrame([
        {"expiry": expiry, "time_to_expiry": 1.0, "strike": 100.0, "option_type": "call", "bid": call_mid - .1, "ask": call_mid + .1, "mid": call_mid, "volume": 1, "open_interest": 1},
        {"expiry": expiry, "time_to_expiry": 1.0, "strike": 100.0, "option_type": "put", "bid": put_mid - .1, "ask": put_mid + .1, "mid": put_mid, "volume": 1, "open_interest": 1},
    ])
    return OptionChain(quotes=quotes, spot=100.0, as_of=pd.Timestamp("2026-01-01"), ticker="TEST")


def test_parity_residual_is_zero_for_matched_quotes():
    report = put_call_parity_residuals(_chain())
    assert report.iloc[0]["residual"] == pytest.approx(0.0)
    assert not report.iloc[0]["exceeds_spread"]


def test_parity_flags_a_mismatch_larger_than_the_spreads():
    report = put_call_parity_residuals(_chain(call_mid=6.0, put_mid=5.0))
    assert report.iloc[0]["exceeds_spread"]


def test_liquidity_profile_uses_relative_spreads_and_activity():
    chain = _chain()
    profile = liquidity_profile(chain)
    assert profile["n_quotes"] == 2
    assert profile["median_spread"] == pytest.approx(0.2)
    assert profile["median_relative_spread"] == pytest.approx(0.04)
    assert profile["share_with_volume"] == 1.0


def test_volatility_skew_reports_the_expected_downside_shape():
    from quant_system.options.diagnostics import volatility_skew

    class Surface:
        @staticmethod
        def implied_vol(k, t):
            return 0.20 - 0.40 * k

    skew = volatility_skew(Surface(), days=30, wing=0.10)
    assert skew["put_iv"] == pytest.approx(0.24)
    assert skew["atm_iv"] == pytest.approx(0.20)
    assert skew["call_iv"] == pytest.approx(0.16)
    assert skew["put_minus_call"] == pytest.approx(0.08)


def test_term_structure_calls_out_inversion_without_flagging_it_as_bad_data():
    from quant_system.options.diagnostics import atm_term_structure_summary

    class Surface:
        @staticmethod
        def implied_vol(k, t):
            return 0.50 - 0.40 * t

    term = atm_term_structure_summary(Surface(), short_days=30, long_days=180)
    assert term["shape"] == "inverted"
    assert term["slope"] < 0


def test_surface_quality_gate_only_fails_on_material_problems():
    from quant_system.options.diagnostics import surface_quality_gate

    class Arb:
        tradeable = []
    class Surface:
        iv_failures = {}
        arbitrage = Arb()

    assert surface_quality_gate(_chain(), Surface())["ok"]
    Surface.iv_failures = {"solver_failed": 1}
    Arb.tradeable = [object()]
    result = surface_quality_gate(_chain(), Surface())
    assert not result["ok"]
    assert "tradeable_static_arbitrage" in result["reasons"]
