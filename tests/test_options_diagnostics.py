import pandas as pd
import pytest

from quant_system.options.chain import OptionChain
from quant_system.options.diagnostics import put_call_parity_residuals


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
