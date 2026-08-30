"""Tests for exception-frequency and exception-clustering VaR diagnostics."""

import numpy as np
import pandas as pd

from quant_system.risk.validation import (
    basel_traffic_light,
    christoffersen_conditional_coverage_test,
    christoffersen_independence_test,
)


def test_independence_accepts_well_spaced_exceptions():
    rng = np.random.default_rng(7)
    r = pd.Series(np.where(rng.random(10_000) < 0.05, -0.02, 0.0))
    result = christoffersen_independence_test(r, var=0.01)
    assert 400 < result["n_exceptions"] < 600
    assert result["transition_counts"]["n11"] > 0
    assert result["reject"] is False


def test_independence_rejects_clustered_exceptions():
    r = pd.Series(np.zeros(1_000))
    for start in range(0, 1_000, 100):
        r.iloc[start:start + 5] = -0.02
    result = christoffersen_independence_test(r, var=0.01)
    assert result["n_exceptions"] == 50
    assert result["transition_counts"]["n11"] > 0
    assert result["reject"] is True


def test_conditional_coverage_combines_the_two_tests():
    r = pd.Series(np.zeros(1_000))
    for start in range(0, 1_000, 100):
        r.iloc[start:start + 5] = -0.02
    result = christoffersen_conditional_coverage_test(r, var=0.01)
    assert result["lr_cc"] >= result["kupiec"]["lr_pof"]
    assert result["reject"] is True


def test_short_input_is_graceful():
    result = christoffersen_independence_test(pd.Series([-0.01]), var=0.02)
    assert np.isnan(result["lr_ind"])


def test_basel_traffic_light_matches_the_familiar_250_day_cutoffs():
    r = pd.Series(np.zeros(250))
    assert basel_traffic_light(r, var=0.01)["green_limit"] == 4
    r.iloc[:5] = -0.02
    assert basel_traffic_light(r, var=0.01)["zone"] == "yellow"
    r.iloc[:10] = -0.02
    assert basel_traffic_light(r, var=0.01)["zone"] == "red"


def test_basel_traffic_light_handles_a_daily_var_series():
    r = pd.Series([0.0, -0.03, 0.0])
    result = basel_traffic_light(r, pd.Series([0.01, 0.02, 0.01]), level=0.95)
    assert result["exceptions"] == 1
    assert result["n_obs"] == 3
