"""Tests for the stress-scenario table."""

import numpy as np
import pandas as pd
import pytest

from quant_system.risk.stress import (
    historical_worst_moves,
    strategy_beta,
    stress_test,
)


def _benchmark(n=800, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    b = rng.normal(0.0003, 0.01, n)
    b[400] = -0.09                                       # plant a clear worst day
    return pd.Series(b, index=idx)


def test_strategy_beta_recovers_a_known_slope():
    b = _benchmark()
    strat = 1.5 * b
    assert strategy_beta(strat, b) == pytest.approx(1.5, rel=1e-6)


def test_worst_day_is_the_planted_shock():
    b = _benchmark()
    moves = historical_worst_moves(b, windows=(1, 5, 21))
    assert moves[1] == pytest.approx(-0.09)
    assert moves[5] <= moves[1]                          # a week including it is at least as bad


def test_stress_table_applies_beta_to_each_shock():
    b = _benchmark()
    strat = 1.5 * b
    table = stress_test(strat, b)
    assert list(table["scenario"]) == ["Worst market day", "Worst market week", "Worst market month"]
    row = table.set_index("scenario").loc["Worst market day"]
    assert row["est_pnl"] == pytest.approx(1.5 * row["market_move"])
    assert (table["est_pnl"] < 0).all()                 # a long-beta book loses in every shock


def test_custom_scenarios_are_appended():
    b = _benchmark()
    table = stress_test(b, b, custom={"Hypothetical -20%": -0.20})
    assert "Hypothetical -20%" in set(table["scenario"])
    row = table.set_index("scenario").loc["Hypothetical -20%"]
    assert row["market_move"] == pytest.approx(-0.20)
