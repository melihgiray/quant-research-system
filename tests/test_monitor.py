"""Tests for the fail-loud repricing health check."""

import pandas as pd

from quant_system.config import default_config
from quant_system.options import build_surface, synthetic_option_chain
from quant_system.options.arbitrage import ArbitrageReport, ArbitrageViolation
from quant_system.options.monitor import repricing_health
from quant_system.options.surface import POINT_COLUMNS, VolSurface

CFG = default_config()


def _empty_surface(points=None, iv_failures=None, violations=None):
    report = ArbitrageReport(violations=violations or [])
    return VolSurface(
        points=points if points is not None else pd.DataFrame(columns=POINT_COLUMNS),
        spot=100.0, as_of=pd.Timestamp("2021-01-01"), ticker="X",
        rate=0.0, dividend_yield=0.0, arbitrage=report,
        iv_failures=iv_failures or {}, synthetic=True)


def test_a_built_surface_produces_a_well_formed_verdict():
    chain = synthetic_option_chain(ticker="SPY", cfg=CFG.options)
    surface = build_surface(chain, rate=0.04, dividend_yield=0.012, cfg=CFG.options)
    health = repricing_health(surface)
    assert health.n_points > 0
    assert 0.0 <= health.iv_failure_rate <= 1.0
    assert isinstance(health.ok, bool)


def test_a_clean_surface_is_healthy():
    points = pd.DataFrame({c: [0.0, 0.0] for c in POINT_COLUMNS})   # two usable points
    health = repricing_health(_empty_surface(points=points))       # no failures, no arb
    assert health.ok
    assert health.issues == []


def test_empty_chain_fails_loudly():
    health = repricing_health(_empty_surface())
    assert not health.ok
    assert any("no invertible quotes" in i for i in health.issues)


def test_high_iv_failure_rate_fails():
    points = pd.DataFrame({c: [0.0] for c in POINT_COLUMNS})   # one usable point
    health = repricing_health(_empty_surface(points=points, iv_failures={"no_vega": 99}))
    assert not health.ok
    assert any("failure rate" in i for i in health.issues)


def test_tradeable_arbitrage_fails():
    points = pd.DataFrame({c: [0.0] for c in POINT_COLUMNS})
    beyond = ArbitrageViolation("butterfly", 0.5, "stale print", within_spread=False)
    health = repricing_health(_empty_surface(points=points, violations=[beyond]))
    assert not health.ok
    assert health.tradeable_arb == 1
    assert any("beyond the bid-ask" in i for i in health.issues)


def test_mid_price_artefact_arbitrage_does_not_fail():
    points = pd.DataFrame({c: [0.0] for c in POINT_COLUMNS})
    inside = ArbitrageViolation("butterfly", 0.01, "mid artefact", within_spread=True)
    health = repricing_health(_empty_surface(points=points, violations=[inside]))
    assert health.ok                                          # inside the spread, not tradeable
    assert health.tradeable_arb == 0
