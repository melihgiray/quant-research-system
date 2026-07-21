"""Phase 2 tests: chain hygiene, surface construction, no-arbitrage checks.

The arbitrage tests work in both directions, which matters. It is easy to write
a checker that reports nothing, and easy to write one that reports everything.
So: a clean synthetic chain must come back with zero violations, and a chain
with a violation deliberately injected must have exactly that violation found.
"""

import numpy as np
import pandas as pd
import pytest

from quant_system.config import OptionsConfig
from quant_system.options import (
    CALL, PUT, black_scholes_price,
    synthetic_option_chain, build_surface, check_butterfly, check_calendar,
    check_surface,
)
from quant_system.options.arbitrage import BUTTERFLY, CALENDAR, MONOTONICITY
from quant_system.options.chain import _clean_quotes, DAYS_PER_YEAR


RATE = 0.04


@pytest.fixture(scope="module")
def chain():
    return synthetic_option_chain(spot=100.0, rate=RATE)


@pytest.fixture(scope="module")
def surface(chain):
    return build_surface(chain, rate=RATE)


# --------------------------------------------------------------------------- #
# Chain hygiene
# --------------------------------------------------------------------------- #
def test_synthetic_chain_is_labelled_and_populated(chain):
    assert chain.synthetic is True
    assert chain.n_quotes > 0
    assert len(chain.expiries) == 5
    assert "SYNTHETIC" in "\n".join(chain.summary())
    assert (chain.quotes["ask"] >= chain.quotes["bid"]).all()
    assert (chain.quotes["mid"] > 0).all()


def test_hygiene_filters_remove_untradeable_quotes_and_report_why():
    # One row per defect, plus one good row.
    raw = pd.DataFrame([
        # good
        dict(expiry=pd.Timestamp("2025-03-21"), time_to_expiry=0.25, strike=100.0,
             option_type=CALL, bid=5.0, ask=5.2, volume=10, open_interest=100),
        # zero bid: nobody is buying, so there is no market
        dict(expiry=pd.Timestamp("2025-03-21"), time_to_expiry=0.25, strike=110.0,
             option_type=CALL, bid=0.0, ask=1.0, volume=10, open_interest=100),
        # crossed
        dict(expiry=pd.Timestamp("2025-03-21"), time_to_expiry=0.25, strike=120.0,
             option_type=CALL, bid=3.0, ask=2.0, volume=10, open_interest=100),
        # spread far too wide to be a usable price
        dict(expiry=pd.Timestamp("2025-03-21"), time_to_expiry=0.25, strike=130.0,
             option_type=CALL, bid=0.10, ask=5.00, volume=10, open_interest=100),
        # already expired
        dict(expiry=pd.Timestamp("2024-01-01"), time_to_expiry=-0.1, strike=100.0,
             option_type=CALL, bid=5.0, ask=5.2, volume=10, open_interest=100),
    ])
    clean, dropped = _clean_quotes(raw, OptionsConfig())
    assert len(clean) == 1
    assert clean.iloc[0]["strike"] == 100.0
    assert dropped["zero_or_low_bid"] == 1
    assert dropped["crossed_or_locked"] == 1
    assert dropped["spread_too_wide"] == 1
    assert dropped["expired"] == 1


def test_mid_is_used_not_last_price(chain):
    row = chain.quotes.iloc[0]
    assert row["mid"] == pytest.approx(0.5 * (row["bid"] + row["ask"]))


# --------------------------------------------------------------------------- #
# Surface construction
# --------------------------------------------------------------------------- #
def test_surface_recovers_the_generating_smile(chain):
    # The synthetic chain is generated from a known vol function, so inverting
    # its mid prices should return that function to within the bid/ask width.
    surf = build_surface(chain, rate=RATE)
    assert len(surf.points) > 0
    assert surf.synthetic is True

    atm = surf.implied_vol(0.0, 30 / DAYS_PER_YEAR)
    # base_vol 0.22 plus term_slope*sqrt(T); skew term vanishes at k=0.
    expected = 0.22 + 0.02 * np.sqrt(30 / DAYS_PER_YEAR)
    assert atm == pytest.approx(expected, abs=0.02)


def test_surface_has_downward_skew(surface):
    # Equity smiles: downside puts are bid above upside calls.
    short = surface.slice(surface.expiries[0])
    below = short[short["log_moneyness"] < -0.05]["implied_vol"].mean()
    above = short[short["log_moneyness"] > 0.05]["implied_vol"].mean()
    assert below > above


def test_total_variance_interpolation_is_monotone_in_time(surface):
    # Interpolating in total variance is the whole reason for the coordinate
    # choice: it must not manufacture a calendar violation between slices.
    ts = np.linspace(surface.maturities[0], surface.maturities[-1], 40)
    for k in (-0.10, 0.0, 0.10):
        w = np.array([surface.total_variance(k, float(t)) for t in ts])
        assert np.all(np.diff(w) >= -1e-9), f"total variance dipped at k={k}"


def test_interpolation_hits_the_observed_points(surface):
    # At an observed (k, T) the interpolant should return the observed vol.
    row = surface.points.iloc[len(surface.points) // 2]
    got = surface.implied_vol(row["log_moneyness"], row["time_to_expiry"])
    assert got == pytest.approx(row["implied_vol"], abs=1e-9)


def test_interpolation_clamps_instead_of_extrapolating(surface):
    # Far outside the observed wings, hold the edge rather than inventing one.
    edge = surface.slice(surface.expiries[0])["log_moneyness"].min()
    t = surface.maturities[0]
    assert surface.implied_vol(edge - 5.0, t) == pytest.approx(
        surface.implied_vol(edge, t), abs=1e-9)
    # Same beyond the longest expiry.
    long_t = surface.maturities[-1]
    assert surface.total_variance(0.0, long_t * 3) == pytest.approx(
        surface.total_variance(0.0, long_t), abs=1e-9)


def test_atm_term_structure_shape(surface):
    atm = surface.atm_term_structure()
    assert len(atm) == len(surface.maturities)
    assert (atm["atm_vol"] > 0).all()


def test_build_surface_raises_on_empty_chain(chain):
    empty = type(chain)(quotes=chain.quotes.iloc[0:0], spot=100.0,
                        as_of=chain.as_of, ticker="X", synthetic=True)
    with pytest.raises(RuntimeError, match="empty chain"):
        build_surface(empty, rate=RATE)


# --------------------------------------------------------------------------- #
# No-arbitrage checks: must be quiet on clean data, loud on broken data
# --------------------------------------------------------------------------- #
def test_clean_synthetic_surface_has_no_violations(surface):
    report = surface.arbitrage
    assert report.n_butterfly_checks > 0
    assert report.n_calendar_checks > 0
    assert report.clean, f"unexpected violations: {report.counts()}"


def test_injected_butterfly_violation_is_flagged(surface):
    # Make the middle strike of one expiry too expensive. That concavity implies
    # a negative risk-neutral density and must be caught.
    points = surface.points.copy()
    exp = surface.expiries[1]
    idx = points.index[points["expiry"] == exp].tolist()
    assert len(idx) >= 3
    middle = idx[len(idx) // 2]
    points.loc[middle, "call_price"] *= 3.0

    violations = check_butterfly(points, tolerance=1e-6)
    kinds = {v.kind for v in violations}
    assert BUTTERFLY in kinds or MONOTONICITY in kinds
    assert any(v.severity > 0 for v in violations)


def test_injected_calendar_violation_is_flagged(surface):
    # Collapse the far expiry's total variance below the near one's at matched
    # moneyness: uncertainty cannot shrink with more time.
    points = surface.points.copy()
    far = surface.expiries[-1]
    points.loc[points["expiry"] == far, "total_variance"] *= 0.01

    violations = check_calendar(points, tolerance=1e-6)
    assert violations, "a collapsed far-dated variance should violate calendar arb"
    assert all(v.kind == CALENDAR for v in violations)
    assert violations[0].severity > 0


def test_report_summary_reads_cleanly_both_ways(surface):
    clean = "\n".join(surface.arbitrage.summary())
    assert "no violations found" in clean

    broken = surface.points.copy()
    broken.loc[broken.index[len(broken) // 2], "call_price"] *= 5.0
    report = check_surface(broken)
    text = "\n".join(report.summary())
    assert not report.clean
    assert "violation" in text
    assert report.worst() is not None


def test_violations_are_graded_against_the_local_bid_ask():
    # Surfaces are built from mids, but you cannot trade at mid. A violation
    # smaller than the spread you would have to cross is an artifact; one larger
    # is worth investigating. The checker must tell them apart.
    base = pd.DataFrame({
        "expiry": [pd.Timestamp("2025-06-20")] * 3,
        "strike": [95.0, 100.0, 105.0],
        "log_moneyness": [-0.05, 0.0, 0.05],
        "total_variance": [0.01, 0.01, 0.01],
        "call_price": [8.0, 7.0, 4.5],           # concave: a real 0.75 violation
    })

    wide = base.assign(spread=[1.0, 1.0, 1.0])   # violation buried in the spread
    v_wide = check_butterfly(wide, tolerance=1e-6)
    assert v_wide and all(v.within_spread is True for v in v_wide)

    tight = base.assign(spread=[0.001, 0.001, 0.001])
    v_tight = check_butterfly(tight, tolerance=1e-6)
    assert v_tight and all(v.within_spread is False for v in v_tight)

    report = check_surface(tight)
    assert len(report.tradeable) == len(v_tight)
    assert "exceed the local bid-ask spread" in "\n".join(report.summary())


def test_violation_grading_is_none_without_spread_data():
    points = pd.DataFrame({
        "expiry": [pd.Timestamp("2025-06-20")] * 3,
        "strike": [95.0, 100.0, 105.0],
        "log_moneyness": [-0.05, 0.0, 0.05],
        "total_variance": [0.01, 0.01, 0.01],
        "call_price": [8.0, 7.0, 4.5],
    })
    violations = check_butterfly(points, tolerance=1e-6)
    assert violations and all(v.within_spread is None for v in violations)
    assert check_surface(points).tradeable == []


def test_monotonicity_violation_is_distinct_from_butterfly():
    # Call prices rising with strike is its own defect, reported separately.
    points = pd.DataFrame({
        "expiry": [pd.Timestamp("2025-06-20")] * 3,
        "strike": [90.0, 100.0, 110.0],
        "call_price": [5.0, 6.0, 7.0],           # rising: wrong direction
        "log_moneyness": [-0.1, 0.0, 0.1],
        "total_variance": [0.01, 0.01, 0.01],
    })
    violations = check_butterfly(points, tolerance=1e-6)
    assert any(v.kind == MONOTONICITY for v in violations)
