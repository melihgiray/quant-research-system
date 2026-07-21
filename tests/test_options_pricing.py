"""Phase 1 options tests: pricing, Greeks, implied volatility.

The headline test is ``test_analytic_and_fd_greeks_agree``: every Greek, from
both the closed form and central finite differences, across a grid of moneyness
and expiry. Agreement is evidence for both paths at once.
"""

import math

import numpy as np
import pytest

from quant_system.options import (
    CALL, PUT,
    black_scholes_price, binomial_price, american_price,
    european_binomial_price, early_exercise_premium,
    black_scholes_greeks, black_scholes_fd_greeks, american_greeks,
    implied_volatility, implied_volatility_detailed, no_arbitrage_bounds,
)
from quant_system.options.implied_vol import (
    REASON_OK, REASON_BELOW_INTRINSIC, REASON_ABOVE_MAX, REASON_EXPIRED,
    REASON_AT_INTRINSIC,
)


# --------------------------------------------------------------------------- #
# Black-Scholes
# --------------------------------------------------------------------------- #
def test_black_scholes_matches_known_values():
    # Textbook case: S=K=100, T=1, r=5%, vol=20%, no dividend.
    call = black_scholes_price(100, 100, 1.0, 0.05, 0.20, CALL)
    put = black_scholes_price(100, 100, 1.0, 0.05, 0.20, PUT)
    assert call == pytest.approx(10.4506, abs=1e-3)
    assert put == pytest.approx(5.5735, abs=1e-3)


def test_put_call_parity():
    s, k, t, r, v, q = 105.0, 95.0, 0.75, 0.03, 0.28, 0.015
    call = black_scholes_price(s, k, t, r, v, CALL, q)
    put = black_scholes_price(s, k, t, r, v, PUT, q)
    parity = s * math.exp(-q * t) - k * math.exp(-r * t)
    assert (call - put) == pytest.approx(parity, abs=1e-10)


def test_price_is_monotonic_in_vol_and_bounded():
    lo = black_scholes_price(100, 100, 1.0, 0.02, 0.10, CALL)
    hi = black_scholes_price(100, 100, 1.0, 0.02, 0.60, CALL)
    assert hi > lo                                   # more vol, more optionality
    lower, upper = no_arbitrage_bounds(100, 100, 1.0, 0.02, CALL)
    assert lower <= lo <= upper and lower <= hi <= upper


def test_degenerate_corners_return_intrinsic():
    # At expiry.
    assert black_scholes_price(120, 100, 0.0, 0.05, 0.3, CALL) == pytest.approx(20.0)
    assert black_scholes_price(80, 100, 0.0, 0.05, 0.3, CALL) == pytest.approx(0.0)
    assert black_scholes_price(80, 100, 0.0, 0.05, 0.3, PUT) == pytest.approx(20.0)
    # Zero vol: discounted forward intrinsic, not zero.
    zero_vol = black_scholes_price(100, 90, 1.0, 0.05, 0.0, CALL)
    assert zero_vol == pytest.approx(100 - 90 * math.exp(-0.05), abs=1e-10)


def test_vectorised_over_strikes():
    strikes = np.array([80.0, 100.0, 120.0])
    prices = black_scholes_price(100.0, strikes, 1.0, 0.03, 0.25, CALL)
    assert prices.shape == (3,)
    assert np.all(np.diff(prices) < 0)               # calls cheapen as K rises


def test_rejects_bad_inputs():
    with pytest.raises(ValueError):
        black_scholes_price(-100, 100, 1.0, 0.05, 0.2, CALL)
    with pytest.raises(ValueError):
        black_scholes_price(100, 100, -1.0, 0.05, 0.2, CALL)
    with pytest.raises(ValueError):
        black_scholes_price(100, 100, 1.0, 0.05, 0.2, "straddle")


# --------------------------------------------------------------------------- #
# Binomial tree / American exercise
# --------------------------------------------------------------------------- #
def test_binomial_european_converges_to_black_scholes():
    s, k, t, r, v = 100.0, 105.0, 1.0, 0.04, 0.25
    exact = black_scholes_price(s, k, t, r, v, CALL)
    coarse = european_binomial_price(s, k, t, r, v, CALL, steps=25)
    fine = european_binomial_price(s, k, t, r, v, CALL, steps=2000)
    assert abs(fine - exact) < abs(coarse - exact)   # more steps, less error
    assert fine == pytest.approx(exact, abs=5e-3)


def test_american_call_without_dividends_equals_european():
    # Never optimal to exercise a call early with no carry to give up, so the
    # early-exercise premium must vanish. Free check on the whole tree.
    premium = early_exercise_premium(100, 95, 1.0, 0.05, 0.3, CALL,
                                     dividend_yield=0.0, steps=400)
    assert premium == pytest.approx(0.0, abs=1e-9)


def test_american_put_is_worth_more_than_european():
    args = dict(spot=90.0, strike=110.0, time_to_expiry=1.0, rate=0.06,
                vol=0.25, option_type=PUT, steps=400)
    amer = binomial_price(american=True, **args)
    euro = binomial_price(american=False, **args)
    assert amer > euro                               # early exercise has value
    assert amer >= 110.0 - 90.0 - 1e-9               # and at least intrinsic


def test_american_call_on_dividend_payer_can_exceed_european():
    premium = early_exercise_premium(120, 100, 1.0, 0.02, 0.25, CALL,
                                     dividend_yield=0.08, steps=400)
    assert premium > 0                               # fat dividend, early exercise bites


def test_binomial_rejects_unstable_tree():
    # Tiny vol against a large drift pushes the risk-neutral probability out of
    # (0,1). Raising beats returning a plausible-looking wrong number.
    with pytest.raises(ValueError, match="risk-neutral probability"):
        binomial_price(100, 100, 1.0, 0.5, 0.001, CALL, steps=5)


# --------------------------------------------------------------------------- #
# Greeks: analytic vs finite difference
# --------------------------------------------------------------------------- #
GREEK_GRID = [
    (spot, strike, t, opt)
    for spot in (100.0,)
    for strike in (80.0, 95.0, 100.0, 105.0, 130.0)
    for t in (0.08, 0.5, 2.0)
    for opt in (CALL, PUT)
]


@pytest.mark.parametrize("spot,strike,t,opt", GREEK_GRID)
def test_analytic_and_fd_greeks_agree(spot, strike, t, opt):
    r, v, q = 0.04, 0.28, 0.01
    analytic = black_scholes_greeks(spot, strike, t, r, v, opt, q)
    numeric = black_scholes_fd_greeks(spot, strike, t, r, v, opt, q)

    for name in ("delta", "gamma", "vega", "theta", "rho"):
        a = getattr(analytic, name)
        n = getattr(numeric, name)
        assert a == pytest.approx(n, rel=1e-4, abs=1e-5), (
            f"{name} mismatch at K={strike} T={t} {opt}: analytic={a:.8f} fd={n:.8f}"
        )


def test_greek_signs_and_relationships():
    call = black_scholes_greeks(100, 100, 1.0, 0.04, 0.25, CALL)
    put = black_scholes_greeks(100, 100, 1.0, 0.04, 0.25, PUT)

    assert 0 < call.delta < 1
    assert -1 < put.delta < 0
    assert call.delta - put.delta == pytest.approx(math.exp(0.0), abs=1e-9)
    assert call.gamma > 0 and put.gamma > 0
    assert call.gamma == pytest.approx(put.gamma, rel=1e-12)   # same for both
    assert call.vega == pytest.approx(put.vega, rel=1e-12)
    assert call.theta < 0                                       # long option decays
    assert call.rho > 0 and put.rho < 0


def test_greeks_degenerate_at_expiry():
    g = black_scholes_greeks(120, 100, 0.0, 0.04, 0.25, CALL)
    assert g.price == pytest.approx(20.0)
    assert g.gamma == 0.0 and g.vega == 0.0 and g.theta == 0.0
    assert g.delta == pytest.approx(1.0)             # deep ITM at expiry


def test_trader_unit_conversion():
    g = black_scholes_greeks(100, 100, 1.0, 0.04, 0.25, CALL)
    t = g.as_trader_units()
    assert t["vega_per_point"] == pytest.approx(g.vega / 100.0)
    assert t["theta_per_day"] == pytest.approx(g.theta / 365.0)


def test_american_greeks_are_sane():
    # No closed form here, so only structural checks: an American put's delta
    # sits in [-1, 0] and its gamma is positive.
    g = american_greeks(95, 100, 0.5, 0.05, 0.3, PUT, steps=200)
    assert -1.0 <= g.delta <= 0.0
    assert g.gamma > 0
    assert g.vega > 0


# --------------------------------------------------------------------------- #
# Implied volatility
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("strike", [60.0, 85.0, 100.0, 115.0, 160.0])
@pytest.mark.parametrize("true_vol", [0.30, 0.85])
def test_implied_vol_round_trip(strike, true_vol):
    s, t, r, q = 100.0, 0.4, 0.03, 0.01
    price = black_scholes_price(s, strike, t, r, true_vol, CALL, q)
    recovered = implied_volatility(price, s, strike, t, r, CALL, q)
    assert recovered == pytest.approx(true_vol, abs=1e-6)


@pytest.mark.parametrize("strike", [60.0, 160.0])
def test_implied_vol_in_degenerate_wings_is_graceful(strike):
    # At 12% vol with 5 months to run, a 60-strike call is all intrinsic and a
    # 160-strike call is worth ~1e-9. The volatility information survives only
    # in the last few bits of a double, so demanding 1e-6 accuracy here would be
    # demanding more than the input carries. What we do require is that the
    # solver never invents a confident wrong answer: it either recovers the vol
    # to a loose tolerance or declines with a documented reason.
    s, t, r, q, true_vol = 100.0, 0.4, 0.03, 0.01, 0.12
    price = black_scholes_price(s, strike, t, r, true_vol, CALL, q)
    res = implied_volatility_detailed(price, s, strike, t, r, CALL, q)
    if res.ok:
        assert res.vol == pytest.approx(true_vol, abs=1e-3)
    else:
        assert res.reason in (REASON_AT_INTRINSIC, REASON_BELOW_INTRINSIC)
        assert math.isnan(res.vol)


def test_deep_itm_solves_via_put_call_parity():
    # A deep ITM call is nearly all intrinsic. Parity routes the solve through
    # the OTM put, which is pure time value, so the vol still comes back.
    s, k, t, r, q, true_vol = 100.0, 55.0, 0.5, 0.03, 0.0, 0.35
    price = black_scholes_price(s, k, t, r, true_vol, CALL, q)
    intrinsic, _ = no_arbitrage_bounds(s, k, t, r, CALL, q)
    assert (price - intrinsic) / price < 1e-3        # genuinely dominated by intrinsic
    assert implied_volatility(price, s, k, t, r, CALL, q) == pytest.approx(
        true_vol, abs=1e-4)


def test_implied_vol_round_trip_deep_itm_put():
    # Deep ITM: price is dominated by intrinsic and vega is small, the hardest
    # regime for a solver.
    s, k, t, r, true_vol = 100.0, 180.0, 0.25, 0.03, 0.45
    price = black_scholes_price(s, k, t, r, true_vol, PUT)
    assert implied_volatility(price, s, k, t, r, PUT) == pytest.approx(true_vol, abs=1e-5)


def test_implied_vol_rejects_arbitrage_violating_prices():
    s, k, t, r = 100.0, 100.0, 1.0, 0.05
    intrinsic, cap = no_arbitrage_bounds(s, k, t, r, CALL)

    below = implied_volatility_detailed(intrinsic - 1.0, s, k, t, r, CALL)
    assert not below.ok and below.reason == REASON_BELOW_INTRINSIC
    assert math.isnan(below.vol)

    above = implied_volatility_detailed(cap + 1.0, s, k, t, r, CALL)
    assert not above.ok and above.reason == REASON_ABOVE_MAX


def test_implied_vol_reports_expiry():
    res = implied_volatility_detailed(5.0, 100, 100, 0.0, 0.05, CALL)
    assert not res.ok and res.reason == REASON_EXPIRED


def test_implied_vol_widens_bracket_for_extreme_prices():
    # A price implying vol above the default upper bracket should still solve.
    s, k, t, r, true_vol = 100.0, 100.0, 1.0, 0.02, 6.5
    price = black_scholes_price(s, k, t, r, true_vol, CALL)
    res = implied_volatility_detailed(price, s, k, t, r, CALL, vol_upper=1.0)
    assert res.ok and res.reason == REASON_OK
    assert res.vol == pytest.approx(true_vol, abs=1e-4)
    assert res.iterations > 0                        # it had to widen
