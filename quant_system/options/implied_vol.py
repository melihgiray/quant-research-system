"""Implied volatility by Brent root-finding, with the failure cases made explicit.

Inverting Black-Scholes for volatility is easy in the middle of the surface and
genuinely hard at the edges. Vega goes to zero for deep in- and out-of-the-money
options, so a price that is a fraction of a cent away from another price can
imply a wildly different volatility. Any solver will return *a* number there;
the honest thing is to say when that number is meaningless.

So this module does three things before it solves:

1. Checks the price against static no-arbitrage bounds. A European call must sit
   between its discounted forward intrinsic and the discounted spot. A price
   outside that band has no implied volatility at all, and quoting one would be
   inventing data. Crossed or stale quotes hit this constantly in real chains.
2. Checks whether the price sits *at* a bound, where implied vol is zero or
   unbounded rather than merely extreme.
3. Widens the upper bracket if the market price exceeds what the bracket's top
   volatility can produce, up to a hard ceiling.

Only then does it call Brent, which is safe once the root is bracketed because
it cannot leave the interval.

Failures return NaN with a machine-readable reason rather than raising, because
a single bad strike should not abort a whole surface build. The reasons get
logged and, in the surface code, counted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import brentq

from .pricing import CALL, PUT, _validate_option_type, black_scholes_price

logger = logging.getLogger(__name__)

# Reasons an implied vol cannot be produced. Stable strings, safe to count on.
REASON_OK = "ok"
REASON_BELOW_INTRINSIC = "price_below_intrinsic"
REASON_ABOVE_MAX = "price_above_no_arb_max"
REASON_AT_INTRINSIC = "price_at_intrinsic"
REASON_EXPIRED = "expired_or_zero_time"
REASON_NO_BRACKET = "no_bracket_found"
REASON_SOLVER_FAILED = "solver_failed"


@dataclass(frozen=True)
class ImpliedVolResult:
    """Solved volatility plus why it did or did not work."""

    vol: float
    converged: bool
    reason: str
    iterations: int = 0

    @property
    def ok(self) -> bool:
        return self.converged and np.isfinite(self.vol)


def no_arbitrage_bounds(spot: float, strike: float, time_to_expiry: float,
                        rate: float, option_type: str = CALL,
                        dividend_yield: float = 0.0) -> tuple:
    """(lower, upper) static no-arbitrage price bounds for a European option.

    Lower bound is the discounted forward intrinsic, upper bound is the
    discounted spot for a call or the discounted strike for a put. A quote
    outside this band is not a mispricing to trade, it is bad data.
    """
    opt = _validate_option_type(option_type)
    disc_r = float(np.exp(-rate * time_to_expiry))
    disc_q = float(np.exp(-dividend_yield * time_to_expiry))
    if opt == CALL:
        return max(spot * disc_q - strike * disc_r, 0.0), spot * disc_q
    return max(strike * disc_r - spot * disc_q, 0.0), strike * disc_r


def implied_volatility_detailed(
    price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    option_type: str = CALL,
    dividend_yield: float = 0.0,
    vol_lower: float = 1e-6,
    vol_upper: float = 5.0,
    tolerance: float = 1e-8,
    max_upper: float = 50.0,
) -> ImpliedVolResult:
    """Solve for Black-Scholes implied volatility, reporting why on failure.

    Parameters
    ----------
    price : float
        Observed option price (mid, bid or ask, caller's choice).
    vol_lower, vol_upper : float
        Initial bracket. The upper end is widened geometrically up to
        ``max_upper`` if the market price is above what it can generate.
    tolerance : float
        Absolute tolerance on the returned volatility.

    Returns
    -------
    ImpliedVolResult
        ``vol`` is NaN whenever ``converged`` is False.
    """
    opt = _validate_option_type(option_type)
    nan = float("nan")

    if time_to_expiry <= 0:
        return ImpliedVolResult(nan, False, REASON_EXPIRED)
    if not np.isfinite(price) or price < 0:
        return ImpliedVolResult(nan, False, REASON_BELOW_INTRINSIC)

    # Deep in-the-money options are badly conditioned: nearly all of the price
    # is intrinsic, so the time value that actually carries the volatility
    # information is a rounding error on a large number. Put-call parity says
    # the out-of-the-money twin has the same implied vol while being entirely
    # time value, so solve there instead. This is standard desk practice and it
    # matters most for real quotes, where a $40.30 ITM call rounded to the cent
    # retains no usable time value at all.
    forward_pv = spot * float(np.exp(-dividend_yield * time_to_expiry))
    strike_pv = strike * float(np.exp(-rate * time_to_expiry))
    in_the_money = forward_pv > strike_pv if opt == CALL else strike_pv > forward_pv

    solve_price, solve_opt = price, opt
    if in_the_money:
        if opt == CALL:
            solve_price, solve_opt = price - forward_pv + strike_pv, PUT
        else:
            solve_price, solve_opt = price + forward_pv - strike_pv, CALL

    lower_bound, upper_bound = no_arbitrage_bounds(
        spot, strike, time_to_expiry, rate, solve_opt, dividend_yield)

    # Two different tolerances, because they answer different questions.
    # eps_arb is "is this quote outside the no-arbitrage band" and allows a
    # little slack for real quotes that round a hair outside. eps_zero is "is
    # there any time value at all", which is a floating-point question, so it
    # sits at machine precision. Using the loose one for both would throw away
    # perfectly solvable deep-wing options.
    eps_arb = max(1e-10, 1e-8 * max(spot, strike))
    eps_zero = 8.0 * float(np.finfo(float).eps) * max(abs(lower_bound),
                                                      abs(solve_price), 1.0)

    if solve_price < lower_bound - eps_arb:
        logger.debug("IV reject: price %.6g below intrinsic %.6g (K=%.2f)",
                     solve_price, lower_bound, strike)
        return ImpliedVolResult(nan, False, REASON_BELOW_INTRINSIC)
    if solve_price > upper_bound + eps_arb:
        logger.debug("IV reject: price %.6g above no-arb max %.6g (K=%.2f)",
                     solve_price, upper_bound, strike)
        return ImpliedVolResult(nan, False, REASON_ABOVE_MAX)
    if solve_price - lower_bound <= eps_zero:
        # Exactly on intrinsic: there is no time value to invert. Volatility is
        # zero or unresolvable, not "very small". Say so instead of returning
        # the bottom of the bracket and pretending it means something.
        return ImpliedVolResult(nan, False, REASON_AT_INTRINSIC)

    def objective(v: float) -> float:
        return float(black_scholes_price(spot, strike, time_to_expiry, rate, v,
                                         solve_opt, dividend_yield)) - solve_price

    f_low = objective(vol_lower)
    hi = vol_upper
    f_high = objective(hi)
    widenings = 0
    while f_high < 0 and hi < max_upper:
        hi = min(hi * 2.0, max_upper)
        f_high = objective(hi)
        widenings += 1

    if f_low * f_high > 0:
        logger.debug("IV reject: no bracket for price %.6f at K=%.2f "
                     "(f(%.g)=%.3e, f(%.2f)=%.3e)",
                     price, strike, vol_lower, f_low, hi, f_high)
        return ImpliedVolResult(nan, False, REASON_NO_BRACKET, widenings)

    try:
        vol = brentq(objective, vol_lower, hi, xtol=tolerance, maxiter=200)
    except (ValueError, RuntimeError) as exc:
        logger.debug("IV solver failed at K=%.2f: %s", strike, exc)
        return ImpliedVolResult(nan, False, REASON_SOLVER_FAILED, widenings)

    return ImpliedVolResult(float(vol), True, REASON_OK, widenings)


def implied_volatility(price, spot, strike, time_to_expiry, rate,
                       option_type: str = CALL, dividend_yield: float = 0.0,
                       **kwargs) -> float:
    """Implied volatility, or NaN if it cannot be solved. See the detailed form."""
    return implied_volatility_detailed(price, spot, strike, time_to_expiry, rate,
                                       option_type, dividend_yield, **kwargs).vol
