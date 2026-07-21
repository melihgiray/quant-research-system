"""Greeks: analytic where Black-Scholes gives a closed form, finite-difference otherwise.

Every Greek here has a closed form under Black-Scholes, so the analytic path is
the one to use for European options. American options priced on a binomial tree
have no closed-form sensitivities, so those get central finite differences on
the tree instead.

Keeping both and testing them against each other is the point. The unit tests
assert that, for European options, the analytic and finite-difference Greeks
agree across a grid of moneyness and expiry. If the analytic formulas were
wrong the two would diverge; if the finite-difference machinery were wrong the
American Greeks would be silently garbage. Cross-checking catches both.

Units (stated once, since most Greek confusion is unit confusion):
    delta  per 1.00 of spot
    gamma  per 1.00 of spot, twice
    vega   per 1.00 of volatility   (divide by 100 for "per vol point")
    theta  per year                 (divide by 365 for "per calendar day")
    rho    per 1.00 of rate         (divide by 100 for "per 1% rate move")
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Dict

import numpy as np
from scipy.stats import norm

from .pricing import CALL, PUT, _d1_d2, _validate_option_type, black_scholes_price


@dataclass(frozen=True)
class Greeks:
    """Price and the five first/second-order sensitivities.

    See the module docstring for units. ``as_trader_units`` rescales vega,
    theta and rho into the per-point and per-day quotes desks actually use.
    """

    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)

    def as_trader_units(self) -> Dict[str, float]:
        """Vega per vol point, theta per calendar day, rho per 1% rate move."""
        return {
            "price": self.price,
            "delta": self.delta,
            "gamma": self.gamma,
            "vega_per_point": self.vega / 100.0,
            "theta_per_day": self.theta / 365.0,
            "rho_per_percent": self.rho / 100.0,
        }


def black_scholes_greeks(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    vol: float,
    option_type: str = CALL,
    dividend_yield: float = 0.0,
) -> Greeks:
    """Closed-form Black-Scholes Greeks for a European option.

    At expiry (or zero vol) the sensitivities are degenerate: gamma, vega and
    theta are zero, and delta is a step function. We return that limit rather
    than a division by zero.
    """
    opt = _validate_option_type(option_type)
    s = float(spot)
    k = float(strike)
    t = float(time_to_expiry)
    r = float(rate)
    v = float(vol)
    q = float(dividend_yield)

    price = float(black_scholes_price(s, k, t, r, v, opt, q))

    if t <= 0 or v <= 0:
        # No optionality left. Delta is the indicator of being in the money.
        if opt == CALL:
            delta = float(np.exp(-q * t)) if s > k else 0.0
        else:
            delta = -float(np.exp(-q * t)) if s < k else 0.0
        return Greeks(price=price, delta=delta, gamma=0.0, vega=0.0,
                      theta=0.0, rho=0.0)

    d1, d2, _ = _d1_d2(np.asarray(s), np.asarray(k), np.asarray(t),
                       np.asarray(r), np.asarray(v), np.asarray(q))
    d1 = float(d1)
    d2 = float(d2)
    sqrt_t = np.sqrt(t)
    disc_r = np.exp(-r * t)
    disc_q = np.exp(-q * t)
    pdf_d1 = float(norm.pdf(d1))

    gamma = float(disc_q * pdf_d1 / (s * v * sqrt_t))
    vega = float(s * disc_q * pdf_d1 * sqrt_t)
    # Common first term of theta: the time-decay of optionality itself.
    decay = -s * disc_q * pdf_d1 * v / (2.0 * sqrt_t)

    if opt == CALL:
        delta = float(disc_q * norm.cdf(d1))
        theta = float(decay - r * k * disc_r * norm.cdf(d2)
                      + q * s * disc_q * norm.cdf(d1))
        rho = float(k * t * disc_r * norm.cdf(d2))
    else:
        delta = float(-disc_q * norm.cdf(-d1))
        theta = float(decay + r * k * disc_r * norm.cdf(-d2)
                      - q * s * disc_q * norm.cdf(-d1))
        rho = float(-k * t * disc_r * norm.cdf(-d2))

    return Greeks(price=price, delta=delta, gamma=gamma, vega=vega,
                  theta=theta, rho=rho)


def finite_difference_greeks(
    price_fn: Callable[..., float],
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    vol: float,
    option_type: str = CALL,
    dividend_yield: float = 0.0,
    spot_bump: float = 1e-3,
    vol_bump: float = 1e-4,
    time_bump: float = 1e-5,
    rate_bump: float = 1e-5,
) -> Greeks:
    """Central-difference Greeks for any pricer.

    Works on Black-Scholes (where it is a cross-check) and on the binomial tree
    (where it is the only option, since American prices have no closed-form
    sensitivities).

    Parameters
    ----------
    price_fn : callable
        Signature ``(spot, strike, time_to_expiry, rate, vol, option_type,
        dividend_yield) -> float``.
    spot_bump : float
        Relative bump to spot. The rest are absolute.

    Notes
    -----
    Theta is ``-dV/dT``: an option loses value as expiry approaches, so the
    sign flip is deliberate, not a bug. The time bump is clipped so ``T - h``
    never goes negative near expiry; when that clipping happens the difference
    becomes one-sided and slightly less accurate, which is the right tradeoff
    against pricing a negative maturity.
    """
    opt = _validate_option_type(option_type)

    def price(s=spot, k=strike, t=time_to_expiry, r=rate, v=vol, q=dividend_yield):
        return float(price_fn(s, k, t, r, v, opt, q))

    base = price()

    h_s = max(spot * spot_bump, 1e-8)
    up = price(s=spot + h_s)
    down = price(s=spot - h_s)
    delta = (up - down) / (2.0 * h_s)
    gamma = (up - 2.0 * base + down) / (h_s ** 2)

    h_v = vol_bump
    vega = (price(v=vol + h_v) - price(v=max(vol - h_v, 0.0))) / (2.0 * h_v)

    # Keep T - h strictly positive; fall back to a one-sided difference at expiry.
    h_t = min(time_bump, max(time_to_expiry * 0.5, 0.0))
    if h_t > 0:
        theta = -(price(t=time_to_expiry + h_t) - price(t=time_to_expiry - h_t)) / (2.0 * h_t)
    else:
        theta = 0.0

    h_r = rate_bump
    rho = (price(r=rate + h_r) - price(r=rate - h_r)) / (2.0 * h_r)

    return Greeks(price=base, delta=delta, gamma=gamma, vega=vega,
                  theta=theta, rho=rho)


def black_scholes_fd_greeks(spot, strike, time_to_expiry, rate, vol,
                            option_type: str = CALL, dividend_yield: float = 0.0,
                            **bumps) -> Greeks:
    """Finite-difference Greeks on the Black-Scholes pricer (the cross-check)."""
    return finite_difference_greeks(
        lambda s, k, t, r, v, o, q: float(black_scholes_price(s, k, t, r, v, o, q)),
        spot, strike, time_to_expiry, rate, vol, option_type, dividend_yield,
        **bumps)


def american_greeks(spot, strike, time_to_expiry, rate, vol,
                    option_type: str = PUT, dividend_yield: float = 0.0,
                    steps: int = 300, **bumps) -> Greeks:
    """Finite-difference Greeks for an American option on the binomial tree.

    Uses fewer default steps than the pricer: finite differences call the tree
    nine times, and tree noise between neighbouring step counts is the limiting
    error anyway. Bump sizes are deliberately larger than the Black-Scholes
    defaults for the same reason, since the tree's own discretisation noise
    swamps a very small bump.
    """
    from .pricing import binomial_price

    bumps.setdefault("spot_bump", 1e-2)
    bumps.setdefault("vol_bump", 1e-3)
    bumps.setdefault("time_bump", 1e-3)
    bumps.setdefault("rate_bump", 1e-4)

    def tree(s, k, t, r, v, o, q):
        return binomial_price(s, k, t, r, v, o, q, steps=steps, american=True)

    return finite_difference_greeks(tree, spot, strike, time_to_expiry, rate,
                                    vol, option_type, dividend_yield, **bumps)
