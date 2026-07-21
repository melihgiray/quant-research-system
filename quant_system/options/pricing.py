"""Option pricing: Black-Scholes for Europeans, CRR binomial for Americans.

Two pricers, because they answer different questions.

Black-Scholes has a closed form, so it is fast and exact within its own
assumptions, and its Greeks are analytic. It cannot price early exercise.

The Cox-Ross-Rubinstein binomial tree handles early exercise by checking, at
every node, whether exercising beats holding. That matters for American puts
(always worth something to exercise early when deep in the money and rates are
positive) and for American calls on a dividend-paying underlying. With no
dividends an American call is never worth exercising early, so it collapses to
the European price. That identity is one of the unit tests: it is a free
correctness check on the whole tree.

Both pricers handle the degenerate corners the same way. At expiry, or at zero
volatility, an option is worth its discounted forward intrinsic and nothing
more, so the formulas fall back to ``max(forward intrinsic, 0)`` rather than
dividing by zero.
"""

from __future__ import annotations

import logging
from typing import Union

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)

ArrayLike = Union[float, np.ndarray]

CALL = "call"
PUT = "put"


def _validate_option_type(option_type: str) -> str:
    opt = str(option_type).strip().lower()
    if opt not in (CALL, PUT):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    return opt


def _d1_d2(spot, strike, time_to_expiry, rate, vol, dividend_yield):
    """d1 and d2, computed on a guarded time/vol so the corners do not blow up."""
    live = (time_to_expiry > 0) & (vol > 0)
    t_safe = np.where(live, time_to_expiry, 1.0)
    v_safe = np.where(live, vol, 1.0)
    sqrt_t = np.sqrt(t_safe)
    d1 = ((np.log(spot / strike) + (rate - dividend_yield + 0.5 * v_safe ** 2) * t_safe)
          / (v_safe * sqrt_t))
    d2 = d1 - v_safe * sqrt_t
    return d1, d2, live


def _forward_intrinsic(spot, strike, time_to_expiry, rate, dividend_yield, opt: str):
    """Value when there is no optionality left: discounted forward intrinsic."""
    disc_r = np.exp(-rate * time_to_expiry)
    disc_q = np.exp(-dividend_yield * time_to_expiry)
    if opt == CALL:
        return np.maximum(spot * disc_q - strike * disc_r, 0.0)
    return np.maximum(strike * disc_r - spot * disc_q, 0.0)


def black_scholes_price(
    spot: ArrayLike,
    strike: ArrayLike,
    time_to_expiry: ArrayLike,
    rate: ArrayLike,
    vol: ArrayLike,
    option_type: str = CALL,
    dividend_yield: ArrayLike = 0.0,
) -> ArrayLike:
    """Black-Scholes-Merton price of a European option.

    Parameters
    ----------
    spot : float or array
        Underlying price. Must be positive.
    strike : float or array
        Strike. Must be positive.
    time_to_expiry : float or array
        Years to expiry. Zero returns intrinsic value.
    rate : float or array
        Continuously compounded annual risk-free rate.
    vol : float or array
        Annualised volatility as a decimal (0.25 = 25%). Zero returns the
        discounted forward intrinsic.
    option_type : {"call", "put"}
    dividend_yield : float or array
        Continuous annual dividend yield.

    Returns
    -------
    float or array
        Option price, matching the broadcast shape of the inputs.
    """
    opt = _validate_option_type(option_type)
    s = np.asarray(spot, dtype=float)
    k = np.asarray(strike, dtype=float)
    t = np.asarray(time_to_expiry, dtype=float)
    r = np.asarray(rate, dtype=float)
    v = np.asarray(vol, dtype=float)
    q = np.asarray(dividend_yield, dtype=float)

    if np.any(s <= 0) or np.any(k <= 0):
        raise ValueError("spot and strike must be positive")
    if np.any(t < 0):
        raise ValueError("time_to_expiry must be non-negative")
    if np.any(v < 0):
        raise ValueError("vol must be non-negative")

    scalar = all(np.ndim(x) == 0 for x in (s, k, t, r, v, q))

    d1, d2, live = _d1_d2(s, k, t, r, v, q)
    disc_r = np.exp(-r * t)
    disc_q = np.exp(-q * t)

    if opt == CALL:
        main = s * disc_q * norm.cdf(d1) - k * disc_r * norm.cdf(d2)
    else:
        main = k * disc_r * norm.cdf(-d2) - s * disc_q * norm.cdf(-d1)

    limit = _forward_intrinsic(s, k, t, r, q, opt)
    out = np.where(live, main, limit)
    return float(out) if scalar else out


def binomial_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float,
    vol: float,
    option_type: str = CALL,
    dividend_yield: float = 0.0,
    steps: int = 500,
    american: bool = True,
) -> float:
    """Cox-Ross-Rubinstein binomial price. Scalar inputs only.

    The tree uses the standard CRR parameterisation ``u = exp(sigma*sqrt(dt))``,
    ``d = 1/u``, with risk-neutral probability
    ``p = (exp((r-q)*dt) - d) / (u - d)``. Backward induction discounts one step
    at a time; when ``american`` is set, each node takes the max of continuation
    value and immediate exercise.

    Parameters
    ----------
    steps : int
        Number of time steps. Error decays roughly as 1/steps, with the
        characteristic even/odd oscillation of binomial trees.
    american : bool
        True for American exercise, False for European (useful for testing
        convergence against the closed form).

    Raises
    ------
    ValueError
        If the risk-neutral probability falls outside (0, 1), which means the
        tree cannot represent an arbitrage-free process at this step size.
        That happens when volatility is tiny relative to the drift; raising is
        better than silently returning a nonsense number.
    """
    opt = _validate_option_type(option_type)
    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive")
    if steps < 1:
        raise ValueError("steps must be at least 1")
    if time_to_expiry <= 0 or vol <= 0:
        return float(_forward_intrinsic(
            np.asarray(float(spot)), np.asarray(float(strike)),
            np.asarray(float(max(time_to_expiry, 0.0))),
            np.asarray(float(rate)), np.asarray(float(dividend_yield)), opt))

    dt = time_to_expiry / steps
    up = np.exp(vol * np.sqrt(dt))
    down = 1.0 / up
    disc = np.exp(-rate * dt)
    p = (np.exp((rate - dividend_yield) * dt) - down) / (up - down)
    if not (0.0 < p < 1.0):
        raise ValueError(
            f"risk-neutral probability {p:.4f} outside (0,1): the tree is "
            f"unstable at vol={vol:.4f}, dt={dt:.6f}. Use more steps or check inputs."
        )

    # Terminal payoffs across the steps+1 leaf nodes.
    ups = np.arange(steps + 1)
    terminal_spot = spot * up ** ups * down ** (steps - ups)
    if opt == CALL:
        values = np.maximum(terminal_spot - strike, 0.0)
    else:
        values = np.maximum(strike - terminal_spot, 0.0)

    for i in range(steps, 0, -1):
        values = disc * (p * values[1:i + 1] + (1.0 - p) * values[0:i])
        if american:
            ups_i = np.arange(i)
            node_spot = spot * up ** ups_i * down ** (i - 1 - ups_i)
            if opt == CALL:
                exercise = np.maximum(node_spot - strike, 0.0)
            else:
                exercise = np.maximum(strike - node_spot, 0.0)
            values = np.maximum(values, exercise)

    return float(values[0])


def american_price(spot, strike, time_to_expiry, rate, vol,
                   option_type: str = CALL, dividend_yield: float = 0.0,
                   steps: int = 500) -> float:
    """American option price (CRR tree with early exercise)."""
    return binomial_price(spot, strike, time_to_expiry, rate, vol, option_type,
                          dividend_yield, steps, american=True)


def european_binomial_price(spot, strike, time_to_expiry, rate, vol,
                            option_type: str = CALL, dividend_yield: float = 0.0,
                            steps: int = 500) -> float:
    """European price from the same tree, for convergence checks against BS."""
    return binomial_price(spot, strike, time_to_expiry, rate, vol, option_type,
                          dividend_yield, steps, american=False)


def early_exercise_premium(spot, strike, time_to_expiry, rate, vol,
                           option_type: str = PUT, dividend_yield: float = 0.0,
                           steps: int = 500) -> float:
    """American price minus European price: what the early-exercise right is worth.

    Non-negative by construction (an American option can always be held to
    expiry). For a call on a non-dividend payer it should be essentially zero.
    """
    american = binomial_price(spot, strike, time_to_expiry, rate, vol,
                              option_type, dividend_yield, steps, american=True)
    european = binomial_price(spot, strike, time_to_expiry, rate, vol,
                              option_type, dividend_yield, steps, american=False)
    return american - european
