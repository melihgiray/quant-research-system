"""Single-stock options: pricing, Greeks, implied volatility.

Phase 1 of the options leg. Black-Scholes for Europeans, a CRR binomial tree for
American early exercise, analytic Greeks cross-checked against finite
differences, and an implied-vol solver that says when it cannot produce an
answer instead of guessing one.

Nothing here trades. It is the pricing foundation the vol surface and the
options backtests build on.
"""

from .pricing import (
    CALL,
    PUT,
    black_scholes_price,
    binomial_price,
    american_price,
    european_binomial_price,
    early_exercise_premium,
)
from .greeks import (
    Greeks,
    black_scholes_greeks,
    black_scholes_fd_greeks,
    finite_difference_greeks,
    american_greeks,
)
from .implied_vol import (
    ImpliedVolResult,
    implied_volatility,
    implied_volatility_detailed,
    no_arbitrage_bounds,
)

__all__ = [
    "CALL",
    "PUT",
    "black_scholes_price",
    "binomial_price",
    "american_price",
    "european_binomial_price",
    "early_exercise_premium",
    "Greeks",
    "black_scholes_greeks",
    "black_scholes_fd_greeks",
    "finite_difference_greeks",
    "american_greeks",
    "ImpliedVolResult",
    "implied_volatility",
    "implied_volatility_detailed",
    "no_arbitrage_bounds",
]
