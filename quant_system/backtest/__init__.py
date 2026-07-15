"""Backtest engine, transaction-cost models, and walk-forward validation."""

from .engine import BacktestResult, run_backtest, assert_no_lookahead, portfolio_returns
from .costs import transaction_cost_fraction, square_root_impact, half_spread_cost
from .walk_forward import walk_forward, WalkForwardResult
from .capacity import sweep_capital, estimate_capacity, plot_capacity, capacity_summary
from .execution import apply_execution, constrained_holdings, participation_cap_weights

__all__ = [
    "BacktestResult",
    "run_backtest",
    "assert_no_lookahead",
    "portfolio_returns",
    "transaction_cost_fraction",
    "square_root_impact",
    "half_spread_cost",
    "walk_forward",
    "WalkForwardResult",
    "sweep_capital",
    "estimate_capacity",
    "plot_capacity",
    "capacity_summary",
    "apply_execution",
    "constrained_holdings",
    "participation_cap_weights",
]
