"""Stress scenarios: what a market shock would do to the strategy.

Rather than hard-code historical crash magnitudes (which invites stale or
mis-remembered numbers), the default scenarios are read from the benchmark's own
history: its worst single day, week and month actually observed in the sample.
Each shock is passed through the strategy's beta to the benchmark to estimate the
strategy's loss. That is a first-order, linear approximation: it assumes beta
holds in the tail, which is exactly when correlations tend to rise, so treat the
estimate as a floor on the pain, not a precise figure. Custom named scenarios can
be supplied when you want to test a specific hypothetical move.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

_WINDOW_LABELS = {1: "Worst market day", 5: "Worst market week", 21: "Worst market month"}


def strategy_beta(returns: pd.Series, benchmark: pd.Series) -> float:
    """Beta of the strategy to the benchmark over their shared dates."""
    joined = pd.concat([returns.rename("r"), benchmark.rename("b")], axis=1).dropna()
    var = joined["b"].var()
    if var == 0 or joined.empty:
        return float("nan")
    return float(joined["r"].cov(joined["b"]) / var)


def historical_worst_moves(benchmark: pd.Series,
                           windows: Sequence[int] = (1, 5, 21)) -> Dict[int, float]:
    """Worst cumulative benchmark return over each rolling window in the sample."""
    b = benchmark.dropna()
    out: Dict[int, float] = {}
    for w in windows:
        cum = b if w == 1 else (1.0 + b).rolling(w).apply(np.prod, raw=True) - 1.0
        out[w] = float(cum.min())
    return out


def stress_test(returns: pd.Series,
                benchmark: pd.Series,
                windows: Sequence[int] = (1, 5, 21),
                custom: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """Estimated strategy P&L under worst-observed and custom market shocks.

    Returns a table with the scenario name, the benchmark move, the strategy's
    beta, and the beta-implied strategy P&L for each scenario.
    """
    beta = strategy_beta(returns, benchmark)
    moves = historical_worst_moves(benchmark, windows)
    rows = []
    for w in windows:
        move = moves[w]
        rows.append({"scenario": _WINDOW_LABELS.get(w, f"Worst {w}d"),
                     "market_move": move, "beta": beta, "est_pnl": beta * move})
    for name, move in (custom or {}).items():
        rows.append({"scenario": name, "market_move": float(move),
                     "beta": beta, "est_pnl": beta * float(move)})
    return pd.DataFrame(rows)
