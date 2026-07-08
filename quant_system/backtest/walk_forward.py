"""Walk-forward validation.

A single train/test split is a toy: it reports one lucky (or unlucky) draw and
invites overfitting to that one test set. Walk-forward instead slides the
evaluation forward through time:

    |--- in-sample (>=252d) ---|-- OOS 63d --|
              |--- in-sample (expanded) ----|-- OOS 63d --|
                        |--- in-sample ------------|-- OOS 63d --|

Each fold fits/forms the strategy on data ending at the in-sample boundary, then
evaluates on the *next, unseen* quarter. We concatenate ONLY the out-of-sample
quarters into one continuous equity curve. Because consecutive OOS windows tile
the timeline without overlap, that curve is a fair, fully out-of-sample track
record - never reusing a day for both fitting and evaluation.

The strategy is supplied as a callback ``make_weights(price_data, fit_end)`` that
returns a full daily weight matrix. For non-parametric strategies (momentum,
pairs) ``fit_end`` is ignored. For the ML signal it refits using only data up to
``fit_end`` and predicts beyond it - so the refit itself is walk-forward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from ..config import CostConfig, WalkForwardConfig
from .engine import run_backtest, BacktestResult


MakeWeights = Callable[[object, Optional[pd.Timestamp]], pd.DataFrame]


@dataclass
class WalkForwardResult:
    """Concatenated out-of-sample track record plus per-fold diagnostics."""

    returns: pd.Series              # concatenated OOS net returns
    turnover: pd.Series             # concatenated OOS turnover
    costs: pd.Series                # concatenated OOS costs
    equity: pd.Series               # equity curve of the OOS returns
    folds: List[dict] = field(default_factory=list)

    @property
    def n_folds(self) -> int:
        return len(self.folds)

    @property
    def oos_span(self) -> str:
        if self.returns.empty:
            return "(empty)"
        return f"{self.returns.index.min().date()} -> {self.returns.index.max().date()}"


def _make_folds(n: int, wf: WalkForwardConfig) -> List[tuple]:
    """Yield (train_start, train_end, oos_start, oos_end) integer positions.

    OOS windows tile the timeline with no overlap (advance == out_sample), so the
    concatenated OOS curve never double-counts a day.
    """
    folds = []
    train_end = wf.in_sample
    while train_end + wf.out_sample <= n:
        train_start = 0 if wf.expanding else max(0, train_end - wf.in_sample)
        oos_start, oos_end = train_end, train_end + wf.out_sample
        folds.append((train_start, train_end, oos_start, oos_end))
        train_end += wf.out_sample
    return folds


def walk_forward(
    price_data,
    make_weights: MakeWeights,
    wf: Optional[WalkForwardConfig] = None,
    cost: Optional[CostConfig] = None,
    verbose: bool = True,
) -> WalkForwardResult:
    """Run expanding/rolling walk-forward and return the concatenated OOS record.

    Parameters
    ----------
    price_data : PriceData
        Full aligned panel.
    make_weights : callable(price_data, fit_end_timestamp) -> DataFrame
        Produces full-length daily target weights. May refit per fold using only
        data up to ``fit_end``.
    wf : WalkForwardConfig
        Window sizes.
    cost : CostConfig
        Transaction-cost assumptions used in each fold.

    Returns
    -------
    WalkForwardResult
    """
    wf = wf or WalkForwardConfig()
    cost = cost or CostConfig()
    index = price_data.close.index
    n = len(index)

    folds = _make_folds(n, wf)
    if not folds:
        raise ValueError(
            f"not enough data ({n} days) for walk-forward with in_sample={wf.in_sample}, "
            f"out_sample={wf.out_sample}"
        )

    oos_returns, oos_turnover, oos_costs = [], [], []
    fold_meta = []

    for k, (ts, te, os_, oe) in enumerate(folds):
        fit_end = index[te - 1]
        oos_dates = index[os_:oe]

        # Strategy may refit using only data up to fit_end; it returns full-length
        # weights but we evaluate only the OOS slice.
        weights = make_weights(price_data, fit_end)

        # Run the engine on the full panel (so rolling cost stats have history),
        # then keep only the OOS slice - that slice used no future information.
        res: BacktestResult = run_backtest(weights, price_data, cost=cost, check_lookahead=True)

        seg_ret = res.returns.loc[oos_dates]
        seg_to = res.turnover.loc[oos_dates]
        seg_cost = res.costs.loc[oos_dates]

        oos_returns.append(seg_ret)
        oos_turnover.append(seg_to)
        oos_costs.append(seg_cost)

        ann = np.sqrt(252) * seg_ret.mean() / seg_ret.std() if seg_ret.std() > 0 else float("nan")
        fold_meta.append({
            "fold": k,
            "train": f"{index[ts].date()}..{fit_end.date()}",
            "oos": f"{oos_dates.min().date()}..{oos_dates.max().date()}",
            "oos_sharpe": float(ann),
            "oos_return": float((1 + seg_ret).prod() - 1),
        })
        if verbose:
            print(f"[wf] fold {k:2d} | train {index[ts].date()}..{fit_end.date()} "
                  f"| OOS {oos_dates.min().date()}..{oos_dates.max().date()} "
                  f"| OOS Sharpe {ann:5.2f}")

    returns = pd.concat(oos_returns).sort_index()
    returns = returns[~returns.index.duplicated(keep="first")]
    turnover = pd.concat(oos_turnover).sort_index()
    turnover = turnover[~turnover.index.duplicated(keep="first")]
    costs = pd.concat(oos_costs).sort_index()
    costs = costs[~costs.index.duplicated(keep="first")]
    equity = (1.0 + returns.fillna(0.0)).cumprod()

    return WalkForwardResult(
        returns=returns,
        turnover=turnover,
        costs=costs,
        equity=equity,
        folds=fold_meta,
    )
