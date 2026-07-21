"""Run the option strategies and write the results table + equity plot.

    python scripts/build_options_results.py

Everything here runs on a SYNTHETIC option chain, because yfinance provides no
options history and a real EOD chain history is a paid product. The variance
risk premium is an *input* to that chain, so the short-volatility results below
show that the machinery harvests a premium that was put there deliberately.
They are not evidence the premium exists in the market. The premium sweep in
the output makes the dependence explicit.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from quant_system.data.loader import load_price_data
from quant_system.options.backtest import run_options_backtest
from quant_system.options.provider import SyntheticChainProvider
from quant_system.options import strategies as S
from quant_system.performance.analytics import compute_metrics

OUT_DIR = "docs/results"
CAPITAL = 100_000.0
PREMIUM = 0.15


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    close = load_price_data(["AAPL"], "2018-01-01", "2023-12-31",
                            use_synthetic=True).close["AAPL"]
    provider = SyntheticChainProvider(underlying=close, vol_premium=PREMIUM)
    print("\n".join(provider.summary()))
    print()

    runs = [
        ("Buy and hold (benchmark)", S.buy_and_hold_underlying()),
        ("Covered call (30 delta)", S.covered_call()),
        ("Cash-secured put (30 delta)", S.cash_secured_put()),
        ("Delta-hedged short straddle", S.delta_hedged_short_straddle()),
    ]

    results = {}
    print("| Strategy | Return | Ann. | Sharpe | Max DD | Trades | Spread paid |")
    print("|---|---|---|---|---|---|---|")
    for label, strategy in runs:
        res = run_options_backtest(provider, strategy, initial_capital=CAPITAL,
                                   label=label)
        results[label] = res
        m = compute_metrics(res.returns.dropna())
        total = res.equity.iloc[-1] / CAPITAL - 1
        print(f"| {label} | {total:+.1%} | {m['ann_return']:+.2%} | "
              f"{m['sharpe']:.2f} | {m['max_drawdown']:.1%} | {res.n_trades} | "
              f"${res.spread_cost:,.0f} |")

    # The sweep that shows the answer is an assumption, not a finding.
    print("\nVariance premium sweep (delta-hedged short straddle):\n")
    print("| Implied over realised | Total return |")
    print("|---|---|")
    for premium in (0.0, 0.15, 0.30, 0.50):
        prov = SyntheticChainProvider(underlying=close, vol_premium=premium)
        res = run_options_backtest(prov, S.delta_hedged_short_straddle(),
                                   initial_capital=CAPITAL, label="sweep")
        print(f"| {premium:+.0%} | {res.equity.iloc[-1] / CAPITAL - 1:+.2%} |")

    os.makedirs(OUT_DIR, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    for (label, _), colour in zip(runs, ["#666666", "#1f77b4", "#2ca02c", "#d62728"]):
        eq = results[label].equity / CAPITAL
        ax1.plot(eq.index, eq.values, lw=1.3, label=label, color=colour)
    ax1.axhline(1.0, color="grey", lw=0.7)
    ax1.set_ylabel("Growth of $1")
    ax1.set_title(f"Option strategies on a SYNTHETIC chain "
                  f"(variance premium {PREMIUM:+.0%}), net of crossing the spread")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    straddle = results["Delta-hedged short straddle"]
    ax2.plot(straddle.greeks.index, straddle.greeks["delta"], lw=0.9,
             color="#d62728", label="net delta (hedged)")
    ax2.axhline(0.0, color="grey", lw=0.7)
    ax2.set_ylabel("net delta (shares)")
    ax2.set_xlabel("date")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    path = f"{OUT_DIR}/options_strategies.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"\n[options] wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
