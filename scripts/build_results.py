"""Build the README results: walk-forward metrics table + OOS equity chart.

Runs the same walk-forward the CLI runs, on real yfinance data (no synthetic
fallback: if the download fails, this script aborts rather than publishing
made-up numbers). Writes docs/results/equity_oos.png and prints a markdown
table to paste into the README.

Also runs the pairs strategy both ways (per-fold causal selection vs the old
full-sample selection) so the changelog can state honestly what the fix did to
the results.

Usage:  python scripts/build_results.py
"""

import os
import sys

# Make the script runnable as `python scripts/build_results.py` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from quant_system.config import default_config
from quant_system.data.loader import load_price_data
from quant_system.data.universe import universe, FACTOR_ETFS, PAIRS_CANDIDATES
from quant_system.backtest.walk_forward import walk_forward
from quant_system.regime.detector import detect_regime
from quant_system.regime.switcher import apply_regime_sizing
from quant_system.signals.momentum import cross_sectional_momentum
from quant_system.signals.mean_reversion import (
    causal_pairs_weights, find_cointegrated_pair, pairs_signal,
)
from quant_system.signals.ml_signal import train_predict
from quant_system.performance.analytics import compute_metrics

OUT_DIR = "docs/results"


def main() -> int:
    cfg = default_config()
    tickers = sorted(set(
        universe("all")
        + list(dict.fromkeys(FACTOR_ETFS.values()))
        + [t for pair in PAIRS_CANDIDATES for t in pair]
    ))
    print(f"[results] loading {len(tickers)} tickers {cfg.start}..{cfg.end} (real data)")
    pdat = load_price_data(tickers, cfg.start, cfg.end, cache_dir=cfg.cache_dir,
                           allow_synthetic_fallback=False)
    if pdat.synthetic:
        print("[results] refusing to build results from synthetic data")
        return 1
    print(f"[results] panel: {pdat.close.shape[0]} days x {pdat.close.shape[1]} tickers")

    regime = detect_regime(pdat, cfg.regime, seed=cfg.random_seed,
                           benchmark=FACTOR_ETFS["market"])

    def momentum_weights(pdata, fit_end):
        w = cross_sectional_momentum(pdata, cfg.momentum)
        return apply_regime_sizing(w, regime.causal_labels, cfg.regime.defensive_scale)

    def pairs_weights(pdata, fit_end):
        return causal_pairs_weights(pdata, fit_end, PAIRS_CANDIDATES, cfg.pairs)

    def pairs_weights_full_sample(pdata, fit_end):
        # The old behaviour, kept only for the honest before/after comparison.
        best = find_cointegrated_pair(pdata.close, PAIRS_CANDIDATES,
                                      cfg.pairs.coint_pvalue_max,
                                      fdr_alpha=cfg.pairs.coint_pvalue_max)
        if best is None:
            return pd.DataFrame(0.0, index=pdata.close.index, columns=pdata.close.columns)
        return pairs_signal(pdata, (best[0], best[1]), cfg.pairs)

    def ml_weights(pdata, fit_end):
        w = train_predict(pdata, fit_end, cfg.ml, max_weight=cfg.risk.max_weight)
        return apply_regime_sizing(w, regime.causal_labels, cfg.regime.defensive_scale)

    runs = [
        ("Cross-sectional momentum", pdat.subset(universe("sectors")), momentum_weights),
        ("Pairs (per-fold selection)", pdat, pairs_weights),
        ("Pairs (old full-sample selection)", pdat, pairs_weights_full_sample),
        ("ML directional", pdat.subset(universe("largecaps")), ml_weights),
    ]

    results = {}
    for label, panel, make in runs:
        print(f"[results] walk-forward: {label}")
        wf = walk_forward(panel, make, cfg.walk_forward, cfg.cost, verbose=False)
        m = compute_metrics(wf.returns, turnover=wf.turnover)
        results[label] = {"wf": wf, "metrics": m}
        print(f"          Sharpe {m['sharpe']:+.2f}  maxDD {m['max_drawdown']:+.1%}  "
              f"turnover {m['ann_turnover']:.1f}x  ({wf.oos_span})")

    # ---- markdown table ----
    span = results["Cross-sectional momentum"]["wf"].oos_span
    print("\n--- paste into README ---")
    print(f"Out-of-sample walk-forward, {span}, net of costs.\n")
    print("| Strategy | OOS Sharpe | Ann. return | Max drawdown | Turnover |")
    print("|---|---|---|---|---|")
    for label in ("Cross-sectional momentum", "Pairs (per-fold selection)", "ML directional"):
        m = results[label]["metrics"]
        print(f"| {label} | {m['sharpe']:.2f} | {m['ann_return']:+.1%} "
              f"| {m['max_drawdown']:.1%} | {m['ann_turnover']:.1f}x |")

    old = results["Pairs (old full-sample selection)"]["metrics"]
    new = results["Pairs (per-fold selection)"]["metrics"]
    print(f"\npairs comparison: old Sharpe {old['sharpe']:+.2f} -> new {new['sharpe']:+.2f}; "
          f"old ann ret {old['ann_return']:+.1%} -> new {new['ann_return']:+.1%}")

    # ---- equity chart ----
    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for label, color in (("Cross-sectional momentum", "#1f77b4"),
                         ("Pairs (per-fold selection)", "#2ca02c"),
                         ("ML directional", "#d62728")):
        eq = results[label]["wf"].equity
        ax.plot(eq.index, eq.values, lw=1.3, label=label, color=color)
    ax.axhline(1.0, color="grey", lw=0.7)
    ax.set_title(f"Out-of-sample equity, walk-forward, net of costs ({span})")
    ax.set_ylabel("Growth of $1")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = f"{OUT_DIR}/equity_oos.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"\n[results] wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
