"""Compare defensive sizing driven by the realized-vol regime vs the GARCH regime.

Both detectors call a "defensive" day and halve risk on it; they differ in how
they see the volatility. The vol-ratio detector compares trailing 21d to 252d
realized vol (backward-looking, slow). The GARCH detector uses a one-step-ahead
conditional-vol forecast versus its own trailing baseline (reacts to a shock the
day it happens). The question this answers: does forecasting the volatility beat
measuring it, for the purpose of turning risk down before the damage.

Runs the sector-momentum sleeve three ways through the same real-data walk-forward
(no regime, vol-ratio regime, GARCH regime) and prints a markdown row. Aborts on
synthetic data. Requires the `garch` extra.

Usage:  python scripts/build_regime_results.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quant_system.config import default_config
from quant_system.data.loader import load_price_data
from quant_system.data.universe import universe, FACTOR_ETFS, PAIRS_CANDIDATES
from quant_system.backtest.walk_forward import walk_forward
from quant_system.regime.detector import detect_regime, garch_regime
from quant_system.regime.switcher import apply_regime_sizing
from quant_system.signals.momentum import cross_sectional_momentum
from quant_system.performance.analytics import compute_metrics

OUT_DIR = "docs/results"


def main() -> int:
    cfg = default_config()
    tickers = sorted(set(
        universe("all")
        + list(dict.fromkeys(FACTOR_ETFS.values()))
        + [t for pair in PAIRS_CANDIDATES for t in pair]
    ))
    print(f"[regime] loading {len(tickers)} tickers {cfg.start}..{cfg.end} (real data)")
    pdat = load_price_data(tickers, cfg.start, cfg.end, cache_dir=cfg.cache_dir,
                           allow_synthetic_fallback=False)
    if pdat.synthetic:
        print("[regime] refusing to build results from synthetic data")
        return 1

    market = FACTOR_ETFS["market"]
    vol_ratio = detect_regime(pdat, cfg.regime, seed=cfg.random_seed,
                              benchmark=market).causal_labels
    print("[regime] fitting GARCH regime on the market proxy...")
    garch = garch_regime(pdat, cfg.regime, benchmark=market)

    panel = pdat.subset(universe("sectors"))
    scale = cfg.regime.defensive_scale

    def raw(pdata, fit_end):
        return cross_sectional_momentum(pdata, cfg.momentum)

    def vol_ratio_sized(pdata, fit_end):
        return apply_regime_sizing(cross_sectional_momentum(pdata, cfg.momentum), vol_ratio, scale)

    def garch_sized(pdata, fit_end):
        return apply_regime_sizing(cross_sectional_momentum(pdata, cfg.momentum), garch, scale)

    variants = [
        ("No regime", raw),
        ("Vol-ratio regime", vol_ratio_sized),
        ("GARCH regime", garch_sized),
    ]
    results = {}
    for label, make in variants:
        print(f"[regime] walk-forward: {label}")
        wf = walk_forward(panel, make, cfg.walk_forward, cfg.cost, verbose=False)
        m = compute_metrics(wf.returns, turnover=wf.turnover)
        results[label] = {"wf": wf, "metrics": m}
        print(f"         Sharpe {m['sharpe']:+.2f}  ann ret {m['ann_return']:+.1%}  "
              f"maxDD {m['max_drawdown']:+.1%}  vol {m['ann_vol']:.1%}")

    span = results["No regime"]["wf"].oos_span
    print("\n--- paste into README ---")
    print(f"Sector-momentum sleeve, out-of-sample walk-forward, {span}, net of costs.\n")
    print("| Defensive sizing | OOS Sharpe | Ann. return | Ann. vol | Max drawdown |")
    print("|---|---|---|---|---|")
    for label in ("No regime", "Vol-ratio regime", "GARCH regime"):
        m = results[label]["metrics"]
        print(f"| {label} | {m['sharpe']:.2f} | {m['ann_return']:+.1%} "
              f"| {m['ann_vol']:.1%} | {m['max_drawdown']:.1%} |")

    os.makedirs(OUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for label, color in (("No regime", "#7f7f7f"), ("Vol-ratio regime", "#1f77b4"),
                         ("GARCH regime", "#d62728")):
        eq = results[label]["wf"].equity
        ax.plot(eq.index, eq.values, lw=1.3, label=label, color=color)
    ax.axhline(1.0, color="grey", lw=0.7)
    ax.set_title(f"Defensive sizing: realized-vol vs GARCH regime ({span})")
    ax.set_ylabel("Growth of $1")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = f"{OUT_DIR}/regime_sizing.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"\n[regime] wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
