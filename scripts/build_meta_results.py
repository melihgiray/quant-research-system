"""Compare the ML sleeve with and without meta-labeling.

Runs the directional ML sleeve two ways through the same real-data walk-forward:
the primary model alone (its side, sized by prediction strength) and the
meta-labeled version (primary sets the side, a secondary model sets the size and
vetoes low-conviction bets). Both get the same regime sizing, so the only
difference is the meta layer. Aborts on synthetic data rather than publishing
made-up numbers.

Prints a markdown row for the README.

Usage:  python scripts/build_meta_results.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_system.config import default_config
from quant_system.data.loader import load_price_data
from quant_system.data.universe import universe, FACTOR_ETFS
from quant_system.research import research_universe
from quant_system.backtest.walk_forward import walk_forward
from quant_system.regime.detector import detect_regime
from quant_system.regime.switcher import apply_regime_sizing
from quant_system.signals.ml_signal import train_predict
from quant_system.signals.meta_labeling import meta_train_predict
from quant_system.performance.analytics import compute_metrics


def main() -> int:
    cfg = default_config()
    tickers = research_universe()
    print(f"[meta] loading {len(tickers)} tickers {cfg.start}..{cfg.end} (real data)")
    pdat = load_price_data(tickers, cfg.start, cfg.end, cache_dir=cfg.cache_dir,
                           allow_synthetic_fallback=False)
    if pdat.synthetic:
        print("[meta] refusing to build results from synthetic data")
        return 1

    regime = detect_regime(pdat, cfg.regime, seed=cfg.random_seed,
                           benchmark=FACTOR_ETFS["market"])
    panel = pdat.subset(universe("largecaps"))

    def primary(pdata, fit_end):
        w = train_predict(pdata, fit_end, cfg.ml, max_weight=cfg.risk.max_weight)
        return apply_regime_sizing(w, regime.causal_labels, cfg.regime.defensive_scale)

    def meta(pdata, fit_end):
        w = meta_train_predict(pdata, fit_end, cfg.ml, max_weight=cfg.risk.max_weight)
        return apply_regime_sizing(w, regime.causal_labels, cfg.regime.defensive_scale)

    results = {}
    for label, make in (("ML primary", primary), ("ML meta-labeled", meta)):
        print(f"[meta] walk-forward: {label}")
        wf = walk_forward(panel, make, cfg.walk_forward, cfg.cost, verbose=False)
        m = compute_metrics(wf.returns, turnover=wf.turnover)
        results[label] = m
        print(f"       Sharpe {m['sharpe']:+.2f}  ann ret {m['ann_return']:+.1%}  "
              f"maxDD {m['max_drawdown']:+.1%}  turnover {m['ann_turnover']:.1f}x")

    print("\n--- paste into README ---")
    print("| ML sleeve | OOS Sharpe | Ann. return | Max drawdown | Turnover |")
    print("|---|---|---|---|---|")
    for label in ("ML primary", "ML meta-labeled"):
        m = results[label]
        print(f"| {label} | {m['sharpe']:.2f} | {m['ann_return']:+.1%} "
              f"| {m['max_drawdown']:.1%} | {m['ann_turnover']:.1f}x |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
