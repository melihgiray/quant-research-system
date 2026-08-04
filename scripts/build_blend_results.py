"""Build the blended-book results: three sleeves combined into one book.

Runs the same real-data walk-forward as build_results.py for the three sleeves
(cross-sectional momentum, per-fold pairs, ML directional), then combines their
out-of-sample return streams into one book by inverse-volatility allocation and
targets the book's volatility. Everything is causal: the allocation for a day
uses trailing sleeve vol through the prior day, and the vol-target scaler is
lagged the same way.

Each sleeve's returns are already net of that sleeve's trading costs, so the
blend is a fund-of-strategies over net streams, not a re-costed super-book. That
is stated in the README row rather than hidden.

Writes docs/results/equity_blend.png and prints a markdown row to paste into the
README. Aborts on synthetic data rather than publishing made-up numbers.

Usage:  python scripts/build_blend_results.py
"""

import os
import sys

# Make the script runnable as `python scripts/build_blend_results.py` from root.
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
from quant_system.signals.mean_reversion import causal_pairs_weights
from quant_system.signals.ml_signal import train_predict
from quant_system.performance.analytics import compute_metrics
from quant_system.portfolio import (
    blend_returns, inverse_vol_allocations, volatility_target,
)

OUT_DIR = "docs/results"
TARGET_VOL = 0.10


def main() -> int:
    cfg = default_config()
    tickers = sorted(set(
        universe("all")
        + list(dict.fromkeys(FACTOR_ETFS.values()))
        + [t for pair in PAIRS_CANDIDATES for t in pair]
    ))
    print(f"[blend] loading {len(tickers)} tickers {cfg.start}..{cfg.end} (real data)")
    pdat = load_price_data(tickers, cfg.start, cfg.end, cache_dir=cfg.cache_dir,
                           allow_synthetic_fallback=False)
    if pdat.synthetic:
        print("[blend] refusing to build results from synthetic data")
        return 1
    print(f"[blend] panel: {pdat.close.shape[0]} days x {pdat.close.shape[1]} tickers")

    regime = detect_regime(pdat, cfg.regime, seed=cfg.random_seed,
                           benchmark=FACTOR_ETFS["market"])

    def momentum_weights(pdata, fit_end):
        w = cross_sectional_momentum(pdata, cfg.momentum)
        return apply_regime_sizing(w, regime.causal_labels, cfg.regime.defensive_scale)

    def pairs_weights(pdata, fit_end):
        return causal_pairs_weights(pdata, fit_end, PAIRS_CANDIDATES, cfg.pairs)

    def ml_weights(pdata, fit_end):
        w = train_predict(pdata, fit_end, cfg.ml, max_weight=cfg.risk.max_weight)
        return apply_regime_sizing(w, regime.causal_labels, cfg.regime.defensive_scale)

    sleeves = [
        ("momentum", pdat.subset(universe("sectors")), momentum_weights),
        ("pairs", pdat, pairs_weights),
        ("ml", pdat.subset(universe("largecaps")), ml_weights),
    ]

    sleeve_returns = {}
    for name, panel, make in sleeves:
        print(f"[blend] walk-forward: {name}")
        wf = walk_forward(panel, make, cfg.walk_forward, cfg.cost, verbose=False)
        sleeve_returns[name] = wf.returns
        m = compute_metrics(wf.returns)
        print(f"        Sharpe {m['sharpe']:+.2f}  ann vol {m['ann_vol']:.1%}")

    # ---- combine: inverse-vol allocation, then vol-target the blend ----
    alloc = inverse_vol_allocations(sleeve_returns, lookback=cfg.walk_forward.out_sample)
    blended = blend_returns(sleeve_returns, allocations=alloc)
    targeted = volatility_target(blended, target_vol=TARGET_VOL,
                                 lookback=252, max_leverage=3.0).dropna()

    # Naive equal-weight blend, as an honest baseline for the inverse-vol choice.
    frame = pd.DataFrame(sleeve_returns).sort_index()
    equal = frame.mean(axis=1)
    equal_targeted = volatility_target(equal, target_vol=TARGET_VOL,
                                       lookback=252, max_leverage=3.0).dropna()

    mb = compute_metrics(targeted)
    me = compute_metrics(equal_targeted)
    span = f"{targeted.index[0].date()}..{targeted.index[-1].date()}"

    print("\n--- paste into README ---")
    print(f"Blended book, inverse-vol across sleeves, targeted to {TARGET_VOL:.0%} "
          f"annual vol, {span}.\n")
    print("| Book | OOS Sharpe | Ann. return | Ann. vol | Max drawdown |")
    print("|---|---|---|---|---|")
    print(f"| Blended (inverse-vol, vol-targeted) | {mb['sharpe']:.2f} "
          f"| {mb['ann_return']:+.1%} | {mb['ann_vol']:.1%} | {mb['max_drawdown']:.1%} |")
    print(f"\nbaseline check: equal-weight blend Sharpe {me['sharpe']:+.2f} "
          f"vs inverse-vol {mb['sharpe']:+.2f}; realised vol lands at "
          f"{mb['ann_vol']:.1%} against a {TARGET_VOL:.0%} target.")

    # ---- equity chart: blended book vs its sleeves ----
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name, color in (("momentum", "#1f77b4"), ("pairs", "#2ca02c"),
                        ("ml", "#d62728")):
        r = sleeve_returns[name].reindex(targeted.index).fillna(0.0)
        ax.plot(r.index, (1 + r).cumprod().values, lw=1.0, alpha=0.6,
                label=f"{name} sleeve", color=color)
    ax.plot(targeted.index, (1 + targeted).cumprod().values, lw=2.0,
            label="blended book (vol-targeted)", color="black")
    ax.axhline(1.0, color="grey", lw=0.7)
    ax.set_title(f"Blended book vs sleeves, out-of-sample, net of costs ({span})")
    ax.set_ylabel("Growth of $1")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = f"{OUT_DIR}/equity_blend.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"\n[blend] wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
