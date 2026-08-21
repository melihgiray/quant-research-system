"""Compare how the three sleeves are blended into one book: inverse-vol vs HRP.

Both take the same three out-of-sample sleeve return streams (momentum, pairs,
ML) and combine them into one vol-targeted book; they differ only in how capital
is split across the sleeves. Inverse-vol weights each sleeve by 1/vol and ignores
how they co-move. HRP clusters the sleeves on their correlation and splits the
risk budget down that tree. Equal weight is the naive baseline. Every allocation
is causal (trailing window, refit and held), and the combined stream is targeted
to 10% annual volatility.

With only three, largely uncorrelated sleeves, HRP and inverse-vol are expected
to land close; this measures whether the correlation-aware method earns anything
here. Aborts on synthetic data.

Usage:  python scripts/build_hrp_blend_results.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from quant_system.config import default_config
from quant_system.data.loader import load_price_data
from quant_system.data.universe import universe, FACTOR_ETFS, PAIRS_CANDIDATES
from quant_system.research import research_universe
from quant_system.backtest.walk_forward import walk_forward
from quant_system.regime.detector import detect_regime
from quant_system.regime.switcher import apply_regime_sizing
from quant_system.signals.momentum import cross_sectional_momentum
from quant_system.signals.mean_reversion import causal_pairs_weights
from quant_system.signals.ml_signal import train_predict
from quant_system.performance.analytics import compute_metrics
from quant_system.portfolio import (
    blend_returns, hrp_allocations, inverse_vol_allocations, volatility_target,
)

OUT_DIR = "docs/results"
TARGET_VOL = 0.10


def _targeted(sleeve_returns, allocations):
    blended = blend_returns(sleeve_returns, allocations=allocations)
    return volatility_target(blended, target_vol=TARGET_VOL,
                             lookback=252, max_leverage=3.0).dropna()


def main() -> int:
    cfg = default_config()
    tickers = research_universe()
    print(f"[hrp-blend] loading {len(tickers)} tickers {cfg.start}..{cfg.end} (real data)")
    pdat = load_price_data(tickers, cfg.start, cfg.end, cache_dir=cfg.cache_dir,
                           allow_synthetic_fallback=False)
    if pdat.synthetic:
        print("[hrp-blend] refusing to build results from synthetic data")
        return 1

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
        print(f"[hrp-blend] walk-forward: {name}")
        sleeve_returns[name] = walk_forward(panel, make, cfg.walk_forward, cfg.cost,
                                            verbose=False).returns

    frame = pd.DataFrame(sleeve_returns).sort_index()
    lookback = cfg.walk_forward.out_sample
    books = {
        "Inverse-vol blend": _targeted(sleeve_returns,
                                       inverse_vol_allocations(sleeve_returns, lookback=lookback)),
        "HRP blend": _targeted(sleeve_returns,
                               hrp_allocations(sleeve_returns, lookback=252, refit_every=21)),
        "Equal-weight blend": _targeted(
            sleeve_returns,
            pd.DataFrame(1.0 / frame.shape[1], index=frame.index, columns=frame.columns)),
    }

    # Score every book on the SAME window: HRP warms up longer than inverse-vol,
    # so without this the Sharpes would be measured over different date ranges.
    common = books["Inverse-vol blend"].index
    for stream in books.values():
        common = common.intersection(stream.index)
    books = {label: stream.reindex(common) for label, stream in books.items()}

    span = f"{common[0].date()}..{common[-1].date()}"
    print("\n--- paste into README ---")
    print(f"Three sleeves combined into one 10%-vol book, {span}, net of costs.\n")
    print("| Sleeve allocation | OOS Sharpe | Ann. return | Ann. vol | Max drawdown |")
    print("|---|---|---|---|---|")
    results = {}
    for label, stream in books.items():
        m = compute_metrics(stream)
        results[label] = (stream, m)
        print(f"| {label} | {m['sharpe']:.2f} | {m['ann_return']:+.1%} "
              f"| {m['ann_vol']:.1%} | {m['max_drawdown']:.1%} |")

    os.makedirs(OUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for label, color in (("Inverse-vol blend", "#1f77b4"), ("HRP blend", "#d62728"),
                         ("Equal-weight blend", "#7f7f7f")):
        stream = results[label][0]
        ax.plot(stream.index, (1 + stream).cumprod().values, lw=1.4, label=label, color=color)
    ax.axhline(1.0, color="grey", lw=0.7)
    ax.set_title(f"Sleeve allocation: inverse-vol vs HRP ({span})")
    ax.set_ylabel("Growth of $1")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = f"{OUT_DIR}/hrp_blend.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"\n[hrp-blend] wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
