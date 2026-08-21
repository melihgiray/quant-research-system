"""Compare Hierarchical Risk Parity against inverse-variance allocation.

Rolling, causal, real-data allocation contest on the full ETF-and-large-cap
universe. Every month, estimate the covariance and correlation from the trailing
year (data through the prior day only), build long-only weights three ways, and
hold them for the next month:

  - HRP: cluster, quasi-diagonalise, split risk down the tree.
  - IVP: inverse-variance (naive risk parity), the baseline HRP is meant to beat.
  - equal weight: 1/N, for context.

This is a gross allocation comparison, no trading-cost model, because the
question here is allocation quality, not execution: does HRP earn similar return
at lower realised risk and with less concentration. Concentration is the
Herfindahl index of the weights and its effective-N reading (1 / Herfindahl).

Writes docs/results/hrp_vs_ivp.png and prints a markdown table. Aborts on
synthetic data rather than publishing made-up numbers.

Usage:  python scripts/build_hrp_results.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quant_system.config import default_config
from quant_system.data.loader import load_price_data
from quant_system.data.universe import universe, FACTOR_ETFS, PAIRS_CANDIDATES
from quant_system.research import research_universe
from quant_system.performance.analytics import compute_metrics
from quant_system.portfolio import hrp_weights, inverse_variance_weights

OUT_DIR = "docs/results"
LOOKBACK = 252          # trailing year for the covariance estimate
STEP = 21               # rebalance monthly


def _rolling_allocation(rets: pd.DataFrame, weigh):
    """Roll a monthly-rebalanced long-only book; return (daily returns, mean Herfindahl)."""
    dates = rets.index
    held = None
    daily, herfindahls = [], []
    for i in range(LOOKBACK, len(dates)):
        if (i - LOOKBACK) % STEP == 0:
            window = rets.iloc[i - LOOKBACK:i].dropna(axis=1, how="any")   # only fully-observed names
            if window.shape[1] >= 2:
                w = weigh(window)
                held = w.reindex(rets.columns).fillna(0.0)
                herfindahls.append(float((w.to_numpy() ** 2).sum()))
        if held is None:
            continue
        today = rets.iloc[i].fillna(0.0)
        daily.append((held * today).sum())
    series = pd.Series(daily, index=dates[LOOKBACK:LOOKBACK + len(daily)])
    return series, float(np.mean(herfindahls))


def main() -> int:
    cfg = default_config()
    tickers = research_universe()
    print(f"[hrp] loading {len(tickers)} tickers {cfg.start}..{cfg.end} (real data)")
    pdat = load_price_data(tickers, cfg.start, cfg.end, cache_dir=cfg.cache_dir,
                           allow_synthetic_fallback=False)
    if pdat.synthetic:
        print("[hrp] refusing to build results from synthetic data")
        return 1
    rets = pdat.close.pct_change()
    print(f"[hrp] panel: {rets.shape[0]} days x {rets.shape[1]} tickers")

    books = {
        "HRP": lambda w: hrp_weights(w),
        "Inverse-variance": lambda w: inverse_variance_weights(w.cov()),
        "Equal weight": lambda w: pd.Series(1.0 / w.shape[1], index=w.columns),
    }

    results = {}
    for name, weigh in books.items():
        series, herf = _rolling_allocation(rets, weigh)
        m = compute_metrics(series)
        results[name] = {"series": series, "metrics": m, "herfindahl": herf}
        print(f"[hrp] {name:17s} Sharpe {m['sharpe']:+.2f}  vol {m['ann_vol']:.1%}  "
              f"eff-N {1.0 / herf:.1f}")

    span = f"{results['HRP']['series'].index[0].date()}..{results['HRP']['series'].index[-1].date()}"
    print("\n--- paste into README ---")
    print(f"Monthly-rebalanced long-only allocation, trailing-year covariance, "
          f"{span}, gross of trading costs.\n")
    print("| Allocator | Sharpe | Ann. return | Ann. vol | Max drawdown | Effective N |")
    print("|---|---|---|---|---|---|")
    for name in ("HRP", "Inverse-variance", "Equal weight"):
        m = results[name]["metrics"]
        eff = 1.0 / results[name]["herfindahl"]
        print(f"| {name} | {m['sharpe']:.2f} | {m['ann_return']:+.1%} | {m['ann_vol']:.1%} "
              f"| {m['max_drawdown']:.1%} | {eff:.1f} |")

    # ---- equity chart ----
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name, color in (("HRP", "#1f77b4"), ("Inverse-variance", "#ff7f0e"),
                        ("Equal weight", "#7f7f7f")):
        eq = (1 + results[name]["series"]).cumprod()
        ax.plot(eq.index, eq.values, lw=1.4, label=name, color=color)
    ax.axhline(1.0, color="grey", lw=0.7)
    ax.set_title(f"HRP vs inverse-variance, monthly rebalance, gross ({span})")
    ax.set_ylabel("Growth of $1")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = f"{OUT_DIR}/hrp_vs_ivp.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"\n[hrp] wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
