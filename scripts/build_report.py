"""Write the HTML tearsheet for the blended book.

Runs the same real-data walk-forward as build_blend_results.py for the three
sleeves, combines them by inverse-vol allocation and a volatility target, then
renders a one-file HTML tearsheet with the market ETF as the beta benchmark.
Aborts on synthetic data rather than publishing made-up numbers.

Writes docs/results/report.html.

Usage:  python scripts/build_report.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
from quant_system.performance.html_report import build_html_report
from quant_system.portfolio import blend_returns, inverse_vol_allocations, volatility_target

OUT_PATH = "docs/results/report.html"
TARGET_VOL = 0.10


def main() -> int:
    cfg = default_config()
    tickers = research_universe()
    print(f"[report] loading {len(tickers)} tickers {cfg.start}..{cfg.end} (real data)")
    pdat = load_price_data(tickers, cfg.start, cfg.end, cache_dir=cfg.cache_dir,
                           allow_synthetic_fallback=False)
    if pdat.synthetic:
        print("[report] refusing to build a report from synthetic data")
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
        print(f"[report] walk-forward: {name}")
        sleeve_returns[name] = walk_forward(panel, make, cfg.walk_forward, cfg.cost,
                                            verbose=False).returns

    alloc = inverse_vol_allocations(sleeve_returns, lookback=cfg.walk_forward.out_sample)
    blended = blend_returns(sleeve_returns, allocations=alloc)
    targeted = volatility_target(blended, target_vol=TARGET_VOL,
                                 lookback=252, max_leverage=3.0).dropna()

    market = pdat.close[FACTOR_ETFS["market"]].pct_change()
    html = build_html_report(targeted, benchmark=market,
                             title="Blended book (inverse-vol, vol-targeted)")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as fh:
        fh.write(html)
    print(f"[report] wrote {OUT_PATH} ({len(html) // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
