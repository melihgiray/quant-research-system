"""Command-line entry point: ``python -m quant_system.cli``.

End-to-end run:
  1. Load (and cache) data for the configured universe + factor ETFs.
  2. Detect the market regime (for sizing + an inspectable overlay).
  3. For each selected strategy, run WALK-FORWARD validation and keep only the
     concatenated out-of-sample return stream.
  4. Decompose the OOS returns against the Fama-French 3 factors.
  5. Print a full tearsheet and save plots to reports/.
  6. Return exit code 0 on success.

Everything is parameterised via flags; there are no hard-coded dates in the logic
(defaults live in config.py and are overridable).
"""

from __future__ import annotations

import argparse
import sys
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import Config, default_config
from .data.loader import load_price_data
from .data.universe import universe, FACTOR_ETFS, PAIRS_CANDIDATES
from .backtest.walk_forward import walk_forward
from .backtest.engine import run_backtest
from .backtest.capacity import sweep_capital, estimate_capacity, plot_capacity, capacity_summary
from .regime.detector import detect_regime, market_proxy_returns
from .regime.switcher import apply_regime_sizing
from .signals.momentum import cross_sectional_momentum
from .signals.mean_reversion import (
    causal_pairs_weights, find_cointegrated_pair, pairs_signal, scan_candidate_pairs,
)
from .performance.multiple_testing import fdr_report, fdr_summary_lines
from .signals.ml_signal import (
    train_predict, shap_feature_importance, ml_feature_significance,
)
from .signals.feature_selection import feature_fdr_summary
from .signals.cv import purged_cv_scores
from .performance.tearsheet import format_tearsheet, save_report_plots
from .performance.factor_decomp import factor_decomposition
from .performance.bootstrap import bootstrap_summary


def build_config(args) -> Config:
    """Construct the immutable Config from CLI args (overriding defaults)."""
    from .config import ExecutionConfig

    cfg = default_config().with_(
        start=args.start,
        end=args.end,
        use_synthetic=args.synthetic,
        reports_dir=args.reports_dir,
        risk_free_rate=args.rf,
        execution=ExecutionConfig(max_participation=args.max_participation),
    )
    return cfg


def get_risk_free_rate(args) -> float:
    """Resolve the annual risk-free rate: explicit flag, optional FRED, else 0."""
    if args.rf_from_fred:
        try:
            from pandas_datareader import data as pdr
            s = pdr.DataReader("DGS3MO", "fred", args.start, args.end)["DGS3MO"]
            rf = float(s.dropna().mean()) / 100.0
            print(f"[cli] risk-free (FRED DGS3MO avg): {rf:.2%}")
            return rf
        except Exception as exc:
            print(f"[cli] FRED rf fetch failed ({exc}); using rf={args.rf:.2%}")
    return args.rf


def _required_tickers(strategy: str) -> List[str]:
    """Universe + factor ETFs (+ pair candidates when pairs are in play)."""
    base = list(dict.fromkeys(FACTOR_ETFS.values()))
    if strategy in ("pairs", "all"):
        for a, b in PAIRS_CANDIDATES:
            base += [a, b]
    return list(dict.fromkeys(base))


def _momentum_weights_fn(cfg, regime, use_regime):
    def make(pdata, fit_end):
        w = cross_sectional_momentum(pdata, cfg.momentum)
        if use_regime and regime is not None:
            w = apply_regime_sizing(w, regime.causal_labels, cfg.regime.defensive_scale)
        return w
    return make


def _pairs_weights_fn(cfg):
    def make(pdata, fit_end):
        return causal_pairs_weights(pdata, fit_end, PAIRS_CANDIDATES, cfg.pairs,
                                    verbose=True)
    return make


def _ml_weights_fn(cfg, regime, use_regime):
    def make(pdata, fit_end):
        w = train_predict(pdata, fit_end, cfg.ml, max_weight=cfg.risk.max_weight)
        if use_regime and regime is not None:
            w = apply_regime_sizing(w, regime.causal_labels, cfg.regime.defensive_scale)
        return w
    return make


def run_one_strategy(name: str, pdat, cfg: Config, regime, rf: float,
                     use_regime: bool, walk: bool, n_trials: int = 1,
                     bootstrap: bool = False) -> Optional[dict]:
    """Run a single strategy end-to-end; return a dict of artefacts or None."""
    universe_data = pdat  # signals operate on the loaded panel (factor ETFs ignored by ranking)

    if name == "momentum":
        sub = pdat.subset(universe("sectors"))
        make = _momentum_weights_fn(cfg, regime, use_regime)
        signal_label = "Cross-sectional momentum (12-1, monthly)"
        sig_universe = sub
    elif name == "pairs":
        scanned = scan_candidate_pairs(pdat.close, PAIRS_CANDIDATES)
        if not scanned:
            print("[cli] pairs: no candidate pair has enough data - skipping")
            return None
        # Diagnostic only: the full-sample scan with the FDR correction. The
        # selection that actually trades happens per walk-forward fold, on data
        # available at each fold's boundary.
        report = fdr_report([f"{a}/{b}" for a, b, _ in scanned],
                            [p for _, _, p in scanned],
                            alpha=cfg.pairs.coint_pvalue_max)
        print("\n".join(fdr_summary_lines(report, cfg.pairs.coint_pvalue_max)))
        print("[cli] pairs: selection is per fold (see [pairs] lines below)")
        make = _pairs_weights_fn(cfg)
        signal_label = f"Pairs trade, per-fold selection (z>{cfg.pairs.entry_z})"
        sig_universe = pdat
    elif name == "ml":
        sub = pdat.subset(universe("largecaps")) if any(
            t in pdat.close.columns for t in universe("largecaps")) else pdat
        make = _ml_weights_fn(cfg, regime, use_regime)
        signal_label = "ML gradient-boosting directional signal"
        sig_universe = sub
    else:
        raise ValueError(f"unknown strategy {name!r}")

    # Walk-forward (preferred) or single full-sample backtest.
    if walk:
        wf = walk_forward(sig_universe, make, cfg.walk_forward, cfg.cost, verbose=True,
                          execution=cfg.execution)
        returns, turnover = wf.returns, wf.turnover
        span_note = f"walk-forward OOS, {wf.n_folds} folds, {wf.oos_span}"
    else:
        w = make(sig_universe, sig_universe.close.index[-1])
        res = run_backtest(w, sig_universe, cost=cfg.cost, execution=cfg.execution)
        returns, turnover = res.returns, res.turnover
        span_note = "full-sample (no walk-forward)"

    # Factor decomposition on the OOS stream.
    fd = factor_decomposition(returns, pdat.close, rf_annual=rf)
    extra = [span_note]
    if bootstrap:
        extra += [""] + bootstrap_summary(returns)
    if fd is not None:
        extra += [""] + fd.summary()
    else:
        extra += ["(factor ETFs unavailable - decomposition skipped)"]

    print("\n" + format_tearsheet(returns, turnover, rf_annual=rf,
                                  title=signal_label, extra_lines=extra,
                                  n_trials=n_trials))

    return {
        "name": name,
        "label": signal_label,
        "returns": returns,
        "turnover": turnover,
        "factor": fd,
        "sig_universe": sig_universe,
    }


def main(argv: Optional[List[str]] = None) -> int:
    """Parse args, run the selected strategies, save reports. Returns exit code."""
    warnings.filterwarnings("ignore")
    p = argparse.ArgumentParser(
        prog="quant_system",
        description="Quantitative research system: signals + walk-forward backtest + tearsheet",
    )
    p.add_argument("--strategy", choices=["momentum", "pairs", "ml", "all"], default="all")
    p.add_argument("--universe", choices=["sectors", "largecaps", "all"], default="all")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--synthetic", action="store_true",
                   help="use the offline deterministic data generator (no network)")
    p.add_argument("--no-walk-forward", action="store_true",
                   help="single full-sample backtest instead of walk-forward")
    p.add_argument("--no-regime", action="store_true", help="disable regime-based sizing")
    p.add_argument("--rf", type=float, default=0.0, help="annual risk-free rate")
    p.add_argument("--rf-from-fred", action="store_true",
                   help="fetch 3M T-bill rate from FRED (needs pandas-datareader + network)")
    p.add_argument("--reports-dir", default="reports")
    p.add_argument("--shap", action="store_true",
                   help="compute SHAP feature importance for the ML signal")
    p.add_argument("--n-trials", type=int, default=10,
                   help="effective number of strategy/parameter configs searched "
                        "(used to deflate the Sharpe; higher = more conservative)")
    p.add_argument("--bootstrap", action="store_true",
                   help="add stationary-bootstrap confidence intervals to each tearsheet")
    p.add_argument("--capacity", action="store_true",
                   help="run a capital/cost-sensitivity sweep on the momentum book")
    p.add_argument("--cv", action="store_true",
                   help="score the ML classifier with purged, embargoed K-fold CV")
    p.add_argument("--feature-fdr", action="store_true",
                   help="permutation-null p-value per ML feature, FDR-corrected")
    p.add_argument("--max-participation", type=float, default=None, metavar="FRAC",
                   help="cap daily trading in a name at this fraction of its ADV "
                        "(e.g. 0.05); unfilled amounts carry to later days")
    args = p.parse_args(argv)

    cfg = build_config(args)
    rf = get_risk_free_rate(args)
    use_regime = not args.no_regime
    walk = not args.no_walk_forward

    # ---- data ----
    uni = universe("all") if args.universe == "all" else universe(args.universe)
    tickers = list(dict.fromkeys(uni + _required_tickers(args.strategy)))
    print(f"[cli] loading {len(tickers)} tickers {args.start}..{args.end} "
          f"({'synthetic' if args.synthetic else 'yfinance+cache'})")
    pdat = load_price_data(tickers, args.start, args.end, cache_dir=cfg.cache_dir,
                           use_synthetic=args.synthetic, seed=cfg.random_seed)
    if pdat.synthetic and not args.synthetic:
        print("[cli] NOTE: live data unavailable - using SYNTHETIC fallback "
              "(numbers are illustrative, not live).")

    # ---- regime ----
    regime = detect_regime(pdat, cfg.regime, seed=cfg.random_seed,
                           benchmark=FACTOR_ETFS["market"])
    print(f"[cli] regime detector: {regime.method}; "
          f"defensive {regime.defensive_fraction:.0%} of days")

    # ---- strategies ----
    strategies = ["momentum", "pairs", "ml"] if args.strategy == "all" else [args.strategy]
    artefacts: Dict[str, dict] = {}
    for name in strategies:
        try:
            art = run_one_strategy(name, pdat, cfg, regime, rf, use_regime, walk,
                                   n_trials=args.n_trials, bootstrap=args.bootstrap)
            if art:
                artefacts[name] = art
        except Exception as exc:
            print(f"[cli] strategy '{name}' failed: {exc}")

    if not artefacts:
        print("[cli] no strategy produced results")
        return 1

    # ---- purged cross-validation (optional, ML only) ----
    if args.cv:
        print("\n[cli] purged cross-validation of the ML classifier...")
        cv_universe = pdat.subset(universe("largecaps")) if any(
            t in pdat.close.columns for t in universe("largecaps")) else pdat
        cv_res = purged_cv_scores(cv_universe, cfg.ml, seed=cfg.random_seed)
        if cv_res is not None:
            print("\n".join(cv_res.summary()))
        else:
            print("[cli] not enough data for the CV folds")

    # ---- SHAP (optional, ML only) ----
    if args.shap and "ml" in artefacts:
        print("\n[cli] computing SHAP feature importance (ML)...")
        sh = shap_feature_importance(artefacts["ml"]["sig_universe"], cfg.ml,
                                     seed=cfg.random_seed)
        if sh is not None:
            print("\n".join(sh.summary()))

    # ---- feature-significance FDR (optional, ML only) ----
    if args.feature_fdr:
        print("\n[cli] ML feature significance (permutation null, FDR)...")
        fdr_universe = pdat.subset(universe("largecaps")) if any(
            t in pdat.close.columns for t in universe("largecaps")) else pdat
        report = ml_feature_significance(fdr_universe, cfg.ml, seed=cfg.random_seed)
        if report is not None:
            print("\n".join(feature_fdr_summary(report, cfg.pairs.coint_pvalue_max)))
        else:
            print("[cli] not enough data for the feature scan")

    # ---- plots ----
    market_level = (1.0 + market_proxy_returns(pdat, FACTOR_ETFS["market"]).fillna(0.0)).cumprod()
    saved: List[str] = []
    for name, art in artefacts.items():
        fd = art["factor"]
        saved += save_report_plots(
            cfg.reports_dir,
            returns=art["returns"],
            price=market_level,
            regimes=regime.labels,
            strat_excess=(art["returns"] - rf / 252) if fd is not None else None,
            market_excess=fd.market_excess if fd is not None else None,
            alpha_daily=fd.alpha_daily if fd is not None else 0.0,
            beta=fd.betas.get("MKT", 0.0) if fd is not None else 0.0,
            signal_values=art["turnover"],
            prefix=name,
        )
    print(f"\n[cli] saved {len(saved)} plot(s) to {cfg.reports_dir}/")
    for s in saved:
        print(f"        {s}")

    # ---- capacity / cost sensitivity (optional) ----
    if args.capacity:
        print("\n[cli] capacity / cost-sensitivity sweep (momentum book)...")
        sub = pdat.subset(universe("sectors"))
        w_cap = cross_sectional_momentum(sub, cfg.momentum)
        capitals = [1e5, 1e6, 5e6, 1e7, 5e7, 1e8, 5e8, 1e9, 5e9]
        sweep = sweep_capital(w_cap, sub, cfg.cost, capitals)
        cap = estimate_capacity(sweep)
        print("\n".join(capacity_summary(sweep, cap)))
        path = plot_capacity(sweep, f"{cfg.reports_dir}/capacity.png", cap.get("capacity"))
        print(f"[cli] saved {path}")

    print("\n[cli] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
