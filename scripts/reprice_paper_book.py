"""Reprice a small paper option book from a live chain and check its health.

Pulls the current chain, builds the vol surface, fits an arbitrage-free SSVI
surface, runs the fail-loud health check, and writes a JSON run log. Exits
non-zero when the data is unhealthy (empty chain, too many implied-vol failures,
or arbitrage beyond the bid-ask spread), so a scheduled job fails loudly instead
of trusting stale data.

This does NOT trade and does NOT commit anything: it reprices and reports. The
companion workflow uploads the log as a build artifact.

Usage:
    python scripts/reprice_paper_book.py --ticker SPY --out paper_book_log.json
    python scripts/reprice_paper_book.py --synthetic          # offline smoke test
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_system.config import default_config
from quant_system.options import build_surface, load_option_chain, synthetic_option_chain
from quant_system.options.monitor import repricing_health
from quant_system.options.ssvi import (
    fit_ssvi_surface, ssvi_surface_arbitrage_free, ssvi_surface_calendar_free,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--rate", type=float, default=0.04)
    parser.add_argument("--dividend-yield", type=float, default=0.012)
    parser.add_argument("--expiries", type=int, default=7)
    parser.add_argument("--out", default="paper_book_log.json")
    parser.add_argument("--synthetic", action="store_true",
                        help="use the generated chain (offline smoke test)")
    args = parser.parse_args()

    cfg = default_config()
    if args.synthetic:
        chain = synthetic_option_chain(ticker=args.ticker, rate=args.rate,
                                       dividend_yield=args.dividend_yield, cfg=cfg.options)
    else:
        try:
            chain = load_option_chain(args.ticker, cfg=cfg.options, max_expiries=args.expiries)
        except RuntimeError as exc:
            print(f"[reprice] live chain unavailable for {args.ticker}: {exc}")
            return 2                                       # infrastructure failure, not a data verdict

    surface = build_surface(chain, rate=args.rate,
                            dividend_yield=args.dividend_yield, cfg=cfg.options)
    health = repricing_health(surface)
    print("\n".join(health.summary()))

    log = {
        "run_time": dt.datetime.now(dt.timezone.utc).isoformat(),
        "ticker": args.ticker,
        "synthetic": bool(surface.synthetic),
        "healthy": health.ok,
        "n_points": health.n_points,
        "iv_failure_rate": round(health.iv_failure_rate, 4),
        "tradeable_arb": health.tradeable_arb,
        "issues": health.issues,
    }

    # Fit the SSVI surface only when there is enough clean data to bother.
    if health.n_points > 0:
        try:
            fit, diag = fit_ssvi_surface(surface.points)
            butterfly, _ = ssvi_surface_arbitrage_free(fit)
            log["ssvi"] = {
                "rho": round(fit.rho, 4), "eta": round(fit.eta, 4), "gamma": round(fit.gamma, 4),
                "butterfly_free": bool(butterfly),
                "calendar_free": bool(ssvi_surface_calendar_free(fit)),
                "term_structure": [
                    {"time_to_expiry": round(float(r.time_to_expiry), 4),
                     "theta": round(float(r.theta), 6), "rmse": round(float(r.rmse), 6)}
                    for r in diag.itertuples()],
            }
        except ValueError as exc:
            log["ssvi_error"] = str(exc)

    with open(args.out, "w") as fh:
        json.dump(log, fh, indent=2)
    print(f"\n[reprice] wrote {args.out}")

    return 0 if health.ok else 1                            # fail loud on unhealthy data


if __name__ == "__main__":
    sys.exit(main())
