"""Build a vol surface from a live option chain and write the plot + a report.

    python scripts/build_vol_surface.py            # SPY, live chain
    python scripts/build_vol_surface.py --ticker AAPL
    python scripts/build_vol_surface.py --synthetic

Writes docs/results/vol_surface_<ticker>.png and prints the chain summary,
the ATM term structure and the no-arbitrage findings.

Live mode refuses to fall back to synthetic data: if the chain cannot be
fetched it exits non-zero. Publishing a "surface" built from generated prices
as though it came from the market would be the one genuinely dishonest thing
this repo could do.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_system.config import default_config
from quant_system.options import (
    build_surface, load_option_chain, plot_surface, synthetic_option_chain,
)

OUT_DIR = "docs/results"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--expiries", type=int, default=7,
                        help="how many expiries to sample across the term structure")
    parser.add_argument("--rate", type=float, default=0.04)
    parser.add_argument("--dividend-yield", type=float, default=0.012)
    parser.add_argument("--synthetic", action="store_true",
                        help="use the generated chain instead of live quotes")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s")

    cfg = default_config()

    if args.synthetic:
        chain = synthetic_option_chain(ticker=args.ticker, rate=args.rate,
                                       dividend_yield=args.dividend_yield,
                                       cfg=cfg.options)
    else:
        try:
            chain = load_option_chain(args.ticker, cfg=cfg.options,
                                      max_expiries=args.expiries)
        except RuntimeError as exc:
            print(f"[surface] live chain unavailable for {args.ticker}: {exc}")
            print("[surface] refusing to substitute synthetic data in live mode; "
                  "pass --synthetic if that is what you want")
            return 1

    print("\n".join(chain.summary()))
    print()

    surface = build_surface(chain, rate=args.rate,
                            dividend_yield=args.dividend_yield, cfg=cfg.options)
    print("\n".join(surface.summary()))

    os.makedirs(OUT_DIR, exist_ok=True)
    suffix = "synthetic" if surface.synthetic else args.ticker.lower()
    path = plot_surface(surface, f"{OUT_DIR}/vol_surface_{suffix}.png")
    print(f"\n[surface] wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
