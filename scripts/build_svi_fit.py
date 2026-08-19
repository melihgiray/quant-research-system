"""Fit SVI to a real vol surface and overlay the fits on the market smiles.

Builds a surface from an option chain (a live chain by default, or the generated
one with --synthetic), fits the raw-SVI smile to each expiry, and plots the
market implied vols against the fitted SVI curve for the best-populated expiries,
annotated with the RMS fit error and the butterfly no-arbitrage verdict. Also
prints the per-expiry SVI parameter table.

Live mode refuses to fall back to synthetic data, for the same reason as the
surface builder: a "market" figure built from generated prices would be the one
dishonest thing this repo could do.

Writes docs/results/svi_fit_<ticker>.png.

Usage:  python scripts/build_svi_fit.py [--ticker SPY] [--synthetic]
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from quant_system.config import default_config
from quant_system.options import (
    build_surface, load_option_chain, synthetic_option_chain,
)
from quant_system.options.svi import (
    SVIParams, fit_svi_points, is_butterfly_arbitrage_free, svi_implied_vol,
)

OUT_DIR = "docs/results"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--rate", type=float, default=0.04)
    parser.add_argument("--dividend-yield", type=float, default=0.012)
    parser.add_argument("--expiries", type=int, default=7)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    cfg = default_config()
    if args.synthetic:
        chain = synthetic_option_chain(ticker=args.ticker, rate=args.rate,
                                       dividend_yield=args.dividend_yield, cfg=cfg.options)
    else:
        try:
            chain = load_option_chain(args.ticker, cfg=cfg.options, max_expiries=args.expiries)
        except RuntimeError as exc:
            print(f"[svi] live chain unavailable for {args.ticker}: {exc}")
            print("[svi] refusing to substitute synthetic data in live mode; pass --synthetic")
            return 1

    surface = build_surface(chain, rate=args.rate,
                            dividend_yield=args.dividend_yield, cfg=cfg.options)
    fits = fit_svi_points(surface.points)
    if fits.empty:
        print("[svi] no expiry had enough points to fit; nothing to plot")
        return 1

    print("\nper-expiry SVI fit:")
    print(fits[["time_to_expiry", "a", "b", "rho", "m", "sigma",
                "rmse", "arb_free", "n_points"]].to_string(index=False))

    # Plot the best-populated expiries (up to four), market points vs SVI curve.
    top = fits.sort_values("n_points", ascending=False).head(4).sort_values("time_to_expiry")
    n = len(top)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.6), squeeze=False)
    for ax, (_, row) in zip(axes[0], top.iterrows()):
        t = row["time_to_expiry"]
        sel = surface.points[surface.points["time_to_expiry"] == t].sort_values("log_moneyness")
        k = sel["log_moneyness"].to_numpy(dtype=float)
        iv = sel["implied_vol"].to_numpy(dtype=float)
        params = SVIParams(row["a"], row["b"], row["rho"], row["m"], row["sigma"])
        grid = np.linspace(k.min(), k.max(), 100)
        ax.plot(k, iv * 100, "o", ms=3, color="#1f77b4", label="market")
        ax.plot(grid, svi_implied_vol(grid, params, t) * 100, "-", color="#d62728", label="SVI")
        ok = "arb-free" if row["arb_free"] else "ARB"
        ax.set_title(f"T={t:.2f}y  rmse={row['rmse']:.1e}  {ok}", fontsize=9)
        ax.set_xlabel("log-moneyness ln(K/F)")
        ax.grid(True, alpha=0.3)
    axes[0][0].set_ylabel("implied vol (%)")
    axes[0][0].legend(fontsize=8)
    src = "synthetic" if surface.synthetic else args.ticker.upper()
    fig.suptitle(f"SVI fit vs market smile ({src})", fontsize=11)
    fig.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    suffix = "synthetic" if surface.synthetic else args.ticker.lower()
    path = f"{OUT_DIR}/svi_fit_{suffix}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"\n[svi] wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
