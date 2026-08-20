"""Fit one arbitrage-free SSVI surface to a real chain and overlay it per expiry.

Builds a surface from an option chain (live by default, synthetic with --synthetic),
fits a single SSVI surface across all expiries (one skew, one power-law curvature,
a monotone at-the-money variance term structure), and plots the market implied
vols against the SSVI-surface fit for the best-populated expiries. The whole
surface is free of butterfly and calendar arbitrage by construction, so unlike
the per-expiry raw-SVI fit it trades some per-slice fit for a globally consistent,
arbitrage-free surface. Prints the surface parameters and the theta term structure.

Live mode refuses to fall back to synthetic data.

Writes docs/results/ssvi_surface_<ticker>.png.

Usage:  python scripts/build_ssvi_surface.py [--ticker SPY] [--synthetic]
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
from quant_system.options import build_surface, load_option_chain, synthetic_option_chain
from quant_system.options.ssvi import (
    fit_ssvi_surface, ssvi_surface_arbitrage_free, ssvi_surface_calendar_free,
    ssvi_surface_slice, ssvi_total_variance,
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
            print(f"[ssvi] live chain unavailable for {args.ticker}: {exc}")
            print("[ssvi] refusing to substitute synthetic data in live mode; pass --synthetic")
            return 1

    surface = build_surface(chain, rate=args.rate,
                            dividend_yield=args.dividend_yield, cfg=cfg.options)
    fit, diag = fit_ssvi_surface(surface.points)
    butterfly, calendar_sufficient = ssvi_surface_arbitrage_free(fit)
    calendar_direct = ssvi_surface_calendar_free(fit)

    print(f"\nSSVI surface: rho={fit.rho:+.3f}  eta={fit.eta:.3f}  gamma={fit.gamma:.3f}")
    print(f"butterfly-free={butterfly}")
    print(f"calendar-free: sufficient-condition={calendar_sufficient}  "
          f"direct-pointwise={calendar_direct}")
    if calendar_direct and not calendar_sufficient:
        print("  (the surface is calendar-free in fact; the sufficient bound just "
              "binds at the steep front month - sufficient, not necessary)")
    print("\nterm structure:")
    print(diag[["time_to_expiry", "theta", "psi", "rmse", "n_points"]].to_string(index=False))

    top = diag.sort_values("n_points", ascending=False).head(4).sort_values("time_to_expiry")
    order = {float(t): i for i, t in enumerate(fit.maturities)}
    n = len(top)
    figure, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.6), squeeze=False)
    for ax, (_, row) in zip(axes[0], top.iterrows()):
        t = float(row["time_to_expiry"])
        sel = surface.points[surface.points["time_to_expiry"] == t].sort_values("log_moneyness")
        k = sel["log_moneyness"].to_numpy(dtype=float)
        iv = sel["implied_vol"].to_numpy(dtype=float)
        grid = np.linspace(k.min(), k.max(), 100)
        w = ssvi_total_variance(grid, ssvi_surface_slice(fit, order[t]))
        ax.plot(k, iv * 100, "o", ms=3, color="#1f77b4", label="market")
        ax.plot(grid, np.sqrt(np.maximum(w, 0.0) / t) * 100, "-", color="#2ca02c", label="SSVI surface")
        ax.set_title(f"T={t:.2f}y  rmse={row['rmse']:.1e}", fontsize=9)
        ax.set_xlabel("log-moneyness ln(K/F)")
        ax.grid(True, alpha=0.3)
    axes[0][0].set_ylabel("implied vol (%)")
    axes[0][0].legend(fontsize=8)
    src = "synthetic" if surface.synthetic else args.ticker.upper()
    verdict = ("butterfly + calendar arbitrage-free" if (butterfly and calendar_direct)
               else "see printout")
    figure.suptitle(f"Arbitrage-free SSVI surface ({src}) - {verdict}", fontsize=11)
    figure.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    suffix = "synthetic" if surface.synthetic else args.ticker.lower()
    path = f"{OUT_DIR}/ssvi_surface_{suffix}.png"
    figure.savefig(path, dpi=120, bbox_inches="tight")
    print(f"\n[ssvi] wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
