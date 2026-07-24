"""Validate our implied-vol solver against vendor IVs on a real option chain,
and draw the crisis-day surface.

This is the options-side analogue of the Phase 1 quadrature check: an
independent implementation (OptionsDX's own IV, computed by whatever engine they
use) agreeing with ours is evidence the solver is right on real quotes, not just
on prices it generated itself.

Data: an OptionsDX end-of-day sample for SPY on 2020-03-06 (the COVID crash
Friday). Pass its path as the first argument or via OPTIONSDX_SPY. The file is
not committed; see the README for where to get it.

We do NOT tune our solver to match the vendor. Expect systematic gaps: the
vendor's risk-free rate, dividend assumption and day count are unknown, and SPY
options are American while we price European, so an early-exercise premium sits
in in-the-money puts. Agreement to a vol point or so near the money, with wider
disagreement in the wings, is the success condition. Everything is reported in
volatility points (1 point = 0.01 = 1%).

    python scripts/validate_optionsdx.py [path/to/spy_sample.csv]
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quant_system.config import default_config
from quant_system.options import build_surface, load_optionsdx_csv
from quant_system.options.chain import _clean_quotes
from quant_system.options.implied_vol import implied_volatility_detailed
from quant_system.options.pricing import CALL, PUT

OUT_DIR = "docs/results"
# March 2020: the Fed had just cut to near zero. SPY yield ~1.8% then. The vendor
# used its own unknown values; these are documented, defensible stand-ins.
RATE = 0.005
DIVIDEND_YIELD = 0.018
DEFAULT_PATH = os.environ.get("OPTIONSDX_SPY",
                              os.path.expanduser("~/Downloads/spy_sample-1.csv"))


def _forward(spot, t):
    return spot * np.exp((RATE - DIVIDEND_YIELD) * t)


def _clean_merged(day, snapshot, cfg):
    """Tradeable rows for a snapshot with vendor IV rejoined. No solving here."""
    v = day.vendor_at(snapshot)
    raw = v[["expiry", "time_to_expiry", "strike", "option_type",
             "bid", "ask", "volume", "open_interest"]].copy()
    clean, _ = _clean_quotes(raw, cfg)
    keys = ["expiry", "strike", "option_type"]
    merged = clean.merge(v[keys + ["vendor_iv", "vendor_delta"]], on=keys, how="left")
    spot = float(v["underlying_last"].iloc[0])
    merged["log_moneyness"] = np.log(merged["strike"] / _forward(spot, merged["time_to_expiry"]))
    return merged, spot


def _solve(frame, spot):
    """Add our_iv and reason to a frame by solving each row's mid price."""
    ours, reasons = [], []
    for row in frame.itertuples():
        res = implied_volatility_detailed(
            price=float(row.mid), spot=spot, strike=float(row.strike),
            time_to_expiry=float(row.time_to_expiry), rate=RATE,
            option_type=row.option_type, dividend_yield=DIVIDEND_YIELD)
        ours.append(res.vol if res.ok else np.nan)
        reasons.append(res.reason)
    out = frame.copy()
    out["our_iv"] = ours
    out["reason"] = reasons
    return out


def _bucket_report(df):
    both = df[df["our_iv"].notna() & df["vendor_iv"].notna()].copy()
    both["diff_pts"] = (both["our_iv"] - both["vendor_iv"]).abs() * 100

    k_bins = [(-np.inf, -0.10, "put wing  (k<-0.10)"),
              (-0.10, 0.10, "near ATM  (|k|<0.10)"),
              (0.10, np.inf, "call wing (k>0.10)")]
    d_bins = [(0, 30, "<30d"), (30, 90, "30-90d"), (90, 365, "90-365d"),
              (365, np.inf, ">365d")]

    print("median |ours - vendor| in vol points, by log-moneyness x DTE:\n")
    print("| moneyness | " + " | ".join(lbl for *_, lbl in d_bins) + " |")
    print("|" + "---|" * (len(d_bins) + 1))
    for lo, hi, klbl in k_bins:
        cells = []
        for dlo, dhi, _ in d_bins:
            m = ((both["log_moneyness"] >= lo) & (both["log_moneyness"] < hi)
                 & (both["time_to_expiry"] * 365 >= dlo) & (both["time_to_expiry"] * 365 < dhi))
            cells.append(f"{both.loc[m, 'diff_pts'].median():.2f}" if m.sum() else "-")
        print(f"| {klbl} | " + " | ".join(cells) + " |")

    atm = both[(both["log_moneyness"].abs() < 0.03)
               & (both["time_to_expiry"] * 365 >= 20)
               & (both["time_to_expiry"] * 365 <= 45)]
    print(f"\nATM (|k|<0.03, 20-45d): {len(atm)} quotes, "
          f"median gap {atm['diff_pts'].median():.2f} vol pts, "
          f"90th pct {atm['diff_pts'].quantile(0.9):.2f}")
    print(f"overall: {len(both)} comparable quotes, "
          f"median gap {both['diff_pts'].median():.2f} vol pts")

    declined = df[df["our_iv"].isna()]
    vendor_priced = declined[declined["vendor_iv"].notna()]
    print(f"\nour solver declined on {len(declined)} of {len(df)} tradeable quotes; "
          f"of those, the vendor still printed an IV on {len(vendor_priced)}")
    if len(declined):
        print("  decline reasons: "
              + ", ".join(f"{r}={n}" for r, n in declined['reason'].value_counts().items()))


def _atm_curve(day, snapshot, cfg):
    """ATM term structure: per expiry, solve our IV at the strike nearest forward."""
    merged, spot = _clean_merged(day, snapshot, cfg)
    picks = []
    for _, g in merged.groupby("time_to_expiry"):
        picks.append(g.iloc[g["log_moneyness"].abs().to_numpy().argmin()])
    near = pd.DataFrame(picks)
    solved = _solve(near, spot)
    solved = solved[solved["our_iv"].notna()]
    return (pd.DataFrame({"dte": solved["time_to_expiry"] * 365, "vol": solved["our_iv"]})
            .sort_values("dte").reset_index(drop=True), spot)


def main():
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    if not os.path.exists(path):
        print(f"[validate] file not found: {path}")
        print("[validate] pass the OptionsDX SPY sample path as an argument, "
              "or set OPTIONSDX_SPY. It is not committed; see README.")
        return 1

    cfg = default_config().options
    day = load_optionsdx_csv(path)
    print("\n".join(day.summary()))
    snaps = day.snapshots
    last = snaps[-1]

    print(f"\n=== IV validation, last snapshot ({last:.2f}h ET), "
          f"rate {RATE:.1%}, div {DIVIDEND_YIELD:.1%} ===\n", flush=True)
    merged, spot = _clean_merged(day, last, cfg)
    _bucket_report(_solve(merged, spot))

    print("\n=== near-ATM agreement across all snapshots ===", flush=True)
    per_snap = []
    for s in snaps:
        m, sp = _clean_merged(day, s, cfg)
        near = m[m["log_moneyness"].abs() < 0.05]
        solved = _solve(near, sp)
        both = solved[solved["our_iv"].notna() & solved["vendor_iv"].notna()]
        if len(both):
            per_snap.append(((both["our_iv"] - both["vendor_iv"]).abs() * 100).median())
    print(f"{len(per_snap)} snapshots, near-ATM median gap: "
          f"min {min(per_snap):.2f}, median {np.median(per_snap):.2f}, "
          f"max {max(per_snap):.2f} vol pts")

    # === figure: crisis surface ===
    os.makedirs(OUT_DIR, exist_ok=True)
    chain = day.chain_at(last, cfg)
    surface = build_surface(chain, rate=RATE, dividend_yield=DIVIDEND_YIELD, cfg=cfg)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.6))
    cmap = plt.get_cmap("viridis")
    expiries = list(surface.expiries)[:12]
    for i, exp in enumerate(expiries):
        sl = surface.slice(exp)
        ax1.plot(sl["log_moneyness"], sl["implied_vol"] * 100, "-",
                 color=cmap(i / max(len(expiries) - 1, 1)), lw=1.1,
                 label=f"{sl['time_to_expiry'].iloc[0] * 365:.0f}d")
    ax1.axvline(0.0, color="grey", lw=0.7, ls="--")
    ax1.set_xlabel("log forward moneyness  ln(K/F)")
    ax1.set_ylabel("implied vol (%)")
    ax1.set_title(f"Smile by expiry ({last:.1f}h)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=6, ncol=2)

    for s, colour in zip([snaps[0], snaps[len(snaps) // 2], last],
                         ["#1f77b4", "#ff7f0e", "#d62728"]):
        curve, _ = _atm_curve(day, s, cfg)
        curve = curve[curve["dte"] <= 400]
        ax2.plot(curve["dte"], curve["vol"] * 100, "o-", ms=2, lw=1.0,
                 color=colour, label=f"{s:.1f}h")
    ax2.set_xlabel("days to expiry")
    ax2.set_ylabel("ATM implied vol (%)")
    ax2.set_title("ATM term structure (inverted)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8, title="snapshot")

    intraday = []
    for s in snaps:
        curve, sp = _atm_curve(day, s, cfg)
        if len(curve) >= 2:
            intraday.append((s, float(np.interp(30, curve["dte"], curve["vol"])) * 100, sp))
    idf = pd.DataFrame(intraday, columns=["hour", "atm30", "spot"])
    ax3.plot(idf["hour"], idf["atm30"], "o-", ms=3, color="#d62728")
    ax3b = ax3.twinx()
    ax3b.plot(idf["hour"], idf["spot"], "s-", ms=3, color="#1f77b4")
    ax3.set_xlabel("hour of 2020-03-06 (ET)")
    ax3.set_ylabel("ATM 30d implied vol (%)", color="#d62728")
    ax3b.set_ylabel("SPY", color="#1f77b4")
    ax3.set_title("Intraday: vol vs spot")
    ax3.grid(True, alpha=0.3)

    fig.suptitle("SPY option surface, 2020-03-06 (COVID crash), real OptionsDX quotes")
    fig.tight_layout()
    path_png = f"{OUT_DIR}/vol_surface_spy_crisis.png"
    fig.savefig(path_png, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print("\n=== crisis surface (last snapshot) ===")
    print("\n".join(surface.summary()))
    print(f"\n[validate] wrote {path_png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
