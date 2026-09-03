"""Run a causal replication study of Moreira and Muir (2017).

The study compares the Ken French market excess return (Mkt-RF) with the same
return scaled by inverse trailing realised variance. It uses an expanding
walk-forward design, so each test fold has a scale multiplier fitted only on
earlier returns. Results are gross of transaction costs: Fama-French factors
are research series, not directly tradable instruments.

Usage:
    python scripts/build_volatility_managed_results.py

Outputs:
    docs/results/volatility_managed_metrics.csv
    docs/results/volatility_managed_equity.png
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from quant_system.performance.analytics import compute_metrics
from quant_system.replications.volatility_managed import (
    download_ken_french_daily,
    walk_forward_volatility_managed,
)


OUT_DIR = "docs/results"
TRAIN_DAYS = 252 * 5
TEST_DAYS = 252
VOL_LOOKBACK = 21


def _metric_table(result) -> pd.DataFrame:
    """Build the small comparison table written by the study runner."""
    rows = {}
    for name, returns in (("Unmanaged Mkt-RF", result.unmanaged_returns),
                          ("Volatility-managed Mkt-RF", result.managed_returns)):
        metrics = compute_metrics(returns)
        rows[name] = {
            "sharpe": metrics["sharpe"],
            "ann_return": metrics["ann_return"],
            "ann_vol": metrics["ann_vol"],
            "max_drawdown": metrics["max_drawdown"],
        }
    rows["Volatility-managed Mkt-RF"]["ann_exposure_turnover"] = float(result.turnover.mean() * 252)
    return pd.DataFrame.from_dict(rows, orient="index")


def _markdown_table(table: pd.DataFrame) -> str:
    """Format a compact Markdown table without requiring an optional package."""
    columns = list(table.columns)
    lines = ["| Strategy | " + " | ".join(columns) + " |",
             "|---|" + "|".join("---:" for _ in columns) + "|"]
    for name, row in table.iterrows():
        values = ["" if pd.isna(row[column]) else f"{row[column]:.4f}" for column in columns]
        lines.append("| " + str(name) + " | " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> int:
    """Download the official factor file, run the study, and save its outputs."""
    print("[vol-managed] downloading the Ken French daily factor file")
    factors = download_ken_french_daily()
    if "Mkt-RF" not in factors:
        raise ValueError("official factor file did not contain Mkt-RF")

    result = walk_forward_volatility_managed(
        factors["Mkt-RF"],
        train_days=TRAIN_DAYS,
        test_days=TEST_DAYS,
        vol_lookback=VOL_LOOKBACK,
    )
    table = _metric_table(result)
    span = f"{result.managed_returns.index.min().date()}..{result.managed_returns.index.max().date()}"
    print(f"[vol-managed] {len(result.folds)} expanding OOS folds, {span}")
    print("[vol-managed] gross returns; factor-series exposure turnover is a proxy, not an executable cost estimate\n")
    print(_markdown_table(table))

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = f"{OUT_DIR}/volatility_managed_metrics.csv"
    table.to_csv(csv_path, float_format="%.10f")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for label, returns, color in (
        ("Unmanaged Mkt-RF", result.unmanaged_returns, "#7f7f7f"),
        ("Volatility-managed Mkt-RF", result.managed_returns, "#1f77b4"),
    ):
        equity = (1.0 + returns).cumprod()
        ax.plot(equity.index, equity.values, lw=1.4, label=label, color=color)
    ax.axhline(1.0, color="black", lw=0.7)
    ax.set_title(f"Volatility-managed market factor, expanding OOS ({span})")
    ax.set_ylabel("Growth of $1")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    chart_path = f"{OUT_DIR}/volatility_managed_equity.png"
    fig.savefig(chart_path, dpi=120, bbox_inches="tight")
    print(f"\n[vol-managed] wrote {csv_path} and {chart_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
