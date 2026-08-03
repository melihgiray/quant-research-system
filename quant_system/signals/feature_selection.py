"""False-discovery-rate control over feature-importance p-values.

``permutation_importance_pvalues`` gives a p-value per feature, but a model has
several features, and testing them all inflates the chance that a pure-noise
feature clears p < 0.05 somewhere. The Benjamini-Hochberg procedure, already
used for the pairs scan, controls the false discovery rate across the whole
feature set, so a feature called significant is significant after accounting for
how many were tested.
"""

from __future__ import annotations

from typing import List

import pandas as pd

from ..performance.multiple_testing import benjamini_hochberg, fdr_adjusted_pvalues


def fdr_control_features(importance_pvalues: pd.DataFrame,
                        alpha: float = 0.10) -> pd.DataFrame:
    """Add q-values and a keep flag to a feature importance/p-value table.

    Parameters
    ----------
    importance_pvalues : pd.DataFrame
        Indexed by feature, with at least a ``p_value`` column (as returned by
        :func:`quant_system.signals.permutation_importance_pvalues`). An
        ``importance`` column is carried through if present.
    alpha : float
        Target false discovery rate.

    Returns
    -------
    pd.DataFrame
        The input with ``q_value`` and ``keep`` columns added, sorted by p-value
        so the strongest feature is on top.
    """
    if "p_value" not in importance_pvalues.columns:
        raise ValueError("importance_pvalues must have a 'p_value' column")
    df = importance_pvalues.copy()
    p = df["p_value"].to_numpy(dtype=float)
    df["q_value"] = fdr_adjusted_pvalues(p)
    df["keep"] = benjamini_hochberg(p, alpha)
    return df.sort_values("p_value")


def feature_fdr_summary(report: pd.DataFrame, alpha: float = 0.10) -> List[str]:
    """Tearsheet-ready lines: each feature's p, q and verdict, plus the count."""
    lines = [f"ML FEATURE SIGNIFICANCE, FDR controlled at {alpha:.0%} "
             f"(Benjamini-Hochberg)"]
    has_imp = "importance" in report.columns
    for feat, row in report.iterrows():
        imp = f" imp={row['importance']:+.4f}" if has_imp else ""
        verdict = "keep" if row["keep"] else "drop"
        lines.append(f" {feat:<14}{imp}  p={row['p_value']:.3f}  "
                     f"q={row['q_value']:.3f}  {verdict}")
    n_keep = int(report["keep"].sum())
    lines.append(f" {n_keep} of {len(report)} feature(s) survive the correction")
    return lines
