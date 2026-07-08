"""Multiple-testing control for strategy searches.

Testing several candidate pairs for cointegration is a multiple-comparisons
problem: run enough tests and something clears p < 0.05 by luck alone. The
Benjamini-Hochberg procedure controls the false discovery rate, the expected
share of the "discoveries" that are actually noise. It is less brutal than a
Bonferroni correction (which controls the chance of even one false positive)
and is the standard choice when you expect a few real effects among the
candidates.

How it works: sort the m p-values ascending, find the largest k such that
p_(k) <= alpha * k / m, and reject hypotheses 1..k. Rejections are the
candidates you can keep with FDR held at alpha.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd


def benjamini_hochberg(pvalues: Sequence[float], alpha: float = 0.10) -> np.ndarray:
    """Which hypotheses survive at false discovery rate ``alpha``.

    Parameters
    ----------
    pvalues : sequence of float
        One p-value per hypothesis, in any order. NaNs never survive.
    alpha : float
        Target FDR. 0.10 means at most ~10% of the kept discoveries are
        expected to be false.

    Returns
    -------
    np.ndarray of bool
        Same length and order as the input; True where the hypothesis is kept.
    """
    p = np.asarray(pvalues, dtype=float)
    keep = np.zeros(len(p), dtype=bool)
    valid = np.isfinite(p)
    m = int(valid.sum())
    if m == 0:
        return keep

    idx = np.where(valid)[0]
    order = idx[np.argsort(p[idx])]
    ranked = p[order]
    thresholds = alpha * np.arange(1, m + 1) / m
    passing = np.where(ranked <= thresholds)[0]
    if passing.size == 0:
        return keep
    k = passing.max()
    keep[order[: k + 1]] = True
    return keep


def fdr_adjusted_pvalues(pvalues: Sequence[float]) -> np.ndarray:
    """BH-adjusted p-values (q-values).

    q_i is the smallest FDR at which hypothesis i would still be kept, which
    makes it the honest number to report next to a raw p-value from a search.
    Computed as the running minimum of p_(k) * m / k from the largest rank down.
    """
    p = np.asarray(pvalues, dtype=float)
    q = np.full(len(p), np.nan)
    valid = np.isfinite(p)
    m = int(valid.sum())
    if m == 0:
        return q

    idx = np.where(valid)[0]
    order = idx[np.argsort(p[idx])]
    ranked = p[order]
    raw = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(raw[::-1])[::-1]
    q[order] = np.clip(adjusted, 0.0, 1.0)
    return q


def fdr_report(names: Sequence[str], pvalues: Sequence[float],
               alpha: float = 0.10) -> pd.DataFrame:
    """Table of raw p-values, q-values and the keep/drop verdict per candidate.

    Sorted by p-value so the strongest candidate is on top.
    """
    keep = benjamini_hochberg(pvalues, alpha)
    q = fdr_adjusted_pvalues(pvalues)
    df = pd.DataFrame({
        "candidate": list(names),
        "pvalue": np.asarray(pvalues, dtype=float),
        "qvalue": q,
        "keep": keep,
    })
    return df.sort_values("pvalue").reset_index(drop=True)


def fdr_summary_lines(report: pd.DataFrame, alpha: float) -> List[str]:
    """Short text block for the CLI: the scan table plus the verdict."""
    lines = [f"PAIR SCAN, FDR controlled at {alpha:.0%} (Benjamini-Hochberg)"]
    for _, row in report.iterrows():
        verdict = "keep" if row["keep"] else "drop"
        lines.append(f" {row['candidate']:<10} p={row['pvalue']:.2e}  "
                     f"q={row['qvalue']:.2e}  {verdict}")
    n_keep = int(report["keep"].sum())
    lines.append(f" {n_keep} of {len(report)} candidate(s) survive the correction")
    return lines
