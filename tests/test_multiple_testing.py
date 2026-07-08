"""Tests for the Benjamini-Hochberg FDR control."""

import numpy as np
import pandas as pd

from quant_system.performance.multiple_testing import (
    benjamini_hochberg, fdr_adjusted_pvalues, fdr_report, fdr_summary_lines,
)
from quant_system.data.loader import load_price_data
from quant_system.data.universe import PAIRS_CANDIDATES
from quant_system.signals.mean_reversion import find_cointegrated_pair, scan_candidate_pairs


def test_single_strong_pvalue_survives_among_nulls():
    p = [0.0001, 0.45, 0.62, 0.71, 0.88, 0.93]
    keep = benjamini_hochberg(p, alpha=0.10)
    assert keep[0]
    assert not keep[1:].any()


def test_all_null_pvalues_yield_no_discoveries():
    rng = np.random.default_rng(3)
    p = rng.uniform(0.2, 1.0, 50)
    assert not benjamini_hochberg(p, alpha=0.10).any()


def test_keep_set_matches_qvalue_threshold():
    # BH rejection at level alpha is the same as q <= alpha. Check the identity.
    rng = np.random.default_rng(5)
    p = np.concatenate([rng.uniform(0, 0.01, 5), rng.uniform(0.05, 1.0, 45)])
    for alpha in (0.05, 0.10, 0.25):
        keep = benjamini_hochberg(p, alpha)
        q = fdr_adjusted_pvalues(p)
        assert np.array_equal(keep, q <= alpha)


def test_qvalues_bounded_and_at_least_p():
    p = [0.001, 0.02, 0.3, 0.9, np.nan]
    q = fdr_adjusted_pvalues(p)
    assert np.isnan(q[-1])
    valid = q[:4]
    assert (valid >= np.array(p[:4]) - 1e-12).all()
    assert (valid <= 1.0).all()


def test_report_and_summary_shape():
    rep = fdr_report(["a/b", "c/d", "e/f"], [0.5, 0.001, 0.7], alpha=0.10)
    assert list(rep["candidate"]) == ["c/d", "a/b", "e/f"]   # sorted by p
    lines = fdr_summary_lines(rep, 0.10)
    assert len(lines) == 5
    assert "1 of 3" in lines[-1]


def test_pair_selection_with_fdr_still_finds_injected_pair():
    tickers = sorted({t for pair in PAIRS_CANDIDATES for t in pair})
    panel = load_price_data(tickers, "2017-01-01", "2023-12-31", use_synthetic=True)
    scanned = scan_candidate_pairs(panel.close, PAIRS_CANDIDATES)
    assert len(scanned) >= 2
    best = find_cointegrated_pair(panel.close, PAIRS_CANDIDATES,
                                  max_pvalue=0.10, fdr_alpha=0.10)
    assert best is not None
    assert best[2] < 0.05                                    # injected pair is real
