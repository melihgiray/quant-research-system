"""Tests for FDR control over feature-importance p-values."""

import numpy as np
import pandas as pd
import pytest

from quant_system.signals.feature_selection import (
    fdr_control_features, feature_fdr_summary,
)


def _table(pvalues, names=None):
    names = names or [f"f{i}" for i in range(len(pvalues))]
    return pd.DataFrame({"importance": np.linspace(0.1, 0.01, len(pvalues)),
                         "p_value": pvalues}, index=pd.Index(names, name="feature"))


def test_one_strong_feature_survives_a_field_of_noise():
    report = fdr_control_features(_table([0.001, 0.4, 0.6, 0.8, 0.95]), alpha=0.10)
    assert report.iloc[0].name == "f0"                 # sorted by p-value
    assert bool(report.loc["f0", "keep"])
    assert not report.loc[[f"f{i}" for i in range(1, 5)], "keep"].any()
    assert int(report["keep"].sum()) == 1


def test_all_noise_keeps_nothing():
    rng = np.random.default_rng(3)
    report = fdr_control_features(_table(rng.uniform(0.2, 1.0, 8)), alpha=0.10)
    assert int(report["keep"].sum()) == 0


def test_keep_matches_qvalue_threshold():
    # BH rejection at alpha is exactly q <= alpha; check the identity holds.
    rng = np.random.default_rng(5)
    p = np.concatenate([rng.uniform(0, 0.01, 3), rng.uniform(0.1, 1.0, 5)])
    for alpha in (0.05, 0.10, 0.25):
        report = fdr_control_features(_table(p), alpha=alpha)
        assert (report["keep"] == (report["q_value"] <= alpha)).all()


def test_qvalue_is_at_least_pvalue():
    report = fdr_control_features(_table([0.001, 0.02, 0.3, 0.9]))
    assert (report["q_value"] >= report["p_value"] - 1e-12).all()


def test_missing_pvalue_column_raises():
    with pytest.raises(ValueError, match="p_value"):
        fdr_control_features(pd.DataFrame({"importance": [0.1, 0.2]}))


def test_summary_shape_and_content():
    report = fdr_control_features(_table([0.001, 0.5, 0.7]), alpha=0.10)
    lines = feature_fdr_summary(report, alpha=0.10)
    assert "FEATURE SIGNIFICANCE" in lines[0]
    assert len(lines) == 1 + 3 + 1                      # header + 3 features + count
    assert "1 of 3" in lines[-1]
