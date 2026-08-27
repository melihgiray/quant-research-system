"""Tests for the self-contained HTML tearsheet."""

import numpy as np
import pandas as pd

from quant_system.performance.html_report import build_html_report


def _returns(n=600, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(rng.normal(0.0004, 0.009, n), index=idx)


def test_report_is_one_self_contained_html_file():
    r = _returns()
    html = build_html_report(r, benchmark=_returns(seed=1), title="Blended book")
    assert html.lstrip().startswith("<!DOCTYPE html>")
    assert "Blended book" in html
    assert "Sharpe" in html
    # Charts are embedded, not linked: base64 data URIs and no external http assets.
    assert "data:image/png;base64," in html
    assert "http://" not in html and "https://" not in html


def test_report_lists_every_year_in_the_sample():
    r = _returns()                                       # spans 2020..2022
    html = build_html_report(r)
    for year in sorted(set(r.index.year)):
        assert f"<td>{year}</td>" in html


def test_report_works_without_a_benchmark():
    html = build_html_report(_returns(), benchmark=None)
    assert "Rolling beta" not in html                    # beta panel omitted
    assert "Rolling Sharpe" in html
    assert "Versus benchmark" not in html                # no benchmark block either


def test_benchmark_block_shows_active_metrics():
    html = build_html_report(_returns(), benchmark=_returns(seed=1))
    assert "Versus benchmark" in html
    assert "Information ratio" in html
    assert "Up capture" in html
    assert "Tracking error" in html
