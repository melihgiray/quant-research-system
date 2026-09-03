"""Tests for the causal volatility-managed-portfolio replication."""

from io import BytesIO
from zipfile import ZipFile

import numpy as np
import pandas as pd

from quant_system.replications.volatility_managed import (
    download_ken_french_daily,
    inverse_variance_exposure,
    walk_forward_volatility_managed,
)
from scripts.build_volatility_managed_results import _markdown_table, _metric_table


def _returns(n: int = 130) -> pd.Series:
    index = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(0.001 + 0.01 * np.sin(np.arange(n)), index=index, name="Mkt-RF")


def test_exposure_on_a_day_does_not_use_that_days_return():
    returns = _returns()
    changed = returns.copy()
    changed.iloc[60] = 1.0
    base = inverse_variance_exposure(returns, lookback=21)
    candidate = inverse_variance_exposure(changed, lookback=21)
    assert base.iloc[60] == candidate.iloc[60]
    assert base.iloc[61] != candidate.iloc[61]


def test_walk_forward_returns_only_non_overlapping_oos_windows():
    result = walk_forward_volatility_managed(_returns(), train_days=63, test_days=21, vol_lookback=21)
    assert len(result.folds) == 3
    assert result.managed_returns.index.is_unique
    assert result.managed_returns.index.min() == _returns().index[63]
    assert len(result.managed_returns) == 63
    for fold in result.folds:
        assert fold["train_end"] < fold["test_start"]


def test_fold_scale_is_fitted_before_its_test_window():
    returns = _returns()
    original = walk_forward_volatility_managed(returns, train_days=63, test_days=21, vol_lookback=21)
    changed = returns.copy()
    changed.iloc[63:84] *= 50.0
    candidate = walk_forward_volatility_managed(changed, train_days=63, test_days=21, vol_lookback=21)
    assert original.folds[0]["multiplier"] == candidate.folds[0]["multiplier"]


def test_ken_french_daily_parser_converts_percent_to_decimal(monkeypatch):
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr(
            "F-F_Research_Data_Factors_daily.CSV",
            "Created by test\n,Mkt-RF,SMB,HML,RF\n20200102,1.00,2.00,-3.00,0.01\n\nFooter\n",
        )

    class Response:
        content = buffer.getvalue()

        def raise_for_status(self):
            return None

    monkeypatch.setattr("quant_system.replications.volatility_managed.requests.get", lambda *args, **kwargs: Response())
    frame = download_ken_french_daily()
    assert frame.loc[pd.Timestamp("2020-01-02"), "Mkt-RF"] == 0.01
    assert frame.loc[pd.Timestamp("2020-01-02"), "HML"] == -0.03


def test_results_table_has_no_optional_formatter_dependency():
    table = _metric_table(walk_forward_volatility_managed(_returns(), train_days=63, test_days=21, vol_lookback=21))
    rendered = _markdown_table(table)
    assert "| Strategy |" in rendered
    assert "Volatility-managed Mkt-RF" in rendered
