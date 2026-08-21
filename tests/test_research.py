"""Tests for the shared research helpers."""

import numpy as np
import pandas as pd

from quant_system.config import default_config
from quant_system.data.loader import load_price_data
from quant_system.data.universe import FACTOR_ETFS
from quant_system.research import research_universe, sleeve_makers

CFG = default_config()


def test_research_universe_is_sorted_unique_and_covers_the_sleeves():
    tickers = research_universe()
    assert tickers == sorted(set(tickers))               # sorted and de-duplicated
    assert FACTOR_ETFS["market"] in tickers              # the market ETF is present
    assert len(tickers) > 30


def _panel():
    return load_price_data(research_universe()[:12], "2018-01-01", "2021-12-31",
                           use_synthetic=True)


def test_sleeve_makers_return_the_three_sleeves():
    makers = sleeve_makers(CFG, regime_labels=None)
    assert set(makers) == {"momentum", "pairs", "ml"}


def test_each_sleeve_produces_weights_on_the_panel_index():
    panel = _panel()
    makers = sleeve_makers(CFG, regime_labels=None)
    fit_end = panel.close.index[-1]
    for name, make in makers.items():
        w = make(panel, fit_end)
        assert list(w.index) == list(panel.close.index), name
        assert np.isfinite(w.to_numpy()).all(), name


def test_regime_labels_scale_the_directional_sleeves():
    panel = _panel()
    idx = panel.close.index
    labels = pd.Series(1.0, index=idx)                   # every day defensive
    plain = sleeve_makers(CFG, regime_labels=None)["momentum"](panel, idx[-1])
    scaled = sleeve_makers(CFG, regime_labels=labels, defensive_scale=0.5)["momentum"](panel, idx[-1])
    # Every day is defensive, so the scaled book is half the gross of the plain one.
    plain_gross = plain.abs().sum(axis=1)
    scaled_gross = scaled.abs().sum(axis=1)
    mask = plain_gross > 0
    assert np.allclose(scaled_gross[mask].to_numpy(), 0.5 * plain_gross[mask].to_numpy())
