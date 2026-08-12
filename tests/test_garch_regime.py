"""Tests for the GARCH-forecast regime definition."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("arch")

from quant_system.config import default_config
from quant_system.data.loader import load_price_data
from quant_system.data.universe import universe
from quant_system.regime.detector import garch_regime, garch_regime_labels


def _calm_then_wild(seed=0):
    rng = np.random.default_rng(seed)
    calm = rng.normal(0.0, 0.006, 500)                  # ~9.5% annual
    wild = rng.normal(0.0, 0.030, 260)                  # a year of ~5x vol
    idx = pd.bdate_range("2016-01-01", periods=len(calm) + len(wild))
    return pd.Series(np.concatenate([calm, wild]), index=idx)


def test_labels_are_binary_with_a_warmup():
    labels = garch_regime_labels(_calm_then_wild(), min_obs=250)
    assert set(labels.dropna().unique()) <= {0.0, 1.0}
    assert labels.iloc[:250].isna().all()               # not enough history yet


def test_defensive_fires_more_in_the_wild_regime():
    labels = garch_regime_labels(_calm_then_wild(seed=1), min_obs=250)
    calm_part = labels.iloc[250:500].mean()             # tail of the calm block
    wild_part = labels.iloc[520:].mean()                # inside the wild block
    assert wild_part > calm_part
    assert wild_part > 0.5                              # mostly defensive when vol spikes


def test_garch_regime_wires_price_data_through():
    cfg = default_config()
    panel = load_price_data(universe("largecaps")[:6], "2016-01-01", "2020-12-31",
                            use_synthetic=True)
    labels = garch_regime(panel, cfg.regime)
    assert labels.name == "regime_garch"
    assert set(labels.dropna().unique()) <= {0.0, 1.0}
    assert len(labels) == panel.close.shape[0]
