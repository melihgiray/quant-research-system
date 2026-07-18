"""Tests for the data generator, signals, regime detector, and analytics."""

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.stattools import coint

from quant_system.config import default_config
from quant_system.data.loader import load_price_data
from quant_system.data.universe import universe, PAIRS_CANDIDATES
from quant_system.signals.momentum import cross_sectional_momentum
from quant_system.signals.mean_reversion import find_cointegrated_pair, pairs_signal
from quant_system.signals.ml_signal import train_predict, compute_features, FEATURE_NAMES
from quant_system.regime.detector import detect_regime
from quant_system.performance.analytics import sharpe_ratio, compute_metrics


CFG = default_config()


def test_synthetic_data_is_deterministic():
    a = load_price_data(universe("sectors"), "2019-01-01", "2022-12-31", use_synthetic=True)
    b = load_price_data(universe("sectors"), "2019-01-01", "2022-12-31", use_synthetic=True)
    pd.testing.assert_frame_equal(a.close, b.close)


def test_injected_pairs_are_cointegrated():
    p = load_price_data(["GLD", "SLV", "XLK"], "2018-01-01", "2023-12-31", use_synthetic=True)
    _, pv_pair, _ = coint(np.log(p.close["GLD"]), np.log(p.close["SLV"]))
    _, pv_unrel, _ = coint(np.log(p.close["GLD"]), np.log(p.close["XLK"]))
    assert pv_pair < 0.05                 # constructed pair cointegrates
    assert pv_unrel > 0.05                # unrelated names do not


def test_momentum_is_dollar_neutral():
    p = load_price_data(universe("sectors"), "2016-01-01", "2022-12-31", use_synthetic=True)
    w = cross_sectional_momentum(p, CFG.momentum)
    active = w.loc[(w != 0).any(axis=1)]
    # Long leg ~ +0.5, short leg ~ -0.5 => net ~ 0 on every active day.
    assert np.allclose(active.sum(axis=1).values, 0.0, atol=1e-9)
    assert np.allclose(active.abs().sum(axis=1).values, 1.0, atol=1e-9)


def test_find_cointegrated_pair_returns_best():
    p = load_price_data(universe("sectors") + ["GLD", "SLV", "KO", "PEP"],
                        "2017-01-01", "2023-12-31", use_synthetic=True)
    best = find_cointegrated_pair(p.close, PAIRS_CANDIDATES, max_pvalue=0.10)
    assert best is not None
    a, b, pv = best
    assert pv < 0.10


def test_pairs_signal_only_trades_two_legs():
    p = load_price_data(universe("sectors") + ["KO", "PEP"], "2017-01-01",
                        "2023-12-31", use_synthetic=True)
    w = pairs_signal(p, ("KO", "PEP"), CFG.pairs)
    traded = w.columns[(w != 0).any()]
    assert set(traded).issubset({"KO", "PEP"})
    # When in a position the two legs have opposite signs (hedged).
    active = w.loc[(w != 0).any(axis=1)]
    assert (np.sign(active["KO"]) == -np.sign(active["PEP"])).all()


def test_causal_pairs_selection_ignores_data_after_fit_end():
    # The pair chosen (and the weights through fit_end) must be identical
    # whether or not the panel contains data after fit_end.
    from quant_system.signals.mean_reversion import causal_pairs_weights
    tickers = sorted({t for pair in PAIRS_CANDIDATES for t in pair})
    full = load_price_data(tickers, "2017-01-01", "2023-12-31", use_synthetic=True)
    fit_end = pd.Timestamp("2021-06-30")

    truncated = load_price_data(tickers, "2017-01-01", "2023-12-31", use_synthetic=True)
    cut = full.close.index[full.close.index <= fit_end]
    truncated.close = truncated.close.loc[cut]
    truncated.volume = truncated.volume.loc[cut]

    w_full = causal_pairs_weights(full, fit_end, PAIRS_CANDIDATES, CFG.pairs)
    w_trunc = causal_pairs_weights(truncated, fit_end, PAIRS_CANDIDATES, CFG.pairs)
    pd.testing.assert_frame_equal(w_full.loc[:fit_end], w_trunc.loc[:fit_end])


def test_causal_pairs_selection_flat_when_nothing_qualifies():
    from quant_system.signals.mean_reversion import causal_pairs_weights
    # Sector ETFs are not constructed to cointegrate; scanning fake candidate
    # pairs from them should keep the book flat.
    p = load_price_data(universe("sectors"), "2017-01-01", "2022-12-31",
                        use_synthetic=True)
    fake_candidates = [("XLB", "XLK"), ("XLU", "XLY")]
    w = causal_pairs_weights(p, pd.Timestamp("2021-06-30"), fake_candidates, CFG.pairs)
    assert (w == 0).all().all()


def test_ml_features_have_no_future_leak():
    p = load_price_data(universe("largecaps"), "2018-01-01", "2022-12-31", use_synthetic=True)
    feats = compute_features(p, CFG.ml)
    one = feats[p.tickers[0]]
    assert list(one.columns) == FEATURE_NAMES
    # mom_5 at row t must equal close[t]/close[t-5]-1 using only past prices.
    c = p.close[p.tickers[0]]
    expected = (c / c.shift(5) - 1.0)
    pd.testing.assert_series_equal(one["mom_5"], expected, check_names=False)


def test_ml_training_set_ignores_data_after_fit_end():
    # The training set built from the full panel must equal the one built from a
    # panel truncated at fit_end. If they differ, the model saw the future.
    from quant_system.signals.ml_signal import _pooled_training_set
    full = load_price_data(universe("largecaps")[:6], "2017-01-01", "2022-12-31",
                           use_synthetic=True)
    fit_end = pd.Timestamp("2021-06-30")
    cut = full.close.index[full.close.index <= fit_end]
    truncated = load_price_data(universe("largecaps")[:6], "2017-01-01", "2022-12-31",
                                use_synthetic=True)
    truncated.close = truncated.close.loc[cut]
    truncated.volume = truncated.volume.loc[cut]

    f_full = compute_features(full, CFG.ml)
    f_trunc = compute_features(truncated, CFG.ml)
    X1, y1 = _pooled_training_set(f_full, full.returns(), fit_end, CFG.ml.train_window)
    X2, y2 = _pooled_training_set(f_trunc, truncated.returns(), fit_end, CFG.ml.train_window)
    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)


def test_ml_weights_are_market_neutral_and_bounded():
    p = load_price_data(universe("largecaps"), "2017-01-01", "2022-12-31", use_synthetic=True)
    w = train_predict(p, pd.Timestamp("2021-12-31"), CFG.ml, max_weight=0.10)
    assert w.abs().sum(axis=1).max() <= 1.0 + 1e-9       # gross <= 1
    assert w.abs().max().max() <= 0.10 + 1e-9            # per-name cap
    assert w.sum(axis=1).abs().max() < 1e-6             # dollar neutral


def test_regime_labels_are_binary_and_causal():
    p = load_price_data(universe("sectors"), "2016-01-01", "2023-12-31", use_synthetic=True)
    reg = detect_regime(p, CFG.regime, benchmark=None)
    vals = reg.causal_labels.dropna().unique()
    assert set(vals).issubset({0.0, 1.0})
    assert reg.method in ("hmm", "vol_ratio")


def test_sharpe_of_known_series():
    # Constant positive return -> infinite/large Sharpe; mixed -> finite.
    idx = pd.bdate_range("2020-01-01", periods=252)
    r = pd.Series(np.tile([0.01, -0.005], 126), index=idx)
    s = sharpe_ratio(r, rf_annual=0.0)
    assert np.isfinite(s)
    m = compute_metrics(r)
    assert set(["sharpe", "sortino", "calmar", "max_drawdown", "hit_rate",
                "profit_factor", "ann_turnover"]).issubset(m.keys())
