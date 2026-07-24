"""Tests for the OptionsDX historical-chain loader.

Loader behaviour is pinned against a tiny hand-built fixture (schema-identical to
the real OptionsDX file, so no license question attaches to it). One test uses
the real 35MB sample if it happens to be present, and is skipped otherwise, so
the suite stays green offline and on a machine without the download.
"""

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_system.config import OptionsConfig
from quant_system.options import build_surface, load_optionsdx_csv
from quant_system.options.chain import _clean_quotes
from quant_system.options.implied_vol import implied_volatility_detailed
from quant_system.options.pricing import CALL, PUT

FIXTURE = Path(__file__).parent / "fixtures" / "optionsdx_tiny.csv"
REAL_FILE = Path(os.environ.get("OPTIONSDX_SPY",
                                os.path.expanduser("~/Downloads/spy_sample-1.csv")))


@pytest.fixture(scope="module")
def day():
    return load_optionsdx_csv(str(FIXTURE))


def test_loads_single_day_with_two_snapshots(day):
    assert day.ticker == "SPY"
    assert day.date == pd.Timestamp("2020-03-06")
    assert day.snapshots == [9.5, 16.0]


def test_wide_rows_split_into_call_and_put(day):
    # 7 wide rows -> 7 calls + 7 puts.
    assert len(day.long) == 14
    counts = day.long["option_type"].value_counts()
    assert counts[CALL] == 7 and counts[PUT] == 7


def test_empty_vendor_iv_becomes_nan(day):
    # The 290-strike 14d call at 09:30 has a blank C_IV in the fixture.
    row = day.long[(day.long["snapshot"] == 9.5)
                   & (day.long["strike"] == 290)
                   & (day.long["option_type"] == CALL)
                   & (day.long["expiry"] == pd.Timestamp("2020-03-20"))]
    assert len(row) == 1
    assert math.isnan(row["vendor_iv"].iloc[0])
    # A neighbouring quote does have one, so the NaN is the blank, not a parse bug.
    assert day.long["vendor_iv"].notna().sum() == 13


def test_open_interest_is_nan_not_faked(day):
    # OptionsDX carries no OI column; we must not invent one.
    assert day.long["open_interest"].isna().all()


def test_time_to_expiry_from_dte(day):
    row = day.long.iloc[0]
    assert row["time_to_expiry"] == pytest.approx(14.0 / 365.0)
    # Consistent with the calendar difference.
    cal = (row["expiry"] - day.date).days
    assert abs(cal - 14) <= 1


def test_hygiene_drops_are_counted(day):
    cfg = OptionsConfig()
    chain = day.chain_at(9.5, cfg)
    # 10 long rows at 09:30, minus the zero-bid put and the crossed call.
    assert chain.n_quotes == 8
    assert chain.dropped.get("zero_or_low_bid") == 1
    assert chain.dropped.get("crossed_or_locked") == 1
    assert day.chain_at(16.0, cfg).n_quotes == 4


def test_as_of_carries_the_snapshot_time(day):
    chain = day.chain_at(9.5)
    assert chain.as_of == pd.Timestamp("2020-03-06 09:30")
    assert day.chain_at(16.0).as_of == pd.Timestamp("2020-03-06 16:00")
    assert chain.synthetic is False


def test_spot_is_per_snapshot(day):
    assert day.chain_at(9.5).spot == pytest.approx(292.97)
    assert day.chain_at(16.0).spot == pytest.approx(300.0)


def test_build_surface_flows_from_a_historical_chain(day):
    # The tiny fixture has too few strikes per expiry for the default minimum,
    # so relax it: the point of the test is that a parsed historical chain feeds
    # the existing surface builder unchanged, not that the fit is good.
    cfg = OptionsConfig(min_points_per_slice=1)
    surface = build_surface(day.chain_at(9.5, cfg), rate=0.005,
                            dividend_yield=0.018, cfg=cfg)
    assert surface.synthetic is False
    assert len(surface.points) >= 3
    assert (surface.points["implied_vol"] > 0).all()


def test_multi_date_file_is_rejected(tmp_path):
    text = FIXTURE.read_text().splitlines()
    # Duplicate the last row but move it to a different QUOTE_DATE.
    bad = text[-1].replace("2020-03-06", "2020-03-09")
    p = tmp_path / "two_days.csv"
    p.write_text("\n".join(text + [bad]) + "\n")
    with pytest.raises(RuntimeError, match="one trading date"):
        load_optionsdx_csv(str(p))


def test_missing_columns_are_rejected(tmp_path):
    lines = FIXTURE.read_text().splitlines()
    header = lines[0].replace("C_BID,", "")          # drop a required column
    p = tmp_path / "bad.csv"
    p.write_text("\n".join([header] + lines[1:]) + "\n")
    with pytest.raises(RuntimeError, match="missing columns"):
        load_optionsdx_csv(str(p))


@pytest.mark.skipif(not REAL_FILE.exists(),
                    reason="OptionsDX SPY sample not present (set OPTIONSDX_SPY)")
def test_real_file_iv_agrees_with_vendor_near_atm():
    # Regression guard on real data: our solver must agree with the vendor's IV
    # near the money to within a few vol points. This is loose on purpose; the
    # exact gap (~2 pts, driven by unknown vendor rate/dividend/American style)
    # is reported by scripts/validate_optionsdx.py, not asserted here.
    day = load_optionsdx_csv(str(REAL_FILE))
    snap = day.snapshots[-1]
    v = day.vendor_at(snap)
    raw = v[["expiry", "time_to_expiry", "strike", "option_type",
             "bid", "ask", "volume", "open_interest"]].copy()
    clean, _ = _clean_quotes(raw, OptionsConfig())
    spot = float(v["underlying_last"].iloc[0])
    fwd = spot * np.exp((0.005 - 0.018) * clean["time_to_expiry"])
    k = np.log(clean["strike"] / fwd)
    dte = clean["time_to_expiry"] * 365
    atm = clean[(k.abs() < 0.03) & (dte >= 20) & (dte <= 45)].merge(
        v[["expiry", "strike", "option_type", "vendor_iv"]],
        on=["expiry", "strike", "option_type"], how="left")

    gaps, solved = [], 0
    for row in atm.itertuples():
        res = implied_volatility_detailed(
            price=float(row.mid), spot=spot, strike=float(row.strike),
            time_to_expiry=float(row.time_to_expiry), rate=0.005,
            option_type=row.option_type, dividend_yield=0.018)
        if res.ok:
            solved += 1
            if pd.notna(row.vendor_iv):
                gaps.append(abs(res.vol - row.vendor_iv))
    assert solved > 0.8 * len(atm)                       # we price nearly all ATM quotes
    assert np.median(gaps) < 0.05                        # within 5 vol points of the vendor
