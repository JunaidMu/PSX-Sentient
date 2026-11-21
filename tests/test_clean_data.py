"""Unit tests for Phase 1.5: USD/PKR bad-tick repair and ex-dividend gap flagging."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.data.clean_data import flag_ex_dividend_gaps, repair_macro_spikes


def _series(values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.bdate_range("2022-07-25", periods=len(values)), name="usdpkr_close")


def test_one_day_spike_is_repaired_with_previous_value() -> None:
    # Real example: 239.9 -> 206.75 -> 239.1 (2022-08-01)
    s = _series([235.25, 239.9, 206.75, 239.1, 238.3])
    fixed, dates = repair_macro_spikes(s, threshold=0.025, max_spike_len=1)
    assert dates == [s.index[2]]
    assert fixed.iloc[2] == 239.9
    # Every other value is untouched
    np.testing.assert_array_equal(fixed.drop(s.index[2]).to_numpy(), s.drop(s.index[2]).to_numpy())


def test_persistent_devaluation_is_kept() -> None:
    # Real example: Jan 2023 devaluation 229.9 -> 250.0 -> 249.15 -> 269.0 (does not revert)
    s = _series([230.04, 229.9, 250.0, 249.15, 269.0, 267.5])
    fixed, dates = repair_macro_spikes(s, threshold=0.025, max_spike_len=1)
    assert dates == []
    np.testing.assert_array_equal(fixed.to_numpy(), s.to_numpy())


def test_repair_never_uses_future_values_as_replacement() -> None:
    s = _series([100.0, 100.0, 90.0, 100.5, 101.0])
    fixed, _ = repair_macro_spikes(s, threshold=0.025, max_spike_len=1)
    # Replacement equals the last good value before the spike, not the next-day value
    assert fixed.iloc[2] == 100.0


def _gap_frame(stock_gap: float, market_gap: float) -> pd.DataFrame:
    """Two days, 5 symbols: on day 2 symbol 'A' gaps by `stock_gap`, others by `market_gap`."""
    rows = []
    for sym in ["A", "B", "C", "D", "E"]:
        g = stock_gap if sym == "A" else market_gap
        rows.append({"symbol": sym, "date": "2024-03-01", "open": 100.0, "close": 100.0, "is_ex_dividend_gap": 0})
        rows.append({"symbol": sym, "date": "2024-03-04", "open": 100.0 * (1 + g), "close": 100.0 * (1 + g), "is_ex_dividend_gap": 0})
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _flag_for(df: pd.DataFrame, sym: str) -> int:
    row = df[(df["symbol"] == sym) & (df["date"] == pd.Timestamp("2024-03-04"))]
    return int(row["is_ex_dividend_gap"].iloc[0])


def test_stock_specific_gap_down_is_flagged() -> None:
    out = flag_ex_dividend_gaps(_gap_frame(stock_gap=-0.05, market_gap=0.0))
    assert _flag_for(out, "A") == 1
    assert _flag_for(out, "B") == 0


def test_market_wide_gap_down_is_not_flagged() -> None:
    # Whole market gaps down 6% together (crash day): nobody is an ex-dividend case
    out = flag_ex_dividend_gaps(_gap_frame(stock_gap=-0.06, market_gap=-0.06))
    assert out["is_ex_dividend_gap"].sum() == 0


def test_small_gap_is_not_flagged_and_original_flags_are_kept() -> None:
    df = _gap_frame(stock_gap=-0.02, market_gap=0.0)
    df.loc[(df["symbol"] == "C") & (df["date"] == pd.Timestamp("2024-03-04")), "is_ex_dividend_gap"] = 1
    out = flag_ex_dividend_gaps(df)
    assert _flag_for(out, "A") == 0
    assert _flag_for(out, "C") == 1
    assert int(out["is_ex_dividend_gap_original"].sum()) == 1
