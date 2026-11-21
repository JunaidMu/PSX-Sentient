"""Unit tests for the pre-training schema & quality gates (src/data/validate.py)."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.data.validate import validate_feature_frame, validate_raw_frame
from src.features.build_features import FEATURE_COLUMNS, compute_features_and_labels
from src.nlp.entity_mapper import KSE30_SYMBOLS


def _raw_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2017-01-02", periods=1600)
    frames = []
    for sym in KSE30_SYMBOLS:
        close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates))))
        frames.append(
            pd.DataFrame(
                {
                    "date": dates.strftime("%Y-%m-%d"),
                    "symbol": sym,
                    "open": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": 1_000_000,
                    "adjustment_factor": 1.0,
                    "is_ex_dividend_gap": 0,
                    "brent_close": 70.0,
                    "usdpkr_close": 280.0,
                    "dxy_close": 104.0,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _failed(results) -> set[str]:
    return {r.name for r in results if not r.passed}


def test_clean_raw_frame_passes_all_checks() -> None:
    assert _failed(validate_raw_frame(_raw_frame())) == set()


def test_missing_column_is_caught() -> None:
    assert "schema.required_columns" in _failed(validate_raw_frame(_raw_frame().drop(columns=["close"])))


def test_corrupted_values_are_caught() -> None:
    df = _raw_frame()
    df.loc[5, "close"] = np.nan
    df.loc[10, "low"] = -1.0
    df.loc[20, "usdpkr_close"] = 2800.0  # decimal-point corruption
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # duplicated row
    failed = _failed(validate_raw_frame(df))
    assert {
        "quality.no_missing_values",
        "quality.positive_prices",
        "quality.usdpkr_close_in_range",
        "quality.unique_symbol_date",
    } <= failed


def test_missing_symbol_and_unadjusted_split_are_caught() -> None:
    df = _raw_frame()
    df = df[df["symbol"] != "OGDC"].copy()
    mask = (df["symbol"] == "LUCK") & (df["date"] >= "2020-01-01")
    df.loc[mask, ["open", "high", "low", "close"]] *= 5.0  # a 1-for-5 split never adjusted
    failed = _failed(validate_raw_frame(df))
    assert {"quality.expected_symbols", "quality.no_unadjusted_splits"} <= failed


def test_feature_gate_catches_nan_and_bad_labels() -> None:
    feats = compute_features_and_labels(_raw_frame().assign(date=lambda d: pd.to_datetime(d["date"])))
    # Synthetic prices are pure noise, so only check the gate's mechanics
    feats.loc[feats.index[0], FEATURE_COLUMNS[0]] = np.nan
    feats.loc[feats.index[1], "target_class"] = 7
    failed = _failed(validate_feature_frame(feats))
    assert {"quality.no_nan", "quality.labels_valid"} <= failed
