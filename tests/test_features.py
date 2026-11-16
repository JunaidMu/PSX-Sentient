"""Unit tests for Phase 2: Feature Engineering & Triple-Barrier 3-Class Labeling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (
    FEATURE_COLUMNS,
    INPUT_CSV_PATH,
    OUTPUT_PARQUET_PATH,
    compute_features_and_labels,
)


@pytest.fixture(scope="module")
def raw_df() -> pd.DataFrame:
    """Load the Phase 1 clean dataset once per test module."""
    assert INPUT_CSV_PATH.exists(), f"Missing input dataset: {INPUT_CSV_PATH}"
    return pd.read_csv(INPUT_CSV_PATH, parse_dates=["date"])


@pytest.fixture(scope="module")
def feature_df() -> pd.DataFrame:
    """Load the generated Phase 2 feature store parquet."""
    assert OUTPUT_PARQUET_PATH.exists(), f"Missing feature store: {OUTPUT_PARQUET_PATH}"
    return pd.read_parquet(OUTPUT_PARQUET_PATH)


def test_feature_store_schema_and_completeness(feature_df: pd.DataFrame) -> None:
    """Verify all required features and target columns exist with zero NaNs or Infs."""
    expected_cols = set(FEATURE_COLUMNS + ["symbol", "date", "next_ret_1d", "B_t", "target_class"])
    missing_cols = expected_cols - set(feature_df.columns)
    assert not missing_cols, f"Missing columns in feature store: {missing_cols}"

    # Verify 30 KSE-100 symbols are present
    assert feature_df["symbol"].nunique() == 30

    # Zero missing or infinite values in any feature or target column
    check_cols = FEATURE_COLUMNS + ["next_ret_1d", "B_t", "target_class"]
    na_counts = feature_df[check_cols].isna().sum()
    assert (na_counts == 0).all(), f"Found NaNs in feature store:\n{na_counts[na_counts > 0]}"

    numeric_vals = feature_df[check_cols].to_numpy(dtype="float64")
    assert np.isfinite(numeric_vals).all(), "Found non-finite (inf/-inf) values in feature store"

    # Strict chronological ordering per symbol
    for _, grp in feature_df.groupby("symbol", sort=False):
        assert grp["date"].is_monotonic_increasing, "Dates are not strictly sorted within symbol"


def test_ex_dividend_intraday_return_adjustment(raw_df: pd.DataFrame) -> None:
    """Verify ret_1d uses intraday return np.log(close/open) when is_ex_dividend_gap == 1."""
    sample_symbols = ["KAPCO", "HUBC", "AKBL"]
    sub = raw_df[raw_df["symbol"].isin(sample_symbols)].copy()
    out = compute_features_and_labels(sub, drop_Incomplete=False)

    ex_div_rows = out[out["is_ex_dividend_gap"] == 1]
    assert len(ex_div_rows) > 0, "Expected at least one ex-dividend row in sample symbols"

    expected_intraday = np.log(ex_div_rows["close"] / ex_div_rows["open"])
    np.testing.assert_allclose(
        ex_div_rows["ret_1d"].to_numpy(),
        expected_intraday.to_numpy(),
        rtol=1e-12,
        atol=1e-12,
    )

    # Normal rows (excluding first row of each symbol) must match np.log(close_t / close_{t-1})
    out["prev_close"] = out.groupby("symbol")["close"].shift(1)
    normal_rows = out[(out["is_ex_dividend_gap"] == 0) & out["prev_close"].notna()]
    expected_normal = np.log(normal_rows["close"] / normal_rows["prev_close"])
    np.testing.assert_allclose(
        normal_rows["ret_1d"].to_numpy(),
        expected_normal.to_numpy(),
        rtol=1e-12,
        atol=1e-12,
    )


def test_zero_lookahead_leakage_in_features(raw_df: pd.DataFrame) -> None:
    """Verify mutating future prices/volumes/macro after cutoff T does not alter features at t <= T."""
    sub = raw_df[raw_df["symbol"].isin(["LUCK", "ENGRO", "HBL", "OGDC", "MARI"])].copy()
    baseline = compute_features_and_labels(sub, drop_Incomplete=False)

    cutoff_date = pd.Timestamp("2022-06-30")
    mutated_input = sub.copy()
    future_mask = mutated_input["date"] > cutoff_date

    # Drastically corrupt all future market and macro values after cutoff_date
    for col in ["open", "high", "low", "close", "volume", "brent_close", "usdpkr_close", "dxy_close"]:
        mutated_input.loc[future_mask, col] = mutated_input.loc[future_mask, col] * 5 + 100

    mutated = compute_features_and_labels(mutated_input, drop_Incomplete=False)

    past_base = baseline[baseline["date"] <= cutoff_date]
    past_mut = mutated[mutated["date"] <= cutoff_date]

    for col in FEATURE_COLUMNS + ["B_t"]:
        np.testing.assert_allclose(
            past_base[col].to_numpy(),
            past_mut[col].to_numpy(),
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
            err_msg=f"Look-ahead leakage detected in column '{col}'!",
        )

    # Also verify target_class at t < cutoff_date is strictly unchanged
    strictly_before_cutoff_base = baseline[baseline["date"] < cutoff_date]
    strictly_before_cutoff_mut = mutated[mutated["date"] < cutoff_date]
    np.testing.assert_allclose(
        strictly_before_cutoff_base["target_class"].to_numpy(),
        strictly_before_cutoff_mut["target_class"].to_numpy(),
        equal_nan=True,
        err_msg="Look-ahead leakage beyond t+1 detected in target_class!",
    )


def test_zero_cross_symbol_leakage(raw_df: pd.DataFrame) -> None:
    """Verify computing features across multiple symbols is identical to computing for one symbol alone."""
    sub_multi = raw_df[raw_df["symbol"].isin(["AKBL", "ATRL", "BAFL"])].copy()
    sub_single = raw_df[raw_df["symbol"] == "ATRL"].copy()

    res_multi = compute_features_and_labels(sub_multi, drop_Incomplete=True)
    res_Multi_atrl = res_multi[res_multi["symbol"] == "ATRL"].reset_index(drop=True)
    res_single_atrl = compute_features_and_labels(sub_single, drop_Incomplete=True).reset_index(drop=True)

    assert len(res_Multi_atrl) == len(res_single_atrl)
    for col in FEATURE_COLUMNS + ["next_ret_1d", "B_t", "target_class"]:
        np.testing.assert_allclose(
            res_Multi_atrl[col].to_numpy(),
            res_single_atrl[col].to_numpy(),
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"Cross-symbol leakage detected in column '{col}'!",
        )


def test_triple_barrier_labeling_and_class_balance(feature_df: pd.DataFrame) -> None:
    """Verify exact Triple-Barrier labeling logic and balanced 3-class distribution."""
    # 1. Verify dynamic barrier B_t = np.maximum(0.0075, 0.60 * vol_20d)
    expected_barrier = np.maximum(0.0075, 0.60 * feature_df["vol_20d"])
    np.testing.assert_allclose(
        feature_df["B_t"].to_numpy(),
        expected_barrier.to_numpy(),
        rtol=1e-12,
        atol=1e-12,
    )
    assert (feature_df["B_t"] >= 0.0075).all()

    # 2. Verify exact 3-class assignment rules
    expected_target = np.where(
        feature_df["next_ret_1d"] > feature_df["B_t"],
        2,
        np.where(feature_df["next_ret_1d"] < -feature_df["B_t"], 0, 1),
    )
    np.testing.assert_array_equal(feature_df["target_class"].to_numpy(), expected_target)

    # 3. Verify class balance: all 3 classes {0, 1, 2} must be well-represented (no class collapse)
    assert set(feature_df["target_class"].unique()) == {0, 1, 2}
    class_props = feature_df["target_class"].value_counts(normalize=True)
    for cls_id in (0, 2):
        prop = float(class_props[cls_id])
        assert 0.15 <= prop <= 0.35, (
            f"Active class {cls_id} proportion {prop:.4f} is outside expected range [0.15, 0.35]"
        )
    neutral_prop = float(class_props[1])
    assert 0.40 <= neutral_prop <= 0.65, (
        f"Neutral class 1 proportion {neutral_prop:.4f} is outside expected range [0.40, 0.65]"
    )


def test_indicator_and_seasonality_bounds(feature_df: pd.DataFrame) -> None:
    """Verify indicator bounds and calendar seasonality flags."""
    assert ((feature_df["rsi_14"] >= 0.0) & (feature_df["rsi_14"] <= 100.0)).all()
    assert (feature_df["vol_20d"] >= 0.0).all()
    assert (feature_df["atr_14_pct"] >= 0.0).all()
    assert (feature_df["bb_width_20"] >= 0.0).all()
    assert (feature_df["close_to_sma20"] > 0.0).all()
    assert (feature_df["close_to_sma50"] > 0.0).all()

    months = pd.to_datetime(feature_df["date"]).dt.month
    np.testing.assert_array_equal(
        feature_df["is_earnings_month"].to_numpy(),
        months.isin([1, 4, 7, 10]).astype("int64").to_numpy(),
    )
    np.testing.assert_array_equal(
        feature_df["is_winter_gas"].to_numpy(),
        months.isin([11, 12, 1, 2]).astype("int64").to_numpy(),
    )
    np.testing.assert_array_equal(
        feature_df["is_monsoon"].to_numpy(),
        months.isin([7, 8]).astype("int64").to_numpy(),
    )
