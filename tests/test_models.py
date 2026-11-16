"""Unit tests for Phase 3: Baseline ML (Purged Walk-Forward, LightGBM, CatBoost, MLflow)."""

from __future__ import annotations

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from src.features.build_features import FEATURE_COLUMNS, OUTPUT_PARQUET_PATH
from src.models.train_baseline import (
    FORBIDDEN_COLUMNS,
    MODELS_DIR,
    compute_classification_metrics,
    generate_purged_walk_forward_splits,
    get_stationary_features,
)


@pytest.fixture(scope="module")
def feature_df() -> pd.DataFrame:
    """Load the Phase 2 feature store parquet."""
    assert OUTPUT_PARQUET_PATH.exists(), f"Missing feature store: {OUTPUT_PARQUET_PATH}"
    df = pd.read_parquet(OUTPUT_PARQUET_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "symbol"]).reset_index(drop=True)


def test_strict_feature_exclusion(feature_df: pd.DataFrame) -> None:
    """Verify metadata, raw non-stationary prices, and target columns are excluded from X."""
    X, cols = get_stationary_features(feature_df)

    assert cols == FEATURE_COLUMNS
    assert len(cols) == 27

    # Explicitly check none of the forbidden columns exist in X
    for bad_col in FORBIDDEN_COLUMNS:
        assert bad_col not in X.columns, f"Forbidden column '{bad_col}' leaked into X!"

    for col in X.columns:
        assert not col.startswith("original_"), f"Raw column '{col}' leaked into X!"


def test_purged_walk_forward_splits_integrity(feature_df: pd.DataFrame) -> None:
    """Verify 3 expanding-window validation folds (2023, 2024, 2025) + Test 2026 with 5-day purge gap."""
    val_folds, test_fold = generate_purged_walk_forward_splits(feature_df, purge_trading_days=5)

    assert [f.name for f in val_folds] == ["Val_2023", "Val_2024", "Val_2025"]
    assert test_fold.name == "Test_2026"

    all_splits = val_folds + [test_fold]
    prev_train_len = 0

    unique_dates = list(sorted(pd.to_datetime(feature_df["date"]).unique()))

    for split in all_splits:
        # 1. Expanding window check
        assert len(split.train_dates) > prev_train_len
        prev_train_len = len(split.train_dates)

        # 2. Exact 5-trading-day purge gap check
        assert len(split.purge_dates) == 5

        # 3. Strict chronological separation: max(train) < min(purge) <= max(purge) < min(eval)
        assert split.train_dates.max() < split.purge_dates.min()
        assert split.purge_dates.max() < split.eval_dates.min()

        # 4. Verify the 5 purge dates are the exact 5 trading days immediately preceding eval_dates[0]
        first_eval_idx = unique_dates.index(split.eval_dates[0])
        expected_purge = np.array(unique_dates[first_eval_idx - 5 : first_eval_idx])
        np.testing.assert_array_equal(split.purge_dates, expected_purge)

        # 5. Zero row overlap between train_mask and eval_mask
        assert not np.any(split.train_mask & split.eval_mask)

    # Verify Test 2026 date bounds
    assert pd.Timestamp(test_fold.eval_dates.min()) >= pd.Timestamp("2026-01-01")
    assert pd.Timestamp(test_fold.eval_dates.max()) <= pd.Timestamp("2026-06-30")


def test_active_signal_precision_metric_calculation() -> None:
    """Verify active_signal_precision and macro_f1 computation on synthetic predictions."""
    y_true = np.array([0, 0, 1, 1, 2, 2], dtype="int64")
    y_pred = np.array([0, 1, 1, 2, 2, 0], dtype="int64")
    # Active predictions (0 or 2) are at indices 0 (pred 0, true 0 -> hit),
    # 3 (pred 2, true 1 -> miss), 4 (pred 2, true 2 -> hit), 5 (pred 0, true 2 -> miss)
    # Total active predictions = 4, hits = 2 -> active_signal_precision = 0.5
    metrics = compute_classification_metrics(y_true, y_pred)
    assert np.isclose(metrics["active_signal_precision"], 0.5)
    assert np.isclose(metrics["active_signal_rate"], 4.0 / 6.0)
    assert 0.0 <= metrics["macro_f1"] <= 1.0


def test_saved_models_and_predictions_anti_collapse() -> None:
    """Verify trained model artifacts, comparison metrics, and non-collapsed Test 2026 predictions."""
    summary_path = MODELS_DIR / "model_comparison_metrics.json"
    best_model_path = MODELS_DIR / "best_model.pkl"
    lgbm_path = MODELS_DIR / "lightgbm_model.pkl"
    cb_path = MODELS_DIR / "catboost_model.pkl"
    test_preds_path = MODELS_DIR / "test_2026_predictions.parquet"
    all_test_preds_path = MODELS_DIR / "all_models_test_2026_predictions.parquet"

    for p in (summary_path, best_model_path, lgbm_path, cb_path, test_preds_path, all_test_preds_path):
        assert p.exists(), f"Missing expected Phase 3 artifact: {p}"

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "LightGBM" in summary["models"]
    assert "CatBoost" in summary["models"]
    assert summary["best_model"] in ("LightGBM", "CatBoost")

    # Verify both models did not collapse into predicting Neutral (Class 1) all the time
    all_preds = pd.read_parquet(all_test_preds_path)
    for model_name in ("LightGBM", "CatBoost"):
        m_df = all_preds[all_preds["model_name"] == model_name]
        assert len(m_df) > 0

        # Must predict all 3 classes {0, 1, 2}
        pred_classes = set(m_df["pred_class"].unique())
        assert pred_classes == {0, 1, 2}, (
            f"Model {model_name} collapsed! Predicted classes: {pred_classes}"
        )

        # Active signal rate (UP or DOWN) must be >= 25% due to balanced class weights
        active_rate = float(m_df["pred_class"].isin([0, 2]).mean())
        assert active_rate >= 0.25, (
            f"Model {model_name} active signal rate {active_rate:.3f} is too low (< 0.25)"
        )

        # Probabilities P_down, P_neutral, P_up must be in [0, 1] and sum to 1.0
        probs = m_df[["P_down", "P_neutral", "P_up"]].to_numpy()
        assert ((probs >= 0.0) & (probs <= 1.0)).all()
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(len(m_df)), rtol=1e-5, atol=1e-5)

        # Net Technical Edge E_tech = P_up - P_down
        expected_edge = m_df["P_up"] - m_df["P_down"]
        np.testing.assert_allclose(m_df["E_tech"].to_numpy(), expected_edge.to_numpy(), rtol=1e-12, atol=1e-12)
        assert ((m_df["E_tech"] >= -1.0) & (m_df["E_tech"] <= 1.0)).all()

    # Verify best_model.pkl loads and can run inference
    best_model = joblib.load(best_model_path)
    assert hasattr(best_model, "predict_proba")
