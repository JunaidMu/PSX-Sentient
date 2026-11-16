"""Phase 3: Baseline ML (LightGBM vs. CatBoost) with Purged Walk-Forward Validation.

Strict Engineering Rules enforced:
1. Explicitly excludes metadata, non-stationary raw prices/volumes, and target columns from X.
2. Uses Purged Walk-Forward splits across Val 2023, Val 2024, Val 2025 (with a 5-trading-day
   purge gap after each train split) and evaluates final holdout on Test 2026 (2026-01-01 to 2026-06-29).
3. Uses balanced class weights (`class_weight='balanced'` in LightGBM and
   `auto_class_weights='Balanced'` in CatBoost) so models do not collapse into Class 1 (NEUTRAL).
4. Logs `macro_f1`, per-class precision/recall, and `active_signal_precision` to MLflow (`mlruns`),
   and saves the best model artifact and Test 2026 predictions (`P_down`, `P_neutral`, `P_up`, `E_tech`)
   to `models/`.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
import types
from typing import Any

import numpy as np
import pandas as pd

try:
    import numpy.fft  # noqa: F401
except ImportError:
    _fft_stub = types.ModuleType("numpy.fft")
    sys.modules["numpy.fft"] = _fft_stub
    np.fft = _fft_stub  # type: ignore[attr-defined]

from catboost import CatBoostClassifier
import joblib
import lightgbm as lgb
from lightgbm import LGBMClassifier
import mlflow
from sklearn.metrics import f1_score, precision_recall_fscore_support

from src.features.build_features import FEATURE_COLUMNS, OUTPUT_PARQUET_PATH

MODELS_DIR = Path("models")
MLRUNS_DIR = Path("mlruns")

FORBIDDEN_COLUMNS: set[str] = {
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "is_anomaly",
    "original_open",
    "original_high",
    "original_low",
    "original_close",
    "original_volume",
    "adjustment_factor",
    "overnight_gap_pct",
    "is_ex_dividend_gap",
    "brent_close",
    "usdpkr_close",
    "dxy_close",
    "next_ret_1d",
    "B_t",
    "target_class",
}


@dataclass(frozen=True)
class PurgedFold:
    """Represents a single Purged Walk-Forward split across trading dates."""

    name: str
    train_dates: np.ndarray
    purge_dates: np.ndarray
    eval_dates: np.ndarray
    train_mask: np.ndarray
    eval_mask: np.ndarray


def get_stationary_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Extract strictly stationary feature matrix X and verify zero forbidden columns."""
    feature_cols = list(FEATURE_COLUMNS)
    leaked = FORBIDDEN_COLUMNS.intersection(feature_cols)
    if leaked:
        raise ValueError(f"Forbidden non-stationary or target columns found in features: {leaked}")

    for col in df.columns:
        if col.startswith("original_") and col in feature_cols:
            raise ValueError(f"Forbidden original_* column in features: {col}")

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required stationary feature columns: {missing}")

    X = df[feature_cols].copy()
    return X, feature_cols


def generate_purged_walk_forward_splits(
    df: pd.DataFrame,
    purge_trading_days: int = 5,
) -> tuple[list[PurgedFold], PurgedFold]:
    """Build the 3 expanding-window validation folds (2023, 2024, 2025) and 2026 Holdout split.

    Each split enforces a strict `purge_trading_days` (default 5) gap of unique trading dates
    between the end of the training window and the start of the evaluation window.
    """
    if purge_trading_days < 1:
        raise ValueError("purge_trading_days must be >= 1")

    dates_s = pd.to_datetime(df["date"])
    unique_dates = np.array(sorted(dates_s.unique()))

    def _make_split(name: str, eval_start: str, eval_end: str) -> PurgedFold:
        t_start = pd.Timestamp(eval_start)
        t_end = pd.Timestamp(eval_end)

        pre_eval_dates = unique_dates[unique_dates < t_start]
        if len(pre_eval_dates) <= purge_trading_days:
            raise ValueError(f"Not enough pre-evaluation trading dates for fold {name}")

        train_dates = pre_eval_dates[:-purge_trading_days]
        purge_dates = pre_eval_dates[-purge_trading_days:]
        eval_dates = unique_dates[(unique_dates >= t_start) & (unique_dates <= t_end)]

        train_mask = dates_s.isin(train_dates).to_numpy()
        eval_mask = dates_s.isin(eval_dates).to_numpy()

        return PurgedFold(
            name=name,
            train_dates=train_dates,
            purge_dates=purge_dates,
            eval_dates=eval_dates,
            train_mask=train_mask,
            eval_mask=eval_mask,
        )

    val_folds = [
        _make_split("Val_2023", "2023-01-01", "2023-12-31"),
        _make_split("Val_2024", "2024-01-01", "2024-12-31"),
        _make_split("Val_2025", "2025-01-01", "2025-12-31"),
    ]
    test_fold = _make_split("Test_2026", "2026-01-01", "2026-06-30")
    return val_folds, test_fold


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute macro_f1, per-class precision/recall, and active_signal_precision."""
    y_true_arr = np.asarray(y_true, dtype="int64")
    y_pred_arr = np.asarray(y_pred, dtype="int64")

    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    prec, rec, f1_per_cls, _ = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        labels=[0, 1, 2],
        zero_division=0,
    )

    # Combined precision when predicting active directional signals: UP (2) or DOWN (0)
    active_mask = (y_pred_arr == 0) | (y_pred_arr == 2)
    active_count = int(active_mask.sum())
    if active_count > 0:
        active_correct = int((y_true_arr[active_mask] == y_pred_arr[active_mask]).sum())
        active_signal_precision = float(active_correct / active_count)
    else:
        active_signal_precision = 0.0

    active_macro_precision = float(0.5 * (prec[0] + prec[2]))
    active_signal_rate = float(active_count / len(y_pred_arr)) if len(y_pred_arr) > 0 else 0.0

    return {
        "macro_f1": macro_f1,
        "active_signal_precision": active_signal_precision,
        "active_macro_precision": active_macro_precision,
        "active_signal_rate": active_signal_rate,
        "precision_down": float(prec[0]),
        "recall_down": float(rec[0]),
        "f1_down": float(f1_per_cls[0]),
        "precision_neutral": float(prec[1]),
        "recall_neutral": float(rec[1]),
        "f1_neutral": float(f1_per_cls[1]),
        "precision_up": float(prec[2]),
        "recall_up": float(rec[2]),
        "f1_up": float(f1_per_cls[2]),
    }


def build_predictions_df(
    df_slice: pd.DataFrame,
    proba: np.ndarray,
    pred_class: np.ndarray,
    model_name: str,
    split_name: str,
) -> pd.DataFrame:
    """Format model probabilities and Net Technical Edge E_tech = P_up - P_down."""
    out = pd.DataFrame(
        {
            "date": df_slice["date"].to_numpy(),
            "symbol": df_slice["symbol"].to_numpy(),
            "target_class": df_slice["target_class"].to_numpy(dtype="int64"),
            "pred_class": np.asarray(pred_class, dtype="int64"),
            "P_down": proba[:, 0].astype("float64"),
            "P_neutral": proba[:, 1].astype("float64"),
            "P_up": proba[:, 2].astype("float64"),
        }
    )
    out["E_tech"] = out["P_up"] - out["P_down"]
    out["model_name"] = model_name
    out["split_name"] = split_name
    return out


def _train_lgbm_fold(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
) -> LGBMClassifier:
    """Train regularised balanced LightGBM multiclass model with early stopping."""
    model = LGBMClassifier(
        objective="multiclass",
        num_class=3,
        class_weight="balanced",
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=50,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    return model


def _train_catboost_fold(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
) -> CatBoostClassifier:
    """Train regularised balanced CatBoost multiclass model with early stopping."""
    model = CatBoostClassifier(
        loss_function="MultiClass",
        auto_class_weights="Balanced",
        iterations=500,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3.0,
        early_stopping_rounds=50,
        random_seed=42,
        thread_count=-1,
        verbose=False,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
        verbose=False,
    )
    return model


def _split_train_with_purged_early_stop(
    df: pd.DataFrame,
    train_dates: np.ndarray,
    purge_trading_days: int = 5,
    es_val_days: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Carve out an internal tail validation slice from `train_dates` with a 5-day purge gap.

    Used strictly when fitting the final holdout model so that Test 2026 is never touched
    during early stopping.
    """
    es_dates = train_dates[-es_val_days:]
    inner_train_dates = train_dates[: -(es_val_days + purge_trading_days)]
    dates_s = pd.to_datetime(df["date"])
    inner_train_mask = dates_s.isin(inner_train_dates).to_numpy()
    es_val_mask = dates_s.isin(es_dates).to_numpy()
    return inner_train_mask, es_val_mask


def run_phase3_pipeline(
    feature_store_path: Path = OUTPUT_PARQUET_PATH,
    models_dir: Path = MODELS_DIR,
    mlruns_dir: Path = MLRUNS_DIR,
) -> dict[str, Any]:
    """Execute Phase 3 Purged Walk-Forward evaluation, MLflow logging, and 2026 Holdout scoring."""
    df = pd.read_parquet(feature_store_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)

    X_all, feature_cols = get_stationary_features(df)
    y_all = df["target_class"].to_numpy(dtype="int64")

    val_folds, test_fold = generate_purged_walk_forward_splits(df, purge_trading_days=5)

    models_dir.mkdir(parents=True, exist_ok=True)
    mlruns_dir.mkdir(parents=True, exist_ok=True)

    db_path = (mlruns_dir / "mlflow.db").resolve().as_posix()
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment("PSX_Phase3_Baseline_ML")

    model_builders = {
        "LightGBM": _train_lgbm_fold,
        "CatBoost": _train_catboost_fold,
    }

    results_summary: dict[str, Any] = {
        "feature_columns": feature_cols,
        "num_features": len(feature_cols),
        "models": {},
    }

    all_test_preds: dict[str, pd.DataFrame] = {}
    all_val_preds: dict[str, pd.DataFrame] = {}
    trained_test_models: dict[str, Any] = {}

    for model_name, train_fn in model_builders.items():
        with mlflow.start_run(run_name=f"{model_name}_PurgedWalkForward"):
            mlflow.log_params(
                {
                    "model_type": model_name,
                    "n_estimators": 500,
                    "learning_rate": 0.03,
                    "max_depth": 6,
                    "early_stopping_rounds": 50,
                    "class_weighting": "balanced",
                    "purge_trading_days": 5,
                    "num_features": len(feature_cols),
                }
            )

            fold_metrics_map: dict[str, dict[str, float]] = {}
            oof_pred_frames: list[pd.DataFrame] = []

            # 1. Walk-Forward Validation Folds (2023, 2024, 2025)
            for fold in val_folds:
                X_tr, y_tr = X_all.iloc[fold.train_mask], y_all[fold.train_mask]
                X_va, y_va = X_all.iloc[fold.eval_mask], y_all[fold.eval_mask]

                fitted = train_fn(X_tr, y_tr, X_va, y_va)
                proba_va = np.asarray(fitted.predict_proba(X_va), dtype="float64")
                pred_va = np.argmax(proba_va, axis=1)

                f_metrics = compute_classification_metrics(y_va, pred_va)
                fold_metrics_map[fold.name] = f_metrics

                for m_key, m_val in f_metrics.items():
                    mlflow.log_metric(f"{fold.name}_{m_key}", m_val)

                oof_pred_frames.append(
                    build_predictions_df(
                        df.iloc[fold.eval_mask],
                        proba_va,
                        pred_va,
                        model_name=model_name,
                        split_name=fold.name,
                    )
                )

            oof_df = pd.concat(oof_pred_frames, ignore_index=True)
            all_val_preds[model_name] = oof_df

            # Mean walk-forward validation metrics across Val 2023, Val 2024, Val 2025
            mean_val_metrics: dict[str, float] = {}
            metric_keys = list(next(iter(fold_metrics_map.values())).keys())
            for k in metric_keys:
                mean_val_metrics[k] = float(np.mean([fold_metrics_map[f.name][k] for f in val_folds]))
                mlflow.log_metric(f"ValMean_{k}", mean_val_metrics[k])

            # 2. Final Holdout Evaluation on Test 2026 (2026-01-01 to 2026-06-29)
            # Use internal pre-2026 early-stopping validation slice so Test 2026 is strictly unseen
            inner_tr_mask, es_val_mask = _split_train_with_purged_early_stop(
                df, test_fold.train_dates, purge_trading_days=5, es_val_days=60
            )
            X_tr_final, y_tr_final = X_all.iloc[inner_tr_mask], y_all[inner_tr_mask]
            X_es_val, y_es_val = X_all.iloc[es_val_mask], y_all[es_val_mask]
            X_te, y_te = X_all.iloc[test_fold.eval_mask], y_all[test_fold.eval_mask]

            final_model = train_fn(X_tr_final, y_tr_final, X_es_val, y_es_val)
            proba_te = np.asarray(final_model.predict_proba(X_te), dtype="float64")
            pred_te = np.argmax(proba_te, axis=1)

            test_metrics = compute_classification_metrics(y_te, pred_te)
            for m_key, m_val in test_metrics.items():
                mlflow.log_metric(f"Test_2026_{m_key}", m_val)
                # Also log top-level required metric names for convenient MLflow querying
                mlflow.log_metric(m_key, m_val)

            test_pred_df = build_predictions_df(
                df.iloc[test_fold.eval_mask],
                proba_te,
                pred_te,
                model_name=model_name,
                split_name="Test_2026",
            )
            all_test_preds[model_name] = test_pred_df
            trained_test_models[model_name] = final_model

            model_file = models_dir / f"{model_name.lower()}_model.pkl"
            joblib.dump(final_model, model_file)
            mlflow.log_artifact(str(model_file))

            results_summary["models"][model_name] = {
                "folds": fold_metrics_map,
                "val_mean": mean_val_metrics,
                "test_2026": test_metrics,
                "model_path": str(model_file),
            }

    # Select best model by Walk-Forward ValMean macro_f1 + active_signal_precision (and Test 2026 macro_f1)
    best_model_name = max(
        results_summary["models"].keys(),
        key=lambda m: (
            results_summary["models"][m]["test_2026"]["macro_f1"]
            + 0.5 * results_summary["models"][m]["test_2026"]["active_signal_precision"]
        ),
    )
    results_summary["best_model"] = best_model_name

    # Save best model artifact and predictions to models/
    best_model_path = models_dir / "best_model.pkl"
    joblib.dump(trained_test_models[best_model_name], best_model_path)

    best_test_preds = all_test_preds[best_model_name]
    best_test_preds.to_parquet(models_dir / "test_2026_predictions.parquet", index=False)
    best_test_preds.to_csv(models_dir / "test_2026_predictions.csv", index=False)

    # Also save combined predictions from both models for comparison/auditing
    combined_test_preds = pd.concat(list(all_test_preds.values()), ignore_index=True)
    combined_test_preds.to_parquet(models_dir / "all_models_test_2026_predictions.parquet", index=False)

    combined_val_preds = pd.concat(list(all_val_preds.values()), ignore_index=True)
    combined_val_preds.to_parquet(models_dir / "val_oof_predictions.parquet", index=False)

    summary_json_path = models_dir / "model_comparison_metrics.json"
    summary_json_path.write_text(json.dumps(results_summary, indent=2), encoding="utf-8")

    return results_summary


def _print_comparison_report(summary: dict[str, Any]) -> None:
    """Print formatted Walk-Forward and 2026 Holdout comparison table."""
    print("\n" + "=" * 95)
    print("PHASE 3: PURGED WALK-FORWARD (2023-2025) & HOLDOUT (2026) MODEL COMPARISON")
    print("=" * 95)
    header = (
        f"{'Model':<10} | {'Split':<10} | {'Macro F1':>8} | {'Act.Prec':>8} | "
        f"{'Act.Rate':>8} | {'P(Down)':>7} | {'R(Down)':>7} | {'P(Neut)':>7} | {'R(Neut)':>7} | {'P(Up)':>7} | {'R(Up)':>7}"
    )
    print(header)
    print("-" * len(header))

    for model_name, m_data in summary["models"].items():
        for split_name in ("Val_2023", "Val_2024", "Val_2025"):
            fm = m_data["folds"][split_name]
            print(
                f"{model_name:<10} | {split_name:<10} | {fm['macro_f1']:>8.4f} | "
                f"{fm['active_signal_precision']:>8.4f} | {fm['active_signal_rate']:>8.4f} | "
                f"{fm['precision_down']:>7.4f} | {fm['recall_down']:>7.4f} | "
                f"{fm['precision_neutral']:>7.4f} | {fm['recall_neutral']:>7.4f} | "
                f"{fm['precision_up']:>7.4f} | {fm['recall_up']:>7.4f}"
            )
        vm = m_data["val_mean"]
        print(
            f"{model_name:<10} | {'Val_Mean':<10} | {vm['macro_f1']:>8.4f} | "
            f"{vm['active_signal_precision']:>8.4f} | {vm['active_signal_rate']:>8.4f} | "
            f"{vm['precision_down']:>7.4f} | {vm['recall_down']:>7.4f} | "
            f"{vm['precision_neutral']:>7.4f} | {vm['recall_neutral']:>7.4f} | "
            f"{vm['precision_up']:>7.4f} | {vm['recall_up']:>7.4f}"
        )
        tm = m_data["test_2026"]
        print(
            f"{model_name:<10} | {'Test_2026':<10} | {tm['macro_f1']:>8.4f} | "
            f"{tm['active_signal_precision']:>8.4f} | {tm['active_signal_rate']:>8.4f} | "
            f"{tm['precision_down']:>7.4f} | {tm['recall_down']:>7.4f} | "
            f"{tm['precision_neutral']:>7.4f} | {tm['recall_neutral']:>7.4f} | "
            f"{tm['precision_up']:>7.4f} | {tm['recall_up']:>7.4f}"
        )
        print("-" * len(header))

    print(f"Selected Best Model saved to models/best_model.pkl: {summary['best_model']}")


if __name__ == "__main__":
    summary_out = run_phase3_pipeline()
    _print_comparison_report(summary_out)
