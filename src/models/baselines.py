"""Rule-based and classical baselines, logged to MLflow before the primary models.

Baselines (same purged walk-forward folds and Test_2026 holdout as the primary models):
1. `AlwaysNeutral`       - rule: always predict the majority class NEUTRAL (1).
2. `MomentumRule`        - rule: UP if the last 5-day return exceeds the 20-day volatility,
                           DOWN if it is below minus the 20-day volatility, else NEUTRAL.
3. `LogisticRegression`  - classical: standardised features + balanced multinomial logistic regression.

Writes `models/baseline_metrics.json` (a DVC metrics file).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features.build_features import OUTPUT_PARQUET_PATH
from src.models.tracking import log_dataset, setup_mlflow
from src.models.train_baseline import (
    MODELS_DIR,
    compute_classification_metrics,
    generate_purged_walk_forward_splits,
    get_stationary_features,
)

BASELINE_METRICS_PATH = MODELS_DIR / "baseline_metrics.json"


def predict_always_neutral(df: pd.DataFrame) -> np.ndarray:
    return np.ones(len(df), dtype="int64")


def predict_momentum_rule(df: pd.DataFrame) -> np.ndarray:
    ret_5d = df["ret_5d"].to_numpy(dtype="float64")
    vol = df["vol_20d"].to_numpy(dtype="float64")
    return np.where(ret_5d > vol, 2, np.where(ret_5d < -vol, 0, 1)).astype("int64")


def _fit_logreg(X: pd.DataFrame, y: np.ndarray):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42),
    )
    model.fit(X, y)
    return model


def _evaluate(
    name: str,
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: np.ndarray,
    predict_fn: Callable[[pd.DataFrame, np.ndarray, np.ndarray], np.ndarray],
) -> dict[str, Any]:
    """Evaluate a baseline on the 3 walk-forward folds and the Test_2026 holdout."""
    val_folds, test_fold = generate_purged_walk_forward_splits(df, purge_trading_days=5)
    folds: dict[str, dict[str, float]] = {}
    for fold in val_folds + [test_fold]:
        y_pred = predict_fn(df, fold.train_mask, fold.eval_mask)
        folds[fold.name] = compute_classification_metrics(y[fold.eval_mask], y_pred)

    val_names = [f.name for f in val_folds]
    val_mean = {k: float(np.mean([folds[n][k] for n in val_names])) for k in folds[val_names[0]]}
    return {"folds": {n: folds[n] for n in val_names}, "val_mean": val_mean, "test_2026": folds["Test_2026"]}


def run_baselines(
    feature_store_path: Path = OUTPUT_PARQUET_PATH,
    output_path: Path = BASELINE_METRICS_PATH,
) -> dict[str, Any]:
    df = pd.read_parquet(feature_store_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)
    X, feature_cols = get_stationary_features(df)
    y = df["target_class"].to_numpy(dtype="int64")

    def rule(fn):
        return lambda d, tr, ev: fn(d[ev])

    def logreg(d, tr, ev):
        return _fit_logreg(X[tr], y[tr]).predict(X[ev]).astype("int64")

    baselines = {
        "AlwaysNeutral": ("rule", rule(predict_always_neutral), {"rule": "predict NEUTRAL (1) always"}),
        "MomentumRule": ("rule", rule(predict_momentum_rule), {"rule": "UP if ret_5d > vol_20d; DOWN if ret_5d < -vol_20d"}),
        "LogisticRegression": (
            "classical",
            logreg,
            {"class_weight": "balanced", "max_iter": 2000, "scaler": "StandardScaler", "num_features": len(feature_cols)},
        ),
    }

    setup_mlflow()
    summary: dict[str, Any] = {"baselines": {}}
    for name, (family, fn, params) in baselines.items():
        with mlflow.start_run(run_name=f"Baseline_{name}"):
            mlflow.set_tags({"model_family": "baseline", "baseline_type": family, "model_type": name})
            mlflow.log_params({"model_type": name, "purge_trading_days": 5, **params})
            log_dataset(df, feature_store_path, context="baseline")
            res = _evaluate(name, df, X, y, fn)
            for fold_name, m in res["folds"].items():
                for k, v in m.items():
                    mlflow.log_metric(f"{fold_name}_{k}", v)
            for k, v in res["val_mean"].items():
                mlflow.log_metric(f"ValMean_{k}", v)
            for k, v in res["test_2026"].items():
                mlflow.log_metric(f"Test_2026_{k}", v)
            summary["baselines"][name] = res

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    out = run_baselines()
    print(f"{'Baseline':<20} | {'ValMean F1':>10} | {'Val ActPrec':>11} | {'Test F1':>8} | {'Test ActPrec':>12}")
    for name, res in out["baselines"].items():
        vm, tm = res["val_mean"], res["test_2026"]
        print(
            f"{name:<20} | {vm['macro_f1']:>10.4f} | {vm['active_signal_precision']:>11.4f} | "
            f"{tm['macro_f1']:>8.4f} | {tm['active_signal_precision']:>12.4f}"
        )
    print(f"Saved: {BASELINE_METRICS_PATH}")
