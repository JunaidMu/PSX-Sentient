"""Late-Fusion Meta-Learner (probability stacking) over LightGBM + CatBoost.

Inputs are the models' *out-of-sample* probabilities only:
- Training data for the meta-learner: walk-forward validation predictions for 2023, 2024 and 2025
  (each year predicted by base models that never saw it) -> models/val_oof_predictions.parquet
- Validation: leave-one-year-out across those three years (meta-learner never scores a year it saw)
- Test: Test_2026 predictions of the final base models -> models/all_models_test_2026_predictions.parquet

Meta-learner: balanced multinomial logistic regression on the 6 probabilities [P_down, P_neutral,
P_up] x [LightGBM, CatBoost]. Logged to MLflow as `Stacking_MetaLearner`; metrics in
models/stacking_metrics.json. The Conflict Veto Gate (src/fusion) is applied on top at serving time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.models.tracking import setup_mlflow
from src.models.train_baseline import MODELS_DIR, compute_classification_metrics

OOF_PATH = MODELS_DIR / "val_oof_predictions.parquet"
TEST_PATH = MODELS_DIR / "all_models_test_2026_predictions.parquet"
META_MODEL_PATH = MODELS_DIR / "stacking_meta.pkl"
STACKING_METRICS_PATH = MODELS_DIR / "stacking_metrics.json"
BASE_MODELS = ("LightGBM", "CatBoost")
PROBS = ("P_down", "P_neutral", "P_up")


def to_wide(preds: pd.DataFrame) -> pd.DataFrame:
    """One row per (date, symbol) with each base model's probabilities side by side."""
    wide = None
    for name in BASE_MODELS:
        part = preds[preds["model_name"] == name][["date", "symbol", "split_name", "target_class", *PROBS]]
        part = part.rename(columns={p: f"{name}_{p}" for p in PROBS})
        wide = part if wide is None else wide.merge(
            part.drop(columns=["split_name", "target_class"]), on=["date", "symbol"]
        )
    return wide.sort_values(["date", "symbol"]).reset_index(drop=True)


META_FEATURES = [f"{m}_{p}" for m in BASE_MODELS for p in PROBS]


def _fit_meta(X: pd.DataFrame, y: np.ndarray) -> LogisticRegression:
    return LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42).fit(X, y)


def run_stacking() -> dict[str, Any]:
    oof = to_wide(pd.read_parquet(OOF_PATH))
    test = to_wide(pd.read_parquet(TEST_PATH))

    folds: dict[str, dict[str, float]] = {}
    for year in sorted(oof["split_name"].unique()):
        tr, ev = oof["split_name"] != year, oof["split_name"] == year
        meta = _fit_meta(oof.loc[tr, META_FEATURES], oof.loc[tr, "target_class"].to_numpy())
        folds[year] = compute_classification_metrics(oof.loc[ev, "target_class"].to_numpy(), meta.predict(oof.loc[ev, META_FEATURES]))
    val_mean = {k: float(np.mean([f[k] for f in folds.values()])) for k in next(iter(folds.values()))}

    meta = _fit_meta(oof[META_FEATURES], oof["target_class"].to_numpy())
    test_metrics = compute_classification_metrics(test["target_class"].to_numpy(), meta.predict(test[META_FEATURES]))

    # Reference: the individual base models on the same Test_2026 rows
    base_test = {
        m: compute_classification_metrics(
            test["target_class"].to_numpy(),
            np.argmax(test[[f"{m}_{p}" for p in PROBS]].to_numpy(), axis=1),
        )
        for m in BASE_MODELS
    }

    joblib.dump(meta, META_MODEL_PATH)
    setup_mlflow()
    with mlflow.start_run(run_name="Stacking_MetaLearner"):
        mlflow.set_tags({"model_family": "stacking", "model_type": "LogisticRegression meta-learner"})
        mlflow.log_params({"base_models": ",".join(BASE_MODELS), "meta_features": len(META_FEATURES),
                           "meta_training": "walk-forward OOF 2023-2025", "validation": "leave-one-year-out"})
        for year, m in folds.items():
            for k, v in m.items():
                mlflow.log_metric(f"{year}_{k}", v)
        for k, v in val_mean.items():
            mlflow.log_metric(f"ValMean_{k}", v)
        for k, v in test_metrics.items():
            mlflow.log_metric(f"Test_2026_{k}", v)
        mlflow.log_artifact(str(META_MODEL_PATH))

    summary = {"folds": folds, "val_mean": val_mean, "test_2026": test_metrics, "base_models_test_2026": base_test,
               "meta_coefficients_shape": list(meta.coef_.shape)}
    STACKING_METRICS_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    s = run_stacking()
    print(f"{'Model':<22} | {'Test F1':>8} | {'Test ActPrec':>12}")
    for m, met in s["base_models_test_2026"].items():
        print(f"{m:<22} | {met['macro_f1']:>8.4f} | {met['active_signal_precision']:>12.4f}")
    t = s["test_2026"]
    print(f"{'Stacking meta-learner':<22} | {t['macro_f1']:>8.4f} | {t['active_signal_precision']:>12.4f}")
    print(f"Meta-learner leave-one-year-out val F1: {s['val_mean']['macro_f1']:.4f}")
