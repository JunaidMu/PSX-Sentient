"""Closed-loop retraining: train a challenger on all data up to a few weeks ago, compare it with the
registered champion on the most recent weeks (which neither model has seen), then promote or keep.

Rolling windows (all purged by 5 trading days), recomputed on every run:
- Comparison holdout   : the last COMPARE_DAYS labelled trading days (~6 weeks)
- Challenger training : everything from 2017 up to the purge gap before the holdout
Set CHALLENGER_TRAIN_END / COMPARE_START to pin fixed windows instead (reproducing a past run).

Challenger = CatBoost with hyperparameters tuned by Optuna on an early-stopping slice carved from the
end of its own training window (the holdout is never used for tuning).

Promotion rule: challenger macro-F1 >= champion macro-F1 + MIN_F1_GAIN and challenger active-signal
precision >= champion's - MAX_PRECISION_DROP. On promotion the old champion keeps the alias
`previous_champion` so `--rollback` can restore it instantly.

Usage:
    python -m src.monitoring.retrain              # train challenger, compare, promote or keep
    python -m src.monitoring.retrain --rollback   # point `champion` back at `previous_champion`
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from catboost import CatBoostClassifier
import mlflow
import mlflow.catboost
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
import numpy as np
import optuna
import pandas as pd

from src.features.build_features import FEATURE_COLUMNS
from src.models.tracking import REGISTERED_MODEL_NAME, load_registered_model, log_dataset, setup_mlflow
from src.models.train_baseline import compute_classification_metrics
from src.monitoring.batch import COMBINED_FEATURES_PATH

CHALLENGER_TRAIN_END = os.environ.get("CHALLENGER_TRAIN_END")  # optional fixed windows
COMPARE_START = os.environ.get("COMPARE_START")
COMPARE_DAYS = int(os.environ.get("COMPARE_DAYS", "30"))
PURGE_DAYS = 5
ES_DAYS = 60
OPTUNA_TRIALS = int(os.environ.get("OPTUNA_TRIALS", "8"))
MIN_F1_GAIN = 0.0
MAX_PRECISION_DROP = 0.02
DECISION_PATH = Path("reports/retrain_decision.json")


def _windows(df: pd.DataFrame) -> dict[str, np.ndarray]:
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(df["date"]).unique()))
    if COMPARE_START:
        compare_dates = dates[dates >= pd.Timestamp(COMPARE_START)]
    else:
        compare_dates = dates[-COMPARE_DAYS:]  # rolling: the most recent labelled weeks
    train_dates = dates[dates < compare_dates[0]]
    if CHALLENGER_TRAIN_END:
        train_dates = train_dates[train_dates <= pd.Timestamp(CHALLENGER_TRAIN_END)]
    train_dates = train_dates[:-PURGE_DAYS]  # purge gap before the comparison window
    es_dates = train_dates[-ES_DAYS:]
    inner_dates = train_dates[: -(ES_DAYS + PURGE_DAYS)]
    d = pd.DatetimeIndex(pd.to_datetime(df["date"]))
    return {
        "inner": np.isin(d, inner_dates),
        "es": np.isin(d, es_dates),
        "compare": np.isin(d, compare_dates),
        "train_end": str(pd.Timestamp(train_dates[-1]).date()),
        "compare_range": [str(pd.Timestamp(compare_dates[0]).date()), str(pd.Timestamp(compare_dates[-1]).date())],
        "data_end": str(pd.Timestamp(dates[-1]).date()),
    }


def last_retrain_data_end() -> str | None:
    """Latest labelled date the previous retraining run used (None if it never ran)."""
    if not DECISION_PATH.exists():
        return None
    try:
        return json.loads(DECISION_PATH.read_text(encoding="utf-8")).get("data_end")
    except Exception:
        return None


def _fit_catboost(params: dict[str, Any], X_tr, y_tr, X_es, y_es) -> CatBoostClassifier:
    model = CatBoostClassifier(
        loss_function="MultiClass",
        auto_class_weights="Balanced",
        iterations=500,
        early_stopping_rounds=50,
        random_seed=42,
        thread_count=-1,
        verbose=False,
        **params,
    )
    model.fit(X_tr, y_tr, eval_set=(X_es, y_es), use_best_model=True, verbose=False)
    return model


def tune_challenger(X_tr, y_tr, X_es, y_es, n_trials: int = OPTUNA_TRIALS) -> tuple[dict[str, Any], float]:
    """Optuna search maximising macro-F1 + 0.5 * active precision on the early-stopping slice."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "depth": trial.suggest_int("depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        }
        model = _fit_catboost(params, X_tr, y_tr, X_es, y_es)
        m = compute_classification_metrics(y_es, np.argmax(model.predict_proba(X_es), axis=1))
        return m["macro_f1"] + 0.5 * m["active_signal_precision"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.enqueue_trial({"depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 3.0})  # the champion's settings
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, float(study.best_value)


def _score(model, X, y) -> dict[str, float]:
    return compute_classification_metrics(y, np.argmax(model.predict_proba(X), axis=1))


def run_retraining(features_path: Path = COMBINED_FEATURES_PATH, trigger: str = "manual") -> dict[str, Any]:
    df = pd.read_parquet(features_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)
    X = df[FEATURE_COLUMNS]
    y = df["target_class"].to_numpy(dtype="int64")
    w = _windows(df)

    setup_mlflow()
    champion, champ_info = load_registered_model("champion")
    X_cmp, y_cmp = X[w["compare"]], y[w["compare"]]
    champ_metrics = _score(champion, X_cmp, y_cmp)

    with mlflow.start_run(run_name="Retrain_Challenger_CatBoost") as run:
        mlflow.set_tags({"model_family": "challenger", "model_type": "CatBoost", "trigger": trigger})
        log_dataset(df, features_path, context="retraining")
        best_params, best_tune = tune_challenger(X[w["inner"]], y[w["inner"]], X[w["es"]], y[w["es"]])
        challenger = _fit_catboost(best_params, X[w["inner"]], y[w["inner"]], X[w["es"]], y[w["es"]])
        chall_metrics = _score(challenger, X_cmp, y_cmp)

        mlflow.log_params({**{f"optuna_{k}": v for k, v in best_params.items()},
                           "optuna_trials": OPTUNA_TRIALS, "train_end": w["train_end"],
                           "compare_start": w["compare_range"][0], "compare_end": w["compare_range"][1],
                           "champion_version": champ_info["version"]})
        mlflow.log_metric("optuna_best_objective", best_tune)
        for k, v in chall_metrics.items():
            mlflow.log_metric(f"Compare_challenger_{k}", v)
        for k, v in champ_metrics.items():
            mlflow.log_metric(f"Compare_champion_{k}", v)

        X_ex = X_cmp.head(200).astype("float64")
        info = mlflow.catboost.log_model(challenger, name="model",
                                         signature=infer_signature(X_ex, challenger.predict_proba(X_ex)),
                                         input_example=X_ex.head(5))

        promote = (
            chall_metrics["macro_f1"] >= champ_metrics["macro_f1"] + MIN_F1_GAIN
            and chall_metrics["active_signal_precision"] >= champ_metrics["active_signal_precision"] - MAX_PRECISION_DROP
        )
        client = MlflowClient()
        mv = mlflow.register_model(info.model_uri, REGISTERED_MODEL_NAME)
        for key, val in {"model_type": "CatBoost", "role": "challenger", "trigger": trigger,
                         "compare_macro_f1": f"{chall_metrics['macro_f1']:.4f}",
                         "train_end": w["train_end"], "data_end": w["data_end"]}.items():
            client.set_model_version_tag(REGISTERED_MODEL_NAME, mv.version, key, val)
        client.set_registered_model_alias(REGISTERED_MODEL_NAME, "challenger", mv.version)
        if promote:
            client.set_registered_model_alias(REGISTERED_MODEL_NAME, "previous_champion", champ_info["version"])
            client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", mv.version)
        mlflow.set_tag("decision", "promoted" if promote else "kept_champion")

    decision = {
        "trigger": trigger,
        "decision": "promoted" if promote else "kept_champion",
        "champion_before": champ_info["version"],
        "challenger_version": str(mv.version),
        "champion_after": str(mv.version) if promote else champ_info["version"],
        "compare_window": w["compare_range"],
        "compare_rows": int(w["compare"].sum()),
        "challenger_train_end": w["train_end"],
        "data_end": w["data_end"],
        "optuna_best_params": best_params,
        "champion_metrics": champ_metrics,
        "challenger_metrics": chall_metrics,
        "rule": f"promote if challenger macro_f1 >= champion + {MIN_F1_GAIN} and active precision drop <= {MAX_PRECISION_DROP}",
        "mlflow_run_id": run.info.run_id,
    }
    DECISION_PATH.parent.mkdir(parents=True, exist_ok=True)
    DECISION_PATH.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    return decision


def rollback() -> dict[str, str]:
    """Point `champion` back at `previous_champion` (and remember the demoted version)."""
    setup_mlflow()
    client = MlflowClient()
    prev = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "previous_champion")
    cur = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "champion")
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", prev.version)
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "rolled_back", cur.version)
    return {"champion": str(prev.version), "rolled_back": str(cur.version)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollback", action="store_true")
    ap.add_argument("--trigger", default="manual")
    args = ap.parse_args()
    if args.rollback:
        print(json.dumps(rollback(), indent=2))
    else:
        print(json.dumps(run_retraining(trigger=args.trigger), indent=2))
