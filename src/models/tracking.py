"""Shared MLflow setup and dataset-version logging for baselines and the primary models."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
from typing import Any

import mlflow
import pandas as pd

MLRUNS_DIR = Path("mlruns")
EXPERIMENT_NAME = "PSX_Phase3_Baseline_ML"
REGISTERED_MODEL_NAME = "psx-sentient-direction"
RAW_DATA_PATH = Path("psx_historical_data/stocks_with_macro.csv")
CLEAN_DATA_PATH = Path("psx_historical_data/stocks_clean.csv")


def tracking_uri(mlruns_dir: Path = MLRUNS_DIR) -> str:
    """`MLFLOW_TRACKING_URI` if set (e.g. the MLflow server on EC2), else the local SQLite store."""
    env_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if env_uri:
        return env_uri
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{(mlruns_dir / 'mlflow.db').resolve().as_posix()}"


def setup_mlflow(mlruns_dir: Path = MLRUNS_DIR, experiment: str = EXPERIMENT_NAME) -> None:
    """Point MLflow at the tracking store (tracking + model registry) and select the experiment."""
    mlflow.set_tracking_uri(tracking_uri(mlruns_dir))
    mlflow.set_experiment(experiment)


def file_md5(path: Path) -> str:
    """MD5 of a file's bytes; identical to the hash DVC records in dvc.lock for the same file."""
    if not path.exists():
        return "missing"
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def dataset_version_tags(feature_store_path: Path, df: pd.DataFrame) -> dict[str, Any]:
    """Tags describing exactly which data version a run used."""
    dates = pd.to_datetime(df["date"])
    return {
        "data.raw_path": str(RAW_DATA_PATH),
        "data.raw_md5": file_md5(RAW_DATA_PATH),
        "data.clean_md5": file_md5(CLEAN_DATA_PATH),
        "data.features_path": str(feature_store_path),
        "data.features_md5": file_md5(feature_store_path),
        "data.rows": int(len(df)),
        "data.symbols": int(df["symbol"].nunique()),
        "data.date_min": str(dates.min().date()),
        "data.date_max": str(dates.max().date()),
        "git.commit": git_commit(),
    }


def load_registered_model(alias: str = "champion") -> tuple[Any, dict[str, str]]:
    """Load a registered model version by alias as its native object (predict_proba available).

    Returns (model, info) where info has name, version, alias, model_type and run_id.
    """
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(tracking_uri())
    client = MlflowClient()
    mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, alias)
    model_type = (mv.tags or {}).get("model_type", "CatBoost")
    uri = f"models:/{REGISTERED_MODEL_NAME}@{alias}"
    if model_type == "LightGBM":
        from mlflow import lightgbm as mlflow_lightgbm

        model = mlflow_lightgbm.load_model(uri)
    else:
        from mlflow import catboost as mlflow_catboost

        model = mlflow_catboost.load_model(uri)
    info = {
        "name": REGISTERED_MODEL_NAME,
        "version": str(mv.version),
        "alias": alias,
        "model_type": model_type,
        "run_id": mv.run_id or "",
    }
    return model, info


def log_dataset(df: pd.DataFrame, feature_store_path: Path, context: str = "training") -> None:
    """Log dataset version tags plus an MLflow dataset input for the active run."""
    mlflow.set_tags(dataset_version_tags(feature_store_path, df))
    try:
        ds = mlflow.data.from_pandas(
            df[["date", "symbol", "target_class"]].assign(date=lambda d: d["date"].astype(str)),
            source=str(feature_store_path),
            name="psx_feature_store",
            targets="target_class",
        )
        mlflow.log_input(ds, context=context)
    except Exception as exc:  # Dataset logging is informative only; never fail a run over it
        mlflow.set_tag("data.log_input_error", str(exc)[:250])
