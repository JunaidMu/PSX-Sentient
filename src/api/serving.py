"""Model serving core: champion loading, prediction, SHAP explanations, veto overlay, input logging."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.features.build_features import FEATURE_COLUMNS, OUTPUT_PARQUET_PATH
from src.fusion.late_fusion import apply_late_fusion_row
from src.monitoring.batch import COMBINED_FEATURES_PATH, NEW_FEATURES_PATH

SIGNALS = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}
FALLBACK_MODEL_PATH = Path("models/best_model.pkl")
PREDICTION_LOG = Path(os.environ.get("PREDICTION_LOG", "logs/predictions.jsonl"))


class ChampionModel:
    """Holds the currently served model; reloads when the registry's `champion` alias moves."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.model: Any = None
        self.info: dict[str, str] = {}

    def load(self) -> dict[str, str]:
        with self._lock:
            try:
                from src.models.tracking import load_registered_model

                self.model, self.info = load_registered_model("champion")
                self.info["source"] = "mlflow-registry"
            except Exception as exc:  # registry unreachable -> last exported model on disk
                self.model = joblib.load(FALLBACK_MODEL_PATH)
                self.info = {"name": "best_model.pkl", "version": "local", "alias": "fallback",
                             "model_type": type(self.model).__name__, "source": f"fallback ({str(exc)[:80]})"}
            return self.info

    def ensure_loaded(self) -> None:
        if self.model is None:
            self.load()


CHAMPION = ChampionModel()


class FeatureStore:
    """Read-only lookup of engineered features for (symbol, date): history + post-cutoff batch."""

    def __init__(self) -> None:
        self._df: pd.DataFrame | None = None
        self._stamp: tuple | None = None

    def _files_stamp(self) -> tuple:
        return tuple(Path(p).stat().st_mtime if Path(p).exists() else 0
                     for p in (COMBINED_FEATURES_PATH, NEW_FEATURES_PATH, OUTPUT_PARQUET_PATH))

    @property
    def df(self) -> pd.DataFrame:
        stamp = self._files_stamp()
        if self._df is not None and self._stamp is not None and stamp != self._stamp and any(stamp):
            self._df = None  # the daily job wrote new feature files: reload them
        if self._df is None:
            self._stamp = stamp
            frames = []
            for path in (COMBINED_FEATURES_PATH, NEW_FEATURES_PATH, OUTPUT_PARQUET_PATH):
                if Path(path).exists():
                    frames.append(pd.read_parquet(path))
            if not frames:
                raise FileNotFoundError("No feature data found (run `dvc pull` / the pipeline first)")
            df = pd.concat(frames, ignore_index=True)
            df["date"] = pd.to_datetime(df["date"])
            self._df = df.drop_duplicates(["symbol", "date"], keep="first").sort_values(["symbol", "date"])
        return self._df

    def row(self, symbol: str, date: str | None = None) -> pd.Series:
        sub = self.df[self.df["symbol"] == symbol]
        if sub.empty:
            raise KeyError(f"Unknown symbol {symbol}")
        if date is None:
            return sub.iloc[-1]
        try:
            when = pd.Timestamp(date)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid date '{date}', expected YYYY-MM-DD") from exc
        hit = sub[sub["date"] == when]
        if hit.empty:
            raise KeyError(f"No trading data for {symbol} on {date}")
        return hit.iloc[0]

    def latest_date(self) -> pd.Timestamp:
        counts = self.df.groupby("date")["symbol"].nunique()
        return counts[counts == counts.max()].index.max()


FEATURES = FeatureStore()


def shap_values(model: Any, X: pd.DataFrame, class_idx: int) -> dict[str, float]:
    """Per-feature SHAP contributions (log-odds) towards `class_idx`, computed natively by the GBDT."""
    if hasattr(model, "get_feature_importance"):  # CatBoost
        from catboost import Pool

        vals = np.asarray(model.get_feature_importance(Pool(X), type="ShapValues"))  # (n, classes, F+1)
        contrib = vals[0, class_idx, :-1]
    else:  # LightGBM: contributions laid out class by class, each with a bias term
        raw = np.asarray(model.predict(X, pred_contrib=True))[0]
        width = X.shape[1] + 1
        contrib = raw[class_idx * width:(class_idx + 1) * width][:-1]
    return {f: float(v) for f, v in zip(X.columns, contrib)}


def predict_features(features: dict[str, float], sentiment: float | None = None, top_k: int = 8) -> dict[str, Any]:
    """Score one feature vector: probabilities, confidence, SHAP and the conflict-veto overlay."""
    CHAMPION.ensure_loaded()
    X = pd.DataFrame([[float(features[c]) for c in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
    proba = np.asarray(CHAMPION.model.predict_proba(X), dtype="float64")[0]
    model_class = int(np.argmax(proba))
    shap = shap_values(CHAMPION.model, X, model_class)
    top = sorted(shap.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]

    policy = signal_policy()
    fused = apply_late_fusion_row(P_down=proba[0], P_neutral=proba[1], P_up=proba[2],
                                  S_t=0.0 if sentiment is None else sentiment,
                                  threshold=policy["conviction_threshold"], neutral_cap=policy["neutral_cap"])
    return {
        "probabilities": {"DOWN": round(proba[0], 4), "NEUTRAL": round(proba[1], 4), "UP": round(proba[2], 4)},
        "model_signal": SIGNALS[model_class],
        "confidence": round(float(proba.max()), 4),
        "E_tech": round(float(proba[2] - proba[0]), 4),
        "sentiment_S_t": None if sentiment is None else round(float(sentiment), 4),
        "C_t": round(float(fused["C_t"]), 4),
        "veto_triggered": bool(fused["veto_triggered"]),
        "final_signal": str(fused["fused_signal"]),
        "explanation": {
            "explained_class": SIGNALS[model_class],
            "top_features": [{"feature": f, "shap": round(v, 5), "value": round(float(X.iloc[0][f]), 6)} for f, v in top],
        },
        "model": {k: CHAMPION.info.get(k) for k in ("name", "version", "alias", "model_type")},
        "policy": {"conviction_threshold": policy["conviction_threshold"], "neutral_cap": policy["neutral_cap"]},
    }


PREMARKET_SNAPSHOT = Path("reports/premarket_latest.json")
_POLICY_CACHE: dict[str, Any] = {}


def signal_policy() -> dict[str, Any]:
    """Final-signal thresholds tuned on validation (models/signal_policy.json), reloaded when it changes."""
    from src.models.signal_policy import POLICY_PATH, load_policy

    stamp = POLICY_PATH.stat().st_mtime if POLICY_PATH.exists() else 0
    if _POLICY_CACHE.get("stamp") != stamp:
        _POLICY_CACHE.update(stamp=stamp, policy=load_policy())
    return _POLICY_CACHE["policy"]


def premarket_snapshot() -> dict[str, Any] | None:
    """Latest pre-market snapshot (news sentiment, whales, futures OI, announcements) if collected."""
    if not PREMARKET_SNAPSHOT.exists():
        return None
    return json.loads(PREMARKET_SNAPSHOT.read_text(encoding="utf-8"))


def premarket_context(symbol: str) -> dict[str, Any] | None:
    snap = premarket_snapshot()
    if not snap or symbol not in snap.get("stocks", {}):
        return None
    return {"trading_day": snap["trading_day"], **snap["stocks"][symbol]}


def log_prediction(record: dict[str, Any], features: dict[str, float]) -> None:
    """Append served inputs + outputs to a JSONL log used for live input/prediction drift checks."""
    PREDICTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {"ts": datetime.now(timezone.utc).isoformat(), **{c: float(features[c]) for c in FEATURE_COLUMNS},
             "model_signal": record["model_signal"], "final_signal": record["final_signal"],
             "model_version": record["model"]["version"], "symbol": record.get("symbol"), "date": record.get("date")}
    with PREDICTION_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def read_prediction_log(last_n: int | None = None) -> pd.DataFrame:
    if not PREDICTION_LOG.exists():
        return pd.DataFrame()
    lines = PREDICTION_LOG.read_text(encoding="utf-8").splitlines()
    if last_n:
        lines = lines[-last_n:]
    return pd.DataFrame([json.loads(x) for x in lines if x.strip()])
