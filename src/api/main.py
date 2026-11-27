"""PSX-Sentient serving API (Deliverable 2).

Serves the MLflow-registry `champion` with confidence scores, SHAP explanations and the late-fusion
conflict-veto overlay; exposes Prometheus metrics and live input/prediction drift checks.

Endpoints
- GET  /                 service info
- GET  /health           liveness + served model + data coverage
- GET  /model            served model info;  POST /model/reload  re-read the registry `champion`
- POST /predict          predict for {symbol, date} from the feature store, or for raw `features`
- GET  /signals/latest   predictions for all 30 stocks on the latest trading day
- GET  /drift            latest drift report;  POST /drift/check  drift of recently served inputs
- GET  /metrics          Prometheus metrics (latency, errors, predictions, drift)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

from src.api.serving import (
    CHAMPION,
    FEATURES,
    log_prediction,
    predict_features,
    premarket_context,
    premarket_snapshot,
    read_prediction_log,
)
from src.features.build_features import FEATURE_COLUMNS
from src.nlp.entity_mapper import KSE30_SYMBOLS

ENABLE_NLP = os.environ.get("ENABLE_NLP", "0") == "1"
DRIFT_REPORT = Path("reports/drift_report.json")
RETRAIN_DECISION = Path("reports/retrain_decision.json")
MIN_ROWS_FOR_DRIFT = 150

app = FastAPI(
    title="PSX-Sentient API",
    description="Next-day DOWN/NEUTRAL/UP signals for 30 KSE-100 stocks with SHAP explanations, "
    "a news-sentiment conflict veto, Prometheus metrics and drift monitoring.",
    version="2.0.0",
)
Instrumentator(excluded_handlers=["/metrics"]).instrument(app).expose(app, include_in_schema=False)

PREDICTIONS = Counter("psx_predictions_total", "Predictions served", ["model_signal", "final_signal"])
VETOES = Counter("psx_veto_total", "Predictions neutralised by the conflict veto gate")
CONFIDENCE = Histogram("psx_prediction_confidence", "Top-class probability of served predictions",
                       buckets=(0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 1.0))
MODEL_VERSION = Gauge("psx_model_version", "Registry version of the served champion")
DRIFT_SHARE = Gauge("psx_drift_share", "Share of market features drifting vs. training data")
FEATURE_PSI = Gauge("psx_feature_psi", "PSI of each feature vs. training data", ["feature"])
PRED_PSI = Gauge("psx_prediction_class_psi", "PSI of the predicted-class mix vs. training")
RETRAIN_FLAG = Gauge("psx_retrain_recommended", "1 when the drift rule recommends retraining")


class PredictRequest(BaseModel):
    symbol: str | None = Field(default=None, description="One of the 30 KSE-100 symbols, e.g. OGDC")
    date: str | None = Field(default=None, description="Trading date YYYY-MM-DD (default: latest)")
    features: dict[str, float] | None = Field(default=None, description="Raw 27-feature vector (live input)")
    sentiment: float | None = Field(default=None, ge=-1.0, le=1.0, description="News sentiment S_t in [-1, 1]")
    headline: str | None = Field(default=None, description="Headline scored by FinBERT (needs ENABLE_NLP=1)")


def _publish_model_version() -> None:
    try:
        MODEL_VERSION.set(float(CHAMPION.info.get("version", "nan")))
    except ValueError:
        MODEL_VERSION.set(0)


def _publish_drift(report: dict[str, Any]) -> None:
    DRIFT_SHARE.set(report["drift_share"])
    RETRAIN_FLAG.set(1 if report["retrain_recommended"] else 0)
    if report.get("prediction_class_psi") is not None:
        PRED_PSI.set(report["prediction_class_psi"])
    for name, f in report["features"].items():
        FEATURE_PSI.labels(feature=name).set(f["psi"])


@app.on_event("startup")
def _startup() -> None:
    CHAMPION.load()
    _publish_model_version()
    if DRIFT_REPORT.exists():
        _publish_drift(json.loads(DRIFT_REPORT.read_text(encoding="utf-8")))


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "PSX-Sentient API",
        "version": app.version,
        "status": "ok",
        "endpoints": {"docs": "/docs", "health": "/health", "model": "/model", "predict": "/predict",
                      "latest_signals": "/signals/latest", "premarket": "/premarket",
                      "drift": "/drift", "metrics": "/metrics"},
    }


@app.get("/health")
def health() -> dict[str, Any]:
    CHAMPION.ensure_loaded()
    df = FEATURES.df
    return {
        "status": "ok",
        "model": CHAMPION.info,
        "symbols": int(df["symbol"].nunique()),
        "data_range": [str(df["date"].min().date()), str(df["date"].max().date())],
        "nlp_enabled": ENABLE_NLP,
    }


@app.get("/model")
def model_info() -> dict[str, Any]:
    CHAMPION.ensure_loaded()
    out: dict[str, Any] = {"served": CHAMPION.info}
    if RETRAIN_DECISION.exists():
        out["last_retraining"] = json.loads(RETRAIN_DECISION.read_text(encoding="utf-8"))
    return out


@app.post("/model/reload")
def reload_model() -> dict[str, Any]:
    info = CHAMPION.load()
    _publish_model_version()
    return {"reloaded": True, "served": info}


def _sentiment_for(req: PredictRequest, symbol: str | None, is_latest: bool) -> tuple[float | None, str]:
    """Sentiment for the veto: explicit value > scored headline > today's pre-market news snapshot."""
    if req.sentiment is not None:
        return req.sentiment, "request"
    if req.headline:
        if not ENABLE_NLP:
            raise HTTPException(400, "Headline scoring needs ENABLE_NLP=1; pass `sentiment` instead.")
        from src.nlp.finbert_engine import FinBERTSentimentEngine

        score = FinBERTSentimentEngine(device="cpu").score_texts([req.headline])["sentiment_score"].iloc[0]
        return float(score), "headline (FinBERT)"
    if symbol and is_latest:
        ctx = premarket_context(symbol)
        s_t = (ctx or {}).get("sentiment", {}).get("S_t")
        if s_t is not None:
            return float(s_t), f"pre-market news ({ctx['trading_day']})"
    return None, "none"


@app.post("/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    if req.features is not None:
        missing = [c for c in FEATURE_COLUMNS if c not in req.features]
        if missing:
            raise HTTPException(422, f"Missing features: {missing}")
        features, meta = req.features, {"symbol": req.symbol, "date": req.date, "input": "raw_features"}
    elif req.symbol:
        sym = req.symbol.strip().upper()
        if sym not in KSE30_SYMBOLS:
            raise HTTPException(404, f"{sym} is not one of the 30 tracked KSE-100 stocks")
        try:
            row = FEATURES.row(sym, req.date)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        features = {c: float(row[c]) for c in FEATURE_COLUMNS}
        meta = {"symbol": sym, "date": str(row["date"].date()), "input": "feature_store",
                "close": float(row["close"]) if "close" in row else None}
    else:
        raise HTTPException(422, "Provide either `symbol` (+ optional `date`) or `features`.")

    sym = meta.get("symbol")
    is_latest = req.features is None and req.date is None
    sentiment, source = _sentiment_for(req, sym, is_latest)
    result = {**meta, **predict_features(features, sentiment=sentiment), "sentiment_source": source}
    if sym and is_latest:
        result["premarket_context"] = premarket_context(sym)
    PREDICTIONS.labels(model_signal=result["model_signal"], final_signal=result["final_signal"]).inc()
    CONFIDENCE.observe(result["confidence"])
    if result["veto_triggered"]:
        VETOES.inc()
    log_prediction(result, features)
    return result


@app.get("/features/{symbol}")
def features_for(symbol: str, date: str | None = None) -> dict[str, Any]:
    """The stored 27-feature vector for a stock/date (starting point for live what-if inputs)."""
    try:
        row = FEATURES.row(symbol.strip().upper(), date)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"symbol": symbol.upper(), "date": str(row["date"].date()),
            "features": {c: float(row[c]) for c in FEATURE_COLUMNS}}


@app.get("/signals/latest")
def latest_signals() -> dict[str, Any]:
    date = FEATURES.latest_date()
    rows = []
    for sym in KSE30_SYMBOLS:
        try:
            row = FEATURES.row(sym, str(date.date()))
        except KeyError:
            continue
        ctx = premarket_context(sym) or {}
        s_t = (ctx.get("sentiment") or {}).get("S_t")
        res = predict_features({c: float(row[c]) for c in FEATURE_COLUMNS}, sentiment=s_t, top_k=3)
        rows.append({"symbol": sym, "close": float(row["close"]), **res,
                     "whale_score": (ctx.get("whale") or {}).get("W_ndm"),
                     "futures_oi_pct_free_float": (ctx.get("futures_open_interest") or {}).get("pct_free_float"),
                     "latest_headline": next(iter((ctx.get("sentiment") or {}).get("latest_headlines", [])), None),
                     "announcements": ctx.get("announcements", [])})
    counts = {s: sum(r["final_signal"] == s for r in rows) for s in ("UP", "NEUTRAL", "DOWN")}
    snap = premarket_snapshot()
    return {"date": str(date.date()), "model": CHAMPION.info, "summary": counts,
            "premarket": {k: snap[k] for k in ("trading_day", "generated_at_pkt", "sources", "news", "counts")} if snap else None,
            "signals": rows}


@app.get("/premarket")
def premarket() -> dict[str, Any]:
    """Latest pre-market snapshot: news sentiment, off-market whales, futures OI, announcements."""
    snap = premarket_snapshot()
    if not snap:
        raise HTTPException(404, "No pre-market snapshot yet; run python -m src.data.premarket")
    return snap


@app.get("/drift")
def drift_report() -> dict[str, Any]:
    if not DRIFT_REPORT.exists():
        raise HTTPException(404, "No drift report yet; POST /drift/check or run src.monitoring.drift")
    return json.loads(DRIFT_REPORT.read_text(encoding="utf-8"))


@app.post("/drift/check")
def drift_check(last_n: int = 3000) -> dict[str, Any]:
    """Compare the most recently *served* inputs with the champion's training distribution."""
    from src.monitoring.drift import run_drift_check

    served = read_prediction_log(last_n)
    if len(served) < MIN_ROWS_FOR_DRIFT:
        raise HTTPException(409, f"Need at least {MIN_ROWS_FOR_DRIFT} served predictions, have {len(served)}")
    served["date"] = served["date"].fillna(served["ts"].str[:10])
    CHAMPION.ensure_loaded()
    report = run_drift_check(served, model=CHAMPION.model)
    _publish_drift(report)
    return {k: v for k, v in report.items() if k != "features"}
