"""FastAPI Service for PSX-Sentient (Phase 6).

Exposes:
- GET /health
- GET /signals/latest
- GET /ablation
- POST /predict/{symbol}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data.live_updater import (
    LATEST_SIGNALS_JSON_PATH,
    compute_whale_block_metric,
    generate_latest_signals_snapshot,
)
from src.fusion.late_fusion import ABLATION_METRICS_PATH, apply_late_fusion_row
from src.models.train_baseline import MODELS_DIR
from src.nlp.entity_mapper import KSE30_SYMBOLS

MODEL_COMPARISON_PATH = MODELS_DIR / "model_comparison_metrics.json"

app = FastAPI(
    title="PSX-Sentient API",
    description=(
        "Regime-Adaptive Late-Fusion Stock Movement & Sentiment Engine for 30 KSE-100 Equities "
        "(CatBoost + CPU FinBERT + Conflict Veto Gate + Off-Market Whale Block Detector)."
    ),
    version="1.0.0",
)


class PredictRequest(BaseModel):
    """Optional override payload for real-time what-if or live inference on `/predict/{symbol}`."""

    P_down: float | None = Field(default=None, ge=0.0, le=1.0)
    P_neutral: float | None = Field(default=None, ge=0.0, le=1.0)
    P_up: float | None = Field(default=None, ge=0.0, le=1.0)
    S_t: float | None = Field(default=None, ge=-1.0, le=1.0)
    headline: str | None = Field(default=None)
    ndm_volume: float | None = Field(default=None, ge=0.0)
    ndm_deal_rate: float | None = Field(default=None, ge=0.0)
    ready_close: float | None = Field(default=None, gt=0.0)
    vol_sma_20: float | None = Field(default=None, gt=0.0)


def _load_latest_signals_payload() -> dict[str, Any]:
    """Load `models/latest_signals.json`, generating it on the fly if missing."""
    if not LATEST_SIGNALS_JSON_PATH.exists():
        generate_latest_signals_snapshot()
    return json.loads(LATEST_SIGNALS_JSON_PATH.read_text(encoding="utf-8"))


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """Return service health and frozen model metadata."""
    payload = _load_latest_signals_payload()
    return {
        "status": "ok",
        "service": "psx-sentient-api",
        "model_name": payload.get("model_name", "CatBoost"),
        "model_cutoff_date": payload.get("model_cutoff_date", "2026-06-30"),
        "tracked_symbols_count": len(KSE30_SYMBOLS),
        "latest_signals_ready": LATEST_SIGNALS_JSON_PATH.exists(),
    }


@app.get("/signals/latest")
async def get_latest_signals() -> dict[str, Any]:
    """Return all 30 KSE-100 stocks with probabilities, E_tech, S_t, W_ndm, C_t, veto_triggered, and signal."""
    payload = _load_latest_signals_payload()
    signals = payload.get("signals", [])
    bullish_count = sum(1 for s in signals if s.get("signal") == "UP")
    bearish_count = sum(1 for s in signals if s.get("signal") == "DOWN")
    neutral_count = sum(1 for s in signals if s.get("signal") == "NEUTRAL")
    veto_count = sum(1 for s in signals if s.get("veto_triggered") is True)
    whale_count = sum(1 for s in signals if s.get("is_whale_block") is True)

    return {
        "model_cutoff_date": payload.get("model_cutoff_date", "2026-06-30"),
        "model_name": payload.get("model_name", "CatBoost"),
        "total_symbols": len(signals),
        "summary": {
            "active_signals": bullish_count + bearish_count,
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "neutral_count": neutral_count,
            "veto_count": veto_count,
            "whale_block_count": whale_count,
        },
        "signals": signals,
    }


@app.get("/ablation")
async def get_ablation_and_model_metrics() -> dict[str, Any]:
    """Return Phase 3 model comparison metrics and Phase 5 Late-Fusion ablation metrics."""
    if not MODEL_COMPARISON_PATH.exists():
        raise HTTPException(status_code=404, detail="model_comparison_metrics.json not found")
    if not ABLATION_METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="ablation_metrics.json not found")

    model_comp = json.loads(MODEL_COMPARISON_PATH.read_text(encoding="utf-8"))
    ablation = json.loads(ABLATION_METRICS_PATH.read_text(encoding="utf-8"))
    return {
        "model_comparison": model_comp,
        "ablation_study": ablation,
    }


@app.post("/predict/{symbol}")
async def predict_symbol(
    symbol: str,
    request: PredictRequest | None = Body(default=None),
) -> dict[str, Any]:
    """Run Late-Fusion + Conflict Veto + Off-Market Whale inference for one of the 30 KSE-100 symbols."""
    sym_upper = symbol.strip().upper()
    if sym_upper not in KSE30_SYMBOLS:
        raise HTTPException(
            status_code=404,
            detail=f"Symbol '{sym_upper}' is not in the 30 tracked KSE-100 stocks.",
        )

    payload = _load_latest_signals_payload()
    base_map = {item["symbol"]: dict(item) for item in payload.get("signals", [])}
    if sym_upper not in base_map:
        raise HTTPException(status_code=404, detail=f"No snapshot record found for '{sym_upper}'.")

    rec = base_map[sym_upper]
    req = request or PredictRequest()

    p_down = float(req.P_down if req.P_down is not None else rec["P_down"])
    p_neutral = float(req.P_neutral if req.P_neutral is not None else rec["P_neutral"])
    p_up = float(req.P_up if req.P_up is not None else rec["P_up"])

    # Normalize probabilities if custom probabilities were passed
    prob_sum = p_down + p_neutral + p_up
    if prob_sum > 0:
        p_down, p_neutral, p_up = p_down / prob_sum, p_neutral / prob_sum, p_up / prob_sum

    # Determine sentiment S_t: score live headline via CPU FinBERT if provided without explicit S_t
    if req.S_t is not None:
        s_t = float(req.S_t)
    elif req.headline:
        from src.nlp.finbert_engine import FinBERTSentimentEngine

        engine = FinBERTSentimentEngine(device="cpu")
        scored = engine.score_texts([req.headline])
        s_t = float(scored["sentiment_score"].iloc[0])
        rec["latest_headline"] = req.headline
    else:
        s_t = float(rec["S_t"])

    # Recompute Off-Market Whale metric if custom NDM parameters are supplied
    if req.ndm_volume is not None and req.ndm_deal_rate is not None:
        r_close = float(req.ready_close if req.ready_close is not None else rec["close"])
        v_sma20 = float(req.vol_sma_20 if req.vol_sma_20 is not None else max(rec["ndm_volume"] * 10.0, 100000.0))
        whale_res = compute_whale_block_metric(
            ndm_volume=float(req.ndm_volume),
            ndm_deal_rate=float(req.ndm_deal_rate),
            ready_close=r_close,
            vol_sma_20=v_sma20,
        )
        rec.update(whale_res)

    fused = apply_late_fusion_row(
        P_down=p_down,
        P_neutral=p_neutral,
        P_up=p_up,
        S_t=s_t,
    )

    rec.update(
        {
            "P_down": round(float(fused["P_down"]), 4),
            "P_neutral": round(float(fused["P_neutral"]), 4),
            "P_up": round(float(fused["P_up"]), 4),
            "E_tech": round(float(fused["E_tech"]), 4),
            "S_t": round(float(fused["S_t"]), 4),
            "C_t": round(float(fused["C_t"]), 4),
            "veto_triggered": bool(fused["veto_triggered"]),
            "signal_class": int(fused["fused_class"]),
            "signal": str(fused["fused_signal"]),
        }
    )
    return rec
