"""Tests for the D2 serving API (synthetic model + feature store, so they run on a clean CI runner)."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from catboost import CatBoostClassifier
from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest

from src.api import serving
from src.data.live_updater import compute_whale_block_metric
from src.features.build_features import FEATURE_COLUMNS
from src.nlp.entity_mapper import KSE30_SYMBOLS


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2026-10-01", periods=40)
    rows = [{"symbol": s, "date": d, "close": 100.0, **{c: float(rng.normal()) for c in FEATURE_COLUMNS}}
            for s in KSE30_SYMBOLS for d in dates]
    df = pd.DataFrame(rows)
    df["target_class"] = rng.integers(0, 3, len(df))

    model = CatBoostClassifier(loss_function="MultiClass", iterations=20, depth=3, verbose=False, random_seed=0)
    model.fit(df[FEATURE_COLUMNS], df["target_class"])

    serving.FEATURES._df = df.sort_values(["symbol", "date"])
    serving.CHAMPION.model = model
    serving.CHAMPION.info = {"name": "test-model", "version": "7", "alias": "champion", "model_type": "CatBoost"}
    serving.CHAMPION.load = lambda: serving.CHAMPION.info  # never touch a real registry in tests
    serving.PREDICTION_LOG = tmp_path_factory.mktemp("logs") / "predictions.jsonl"

    from src.api.main import app

    with TestClient(app) as c:
        yield c


def test_health_and_model(client) -> None:
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["symbols"] == 30 and h["model"]["version"] == "7"


def test_predict_returns_probabilities_confidence_and_shap(client) -> None:
    r = client.post("/predict", json={"symbol": "ogdc"})
    assert r.status_code == 200
    res = r.json()
    probs = res["probabilities"]
    assert set(probs) == {"DOWN", "NEUTRAL", "UP"}
    assert abs(sum(probs.values()) - 1.0) < 1e-3
    assert res["confidence"] == max(probs.values())
    top = res["explanation"]["top_features"]
    assert len(top) == 8 and all(t["feature"] in FEATURE_COLUMNS for t in top)
    assert res["final_signal"] in {"DOWN", "NEUTRAL", "UP"}


def test_predict_raw_feature_vector_and_validation(client) -> None:
    feats = {c: 0.0 for c in FEATURE_COLUMNS}
    assert client.post("/predict", json={"features": feats}).status_code == 200
    feats.pop("rsi_14")
    assert client.post("/predict", json={"features": feats}).status_code == 422
    assert client.post("/predict", json={"symbol": "ENGRO"}).status_code == 404
    assert client.post("/predict", json={}).status_code == 422


def test_conflict_veto_with_strong_opposing_sentiment(client) -> None:
    # Find a stock the model calls UP strongly enough for the veto to apply, then add very bad news
    for sym in KSE30_SYMBOLS:
        base = client.post("/predict", json={"symbol": sym}).json()
        if base["E_tech"] > 0.20:
            vetoed = client.post("/predict", json={"symbol": sym, "sentiment": -0.9}).json()
            assert vetoed["veto_triggered"] is True and vetoed["final_signal"] == "NEUTRAL"
            return
    pytest.skip("synthetic model produced no strong UP call")


def test_latest_signals_and_metrics(client) -> None:
    s = client.get("/signals/latest").json()
    assert len(s["signals"]) == 30 and sum(s["summary"].values()) == 30
    metrics = client.get("/metrics").text
    assert "psx_predictions_total" in metrics and "http_request_duration_seconds" in metrics


def test_predictions_are_logged_for_drift(client) -> None:
    client.post("/predict", json={"symbol": "HBL"})
    log = serving.read_prediction_log()
    assert len(log) > 0 and set(FEATURE_COLUMNS) <= set(log.columns)


def test_off_market_whale_block_detector() -> None:
    bull = compute_whale_block_metric(ndm_volume=150_000, ndm_deal_rate=103.5, ready_close=100.0, vol_sma_20=1_000_000)
    assert bull["is_whale_block"] is True and bull["W_ndm"] > 0.2
    small = compute_whale_block_metric(ndm_volume=20_000, ndm_deal_rate=105.0, ready_close=100.0, vol_sma_20=1_000_000)
    assert small["is_whale_block"] is False and small["W_ndm"] == 0.0
