"""Unit tests for Phase 6: Live Updater, Off-Market Whale Parser, and FastAPI Service."""

from __future__ import annotations

from fastapi.testclient import TestClient
import numpy as np

from src.api.main import app
from src.data.live_updater import compute_whale_block_metric, parse_psx_off_market_csv

client = TestClient(app)


def test_off_market_whale_block_detector() -> None:
    """Verify PSX Off-Market (NDM) Whale Block detector threshold (>5% of 20d avg volume) and spread."""
    # 1. Block > 5% of 20d avg volume at a +3.5% premium -> positive W_ndm and is_whale_block = True
    bull_whale = compute_whale_block_metric(
        ndm_volume=150_000,
        ndm_deal_rate=103.50,
        ready_close=100.00,
        vol_sma_20=1_000_000,
    )
    assert bull_whale["is_whale_block"] is True
    assert bull_whale["W_ndm"] > 0.20

    # 2. Block > 5% of 20d avg volume at a -4.0% discount -> negative W_ndm and is_whale_block = True
    bear_whale = compute_whale_block_metric(
        ndm_volume=120_000,
        ndm_deal_rate=96.00,
        ready_close=100.00,
        vol_sma_20=1_000_000,
    )
    assert bear_whale["is_whale_block"] is True
    assert bear_whale["W_ndm"] < -0.20

    # 3. Small NDM deal <= 5% of 20d avg volume -> not an institutional whale block (W_ndm == 0.0)
    small_deal = compute_whale_block_metric(
        ndm_volume=20_000,
        ndm_deal_rate=105.00,
        ready_close=100.00,
        vol_sma_20=1_000_000,
    )
    assert small_deal["is_whale_block"] is False
    assert small_deal["W_ndm"] == 0.0

    # 4. CSV parser test
    csv_text = "symbol,ndm_volume,ndm_deal_rate\nOGDC,200000,345.0\nATRL,150000,820.0\n"
    parsed = parse_psx_off_market_csv(
        csv_text,
        ready_close_map={"OGDC": 335.0, "ATRL": 850.0},
        vol_sma20_map={"OGDC": 1_500_000, "ATRL": 1_000_000},
    )
    assert len(parsed) == 2
    ogdc_row = parsed[parsed["symbol"] == "OGDC"].iloc[0]
    atrl_row = parsed[parsed["symbol"] == "ATRL"].iloc[0]
    assert bool(ogdc_row["is_whale_block"]) is True and float(ogdc_row["W_ndm"]) > 0.0
    assert bool(atrl_row["is_whale_block"]) is True and float(atrl_row["W_ndm"]) < 0.0


def test_health_endpoint() -> None:
    """Verify GET /health returns status ok and 30 tracked symbols."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["tracked_symbols_count"] == 30
    assert data["model_cutoff_date"] == "2026-06-30"


def test_latest_signals_endpoint() -> None:
    """Verify GET /signals/latest returns all 30 KSE-100 stocks with complete Late-Fusion fields."""
    resp = client.get("/signals/latest")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_symbols"] == 30
    assert len(data["signals"]) == 30

    required_fields = {
        "symbol",
        "date",
        "close",
        "P_down",
        "P_neutral",
        "P_up",
        "E_tech",
        "S_t",
        "W_ndm",
        "C_t",
        "veto_triggered",
        "signal",
    }
    for item in data["signals"]:
        assert required_fields.issubset(item.keys())
        assert item["signal"] in ("UP", "DOWN", "NEUTRAL")
        assert np.isclose(item["P_down"] + item["P_neutral"] + item["P_up"], 1.0, atol=1e-3)


def test_ablation_endpoint() -> None:
    """Verify GET /ablation returns Phase 3 model comparison and Phase 5 ablation metrics."""
    resp = client.get("/ablation")
    assert resp.status_code == 200
    data = resp.json()
    assert "model_comparison" in data
    assert "ablation_study" in data
    strategies = data["ablation_study"]["strategies"]
    assert "c_full_late_fusion_conflict_veto" in strategies
    assert strategies["c_full_late_fusion_conflict_veto"]["active_signal_precision"] > 0.60


def test_predict_symbol_endpoint_and_conflict_veto() -> None:
    """Verify POST /predict/{symbol} handles default inference, conflict veto override, and 404."""
    # 1. Default snapshot prediction for OGDC
    resp_default = client.post("/predict/OGDC")
    assert resp_default.status_code == 200
    ogdc = resp_default.json()
    assert ogdc["symbol"] == "OGDC"
    assert ogdc["signal"] in ("UP", "DOWN", "NEUTRAL")

    # 2. Override with strong bullish probabilities (E_tech = 0.50 > 0.20) + negative shock S_t = -0.75 < -0.50
    # Must trigger Conflict Veto Gate and force signal to NEUTRAL
    resp_veto = client.post(
        "/predict/LUCK",
        json={
            "P_down": 0.15,
            "P_neutral": 0.20,
            "P_up": 0.65,
            "S_t": -0.75,
            "ndm_volume": 250000,
            "ndm_deal_rate": 480.0,
            "ready_close": 460.0,
            "vol_sma_20": 2000000,
        },
    )
    assert resp_veto.status_code == 200
    veto_data = resp_veto.json()
    assert veto_data["veto_triggered"] is True
    assert veto_data["signal"] == "NEUTRAL"
    assert veto_data["is_whale_block"] is True
    assert veto_data["W_ndm"] > 0.0

    # 3. Unknown symbol must return 404
    resp_404 = client.post("/predict/INVALID_TICKER")
    assert resp_404.status_code == 404
