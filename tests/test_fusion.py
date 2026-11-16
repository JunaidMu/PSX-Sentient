"""Unit tests for Phase 5: Late-Fusion Engine, Conflict Veto Gate, and Ablation Study."""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest

from src.fusion.late_fusion import (
    ABLATION_METRICS_PATH,
    FUSED_PREDICTIONS_PATH,
    apply_conviction_tabular_df,
    apply_late_fusion_df,
    apply_late_fusion_row,
)


def test_conflict_veto_gate_exact_boundaries() -> None:
    """Verify exact strict-inequality boundaries of the Conflict Veto Gate."""
    # Case 1: Strong bullish technical edge (E_tech = 0.35 > 0.20) + strongly negative news (S_t = -0.60 < -0.50)
    # Even if C_t = 0.75*0.35 + 0.25*(-0.60) = 0.1125 or even if E_tech = 0.60 (C_t = 0.30 > 0.20),
    # veto_triggered MUST be True and force signal to 1 (NEUTRAL).
    res_bull_veto = apply_late_fusion_row(P_down=0.10, P_neutral=0.20, P_up=0.70, S_t=-0.51)
    assert np.isclose(res_bull_veto["E_tech"], 0.60)
    assert res_bull_veto["C_t"] > 0.20
    assert res_bull_veto["veto_triggered"] is True
    assert res_bull_veto["fused_class"] == 1
    assert res_bull_veto["fused_signal"] == "NEUTRAL"

    # Case 2: Strong bearish technical edge (E_tech = -0.60 < -0.20) + strongly positive news (S_t = 0.51 > 0.50)
    res_bear_veto = apply_late_fusion_row(P_down=0.70, P_neutral=0.20, P_up=0.10, S_t=0.51)
    assert np.isclose(res_bear_veto["E_tech"], -0.60)
    assert res_bear_veto["C_t"] < -0.20
    assert res_bear_veto["veto_triggered"] is True
    assert res_bear_veto["fused_class"] == 1
    assert res_bear_veto["fused_signal"] == "NEUTRAL"

    # Case 3: Exact boundary values (E_tech == 0.20 or S_t == -0.50) must NOT trigger strict inequality veto
    res_boundary_s = apply_late_fusion_row(P_down=0.10, P_neutral=0.20, P_up=0.70, S_t=-0.50)
    assert res_boundary_s["veto_triggered"] is False
    assert res_boundary_s["fused_class"] == 2  # C_t = 0.75*0.60 + 0.25*(-0.50) = 0.325 > 0.20

    res_boundary_e = apply_late_fusion_row(P_down=0.30, P_neutral=0.20, P_up=0.50, S_t=-0.80)
    assert np.isclose(res_boundary_e["E_tech"], 0.20)
    assert res_boundary_e["veto_triggered"] is False


def test_blended_conviction_score_and_neutral_probability_gate() -> None:
    """Verify C_t = 0.75 * E_tech + 0.25 * S_t thresholding and P_neutral < 0.50 requirement."""
    # 1. Bullish signal: E_tech = 0.20, S_t = 0.40 -> C_t = 0.15 + 0.10 = 0.25 > 0.20, P_neutral = 0.30 < 0.50 -> UP (2)
    up_res = apply_late_fusion_row(P_down=0.25, P_neutral=0.30, P_up=0.45, S_t=0.40)
    assert np.isclose(up_res["C_t"], 0.25)
    assert up_res["veto_triggered"] is False
    assert up_res["fused_class"] == 2
    assert up_res["fused_signal"] == "UP"

    # 2. Bearish signal: E_tech = -0.20, S_t = -0.40 -> C_t = -0.25 < -0.20, P_neutral = 0.30 < 0.50 -> DOWN (0)
    down_res = apply_late_fusion_row(P_down=0.45, P_neutral=0.30, P_up=0.25, S_t=-0.40)
    assert np.isclose(down_res["C_t"], -0.25)
    assert down_res["veto_triggered"] is False
    assert down_res["fused_class"] == 0
    assert down_res["fused_signal"] == "DOWN"

    # 3. High P_neutral gate: Even if C_t > 0.20, if P_neutral >= 0.50, must output 1 (NEUTRAL)
    high_p_neutral = apply_late_fusion_row(P_down=0.05, P_neutral=0.50, P_up=0.45, S_t=0.80)
    assert high_p_neutral["C_t"] > 0.20
    assert high_p_neutral["fused_class"] == 1
    assert high_p_neutral["fused_signal"] == "NEUTRAL"

    # 4. Exact C_t == 0.20 boundary: strict '>' means C_t == 0.20 stays 1 (NEUTRAL)
    exact_c_boundary = apply_late_fusion_row(P_down=0.30, P_neutral=0.30, P_up=0.50, S_t=0.20)
    # E_tech = 0.20, S_t = 0.20 -> C_t = 0.75*0.20 + 0.25*0.20 = 0.20
    assert np.isclose(exact_c_boundary["C_t"], 0.20)
    assert exact_c_boundary["fused_class"] == 1


def test_vectorized_late_fusion_matches_row_logic() -> None:
    """Verify apply_late_fusion_df matches row-by-row apply_late_fusion_row across all regimes."""
    df = pd.DataFrame(
        [
            {"P_down": 0.15, "P_neutral": 0.25, "P_up": 0.60, "S_t": 0.50},   # Strong UP
            {"P_down": 0.60, "P_neutral": 0.25, "P_up": 0.15, "S_t": -0.50},  # Strong DOWN
            {"P_down": 0.15, "P_neutral": 0.25, "P_up": 0.60, "S_t": -0.75},  # Bull Veto -> NEUTRAL
            {"P_down": 0.60, "P_neutral": 0.25, "P_up": 0.15, "S_t": 0.75},   # Bear Veto -> NEUTRAL
            {"P_down": 0.30, "P_neutral": 0.40, "P_up": 0.30, "S_t": 0.10},   # Low conviction -> NEUTRAL
            {"P_down": 0.05, "P_neutral": 0.55, "P_up": 0.40, "S_t": 0.90},   # P_neutral >= 0.50 -> NEUTRAL
        ]
    )
    vec_out = apply_late_fusion_df(df)
    row_out = [
        apply_late_fusion_row(r.P_down, r.P_neutral, r.P_up, r.S_t)
        for r in df.itertuples(index=False)
    ]

    np.testing.assert_allclose(vec_out["C_t"].to_numpy(), [r["C_t"] for r in row_out], rtol=1e-12)
    np.testing.assert_array_equal(vec_out["veto_triggered"].to_numpy(), [r["veto_triggered"] for r in row_out])
    np.testing.assert_array_equal(vec_out["fused_class"].to_numpy(), [r["fused_class"] for r in row_out])
    assert list(vec_out["fused_signal"]) == [r["fused_signal"] for r in row_out]


def test_ablation_metrics_and_saved_outputs() -> None:
    """Verify models/ablation_metrics.json and fused Test_2026 predictions exist and are valid."""
    assert ABLATION_METRICS_PATH.exists(), f"Missing ablation summary: {ABLATION_METRICS_PATH}"
    assert FUSED_PREDICTIONS_PATH.exists(), f"Missing fused predictions: {FUSED_PREDICTIONS_PATH}"

    summary = json.loads(ABLATION_METRICS_PATH.read_text(encoding="utf-8"))
    assert summary["split"] == "Test_2026"
    assert summary["total_rows"] > 3000

    strategies = summary["strategies"]
    for key in (
        "a_tabular_argmax_baseline",
        "b_conviction_thresholded_tabular",
        "c_full_late_fusion_conflict_veto",
    ):
        assert key in strategies
        m = strategies[key]
        assert 0.0 <= m["macro_f1"] <= 1.0
        assert 0.0 <= m["active_signal_precision"] <= 1.0
        assert m["active_signal_count"] > 0

    # Verify Conflict Veto Gate triggered on conflicting shock rows and Late-Fusion improved precision
    assert strategies["c_full_late_fusion_conflict_veto"]["veto_count"] > 0
    assert (
        strategies["c_full_late_fusion_conflict_veto"]["active_signal_precision"]
        > strategies["a_tabular_argmax_baseline"]["active_signal_precision"]
    )
    assert (
        strategies["c_full_late_fusion_conflict_veto"]["active_signal_precision"]
        > strategies["b_conviction_thresholded_tabular"]["active_signal_precision"]
    )


def test_sentiment_series_zero_target_leakage() -> None:
    """Verify build_test_2026_sentiment_series does not use target_class (strictly causal day-t)."""
    from src.fusion.late_fusion import TEST_PREDICTIONS_PATH, build_test_2026_sentiment_series

    preds = pd.read_parquet(TEST_PREDICTIONS_PATH).head(300).copy()
    base_sent = build_test_2026_sentiment_series(preds)

    corrupted_preds = preds.copy()
    corrupted_preds["target_class"] = 2 - corrupted_preds["target_class"]
    corrupted_sent = build_test_2026_sentiment_series(corrupted_preds)

    np.testing.assert_allclose(
        base_sent["S_t"].to_numpy(),
        corrupted_sent["S_t"].to_numpy(),
        rtol=1e-12,
        atol=1e-12,
        err_msg="Target leakage detected in build_test_2026_sentiment_series!",
    )
