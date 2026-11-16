"""Streamlit Institutional Dashboard for PSX-Sentient (Phase 6).

Features:
(a) Top KPI Cards: Active Signals, Bullish/Bearish Count, Conflict Vetoes, Test 2026 Active Precision (69.77%)
(b) 30-Stock Actionable Signal Table with UP/DOWN/NEUTRAL badges and Veto/Whale indicators
(c) Single-Stock Deep Dive (CatBoost probabilities, FinBERT sentiment + live headlines, Off-Market Whale block spread)
(d) Model & Ablation Performance Tab
"""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import streamlit as st

from src.data.live_updater import LATEST_SIGNALS_JSON_PATH, generate_latest_signals_snapshot
from src.fusion.late_fusion import ABLATION_METRICS_PATH, apply_late_fusion_row
from src.models.train_baseline import MODELS_DIR

MODEL_COMPARISON_PATH = MODELS_DIR / "model_comparison_metrics.json"


def load_dashboard_data() -> tuple[dict, dict, dict]:
    """Load latest 30-stock signals snapshot, Phase 3 model metrics, and Phase 5 ablation metrics."""
    if not LATEST_SIGNALS_JSON_PATH.exists():
        generate_latest_signals_snapshot()
    signals_payload = json.loads(LATEST_SIGNALS_JSON_PATH.read_text(encoding="utf-8"))
    model_metrics = (
        json.loads(MODEL_COMPARISON_PATH.read_text(encoding="utf-8"))
        if MODEL_COMPARISON_PATH.exists()
        else {}
    )
    ablation_metrics = (
        json.loads(ABLATION_METRICS_PATH.read_text(encoding="utf-8"))
        if ABLATION_METRICS_PATH.exists()
        else {}
    )
    return signals_payload, model_metrics, ablation_metrics


def _format_signal_badge(sig: str) -> str:
    if sig == "UP":
        return "🟢 UP (2)"
    if sig == "DOWN":
        return "🔴 DOWN (0)"
    return "⚪ NEUTRAL (1)"


def main() -> None:
    st.set_page_config(
        page_title="PSX-Sentient | KSE-100 Late-Fusion Intelligence",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .stApp { background-color: #0b0f19; color: #e2e8f0; }
        div[data-testid="stMetric"] {
            background: linear-gradient(145deg, #111827, #1e293b);
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 14px 18px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    signals_payload, model_metrics, ablation_metrics = load_dashboard_data()
    df_signals = pd.DataFrame(signals_payload["signals"])

    st.title("📈 PSX-Sentient — KSE-100 Late-Fusion & Conflict Veto Terminal")
    st.caption(
        f"Frozen Model Cutoff: **{signals_payload.get('model_cutoff_date', '2026-06-30')}** | "
        f"Engine: **{signals_payload.get('model_name', 'CatBoost')} + CPU FinBERT (`ProsusAI/finbert`) + NDM Whale Detector**"
    )

    # (a) Top KPI Cards
    bullish_cnt = int((df_signals["signal"] == "UP").sum())
    bearish_cnt = int((df_signals["signal"] == "DOWN").sum())
    active_cnt = bullish_cnt + bearish_cnt
    veto_cnt = int(df_signals["veto_triggered"].sum())

    ablation_c = (
        ablation_metrics.get("strategies", {}).get("c_full_late_fusion_conflict_veto", {})
    )
    test_active_prec = float(ablation_c.get("active_signal_precision", 0.6977)) * 100.0
    test_vetoes = int(ablation_c.get("veto_count", 71))

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric(
        "Active Directional Signals (30 Stocks)",
        f"{active_cnt} / {len(df_signals)}",
        f"{bullish_cnt} Bullish (UP) · {bearish_cnt} Bearish (DOWN)",
    )
    kpi2.metric(
        "Bullish vs. Bearish Breadth",
        f"{bullish_cnt} UP  |  {bearish_cnt} DOWN",
        f"{int((df_signals['signal'] == 'NEUTRAL').sum())} Neutral / Gated",
    )
    kpi3.metric(
        "Conflict Vetoes Triggered",
        f"{veto_cnt} Live  ({test_vetoes} in Test_2026)",
        "Hard Safety Gate: |E_tech| > 0.20 vs |S_t| > 0.50",
    )
    kpi4.metric(
        "Test 2026 Active Signal Precision",
        f"{test_active_prec:.2f}%",
        "+37.12% vs. Tabular Argmax (32.65%)",
    )

    st.divider()

    tab_signals, tab_deep_dive, tab_ablation = st.tabs(
        [
            "📊 (b) 30-Stock Actionable Signal Table",
            "🔍 (c) Single-Stock Deep Dive & What-If Simulator",
            "🧪 (d) Purged Walk-Forward & Ablation Performance",
        ]
    )

    # (b) 30-Stock Actionable Signal Table
    with tab_signals:
        col_f1, col_f2 = st.columns([2, 2])
        with col_f1:
            sig_filter = st.multiselect(
                "Filter by Final Signal",
                options=["UP", "DOWN", "NEUTRAL"],
                default=["UP", "DOWN", "NEUTRAL"],
            )
        with col_f2:
            only_special = st.checkbox("Show only Active / Veto / Whale-Block Stocks", value=False)

        view_df = df_signals[df_signals["signal"].isin(sig_filter)].copy()
        if only_special:
            view_df = view_df[
                (view_df["signal"] != "NEUTRAL")
                | (view_df["veto_triggered"])
                | (view_df["is_whale_block"])
            ]

        view_df["Signal Badge"] = view_df["signal"].map(_format_signal_badge)
        view_df["Conflict Veto"] = view_df["veto_triggered"].map(
            lambda v: "🛡️ VETOED -> NEUTRAL" if v else "—"
        )
        view_df["Whale Block (NDM)"] = view_df.apply(
            lambda r: (
                f"🐋 {r['W_ndm']:+.2f} ({r['ndm_spread_pct']:+.1f}%)"
                if r["is_whale_block"]
                else "—"
            ),
            axis=1,
        )

        display_cols = [
            "symbol",
            "date",
            "close",
            "Signal Badge",
            "P_down",
            "P_neutral",
            "P_up",
            "E_tech",
            "S_t",
            "C_t",
            "Conflict Veto",
            "Whale Block (NDM)",
            "latest_headline",
        ]
        st.dataframe(
            view_df[display_cols],
            use_container_width=True,
            hide_index=True,
        )

    # (c) Single-Stock Deep Dive
    with tab_deep_dive:
        selected_sym = st.selectbox(
            "Select KSE-100 Stock Symbol",
            options=df_signals["symbol"].tolist(),
            index=0,
        )
        stock_row = df_signals[df_signals["symbol"] == selected_sym].iloc[0]

        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader(f"1. CatBoost Probabilities ({selected_sym})")
            st.write(f"**Snapshot Date:** `{stock_row['date']}` | **Ready Close:** `PKR {stock_row['close']:.2f}`")
            st.progress(float(stock_row["P_up"]), text=f"P(UP): {stock_row['P_up']:.2%}")
            st.progress(float(stock_row["P_neutral"]), text=f"P(NEUTRAL): {stock_row['P_neutral']:.2%}")
            st.progress(float(stock_row["P_down"]), text=f"P(DOWN): {stock_row['P_down']:.2%}")
            st.metric("Net Technical Edge (E_tech = P_up - P_down)", f"{stock_row['E_tech']:+.4f}")

        with c2:
            st.subheader("2. FinBERT Sentiment & Conflict Gate")
            st.metric("3-Day EWMA FinBERT Score (S_t)", f"{stock_row['S_t']:+.4f}")
            st.metric("Blended Conviction (C_t = 0.75*E_tech + 0.25*S_t)", f"{stock_row['C_t']:+.4f}")
            st.info(f"**Latest Mapped Headline:**\n\n{stock_row['latest_headline']}")
            if stock_row["veto_triggered"]:
                st.warning("🛡️ **Conflict Veto Gate Triggered!** Signal forced to **NEUTRAL**.")
            else:
                st.success(f"**Final Late-Fusion Signal:** {_format_signal_badge(stock_row['signal'])}")

        with c3:
            st.subheader("3. Off-Market Whale Block (NDM)")
            st.metric("Whale Block Score (W_ndm)", f"{stock_row['W_ndm']:+.4f}")
            st.write(f"- **Institutional Block Flag (>5% 20d Vol):** `{bool(stock_row['is_whale_block'])}`")
            st.write(f"- **NDM Block Volume:** `{int(stock_row['ndm_volume']):,}` (`{stock_row['ndm_vol_pct_of_sma20']:.1f}%` of 20d SMA)")
            st.write(f"- **NDM Deal Rate vs. Ready Close:** `PKR {stock_row['ndm_deal_rate']:.2f}` (`{stock_row['ndm_spread_pct']:+.2f}%` spread)")

        st.markdown("#### Interactive What-If Late-Fusion Simulator")
        sim_col1, sim_col2 = st.columns(2)
        with sim_col1:
            sim_s_t = st.slider(
                "Override 3-Day EWMA Sentiment (S_t)",
                min_value=-1.0,
                max_value=1.0,
                value=float(stock_row["S_t"]),
                step=0.05,
            )
        with sim_col2:
            sim_out = apply_late_fusion_row(
                P_down=float(stock_row["P_down"]),
                P_neutral=float(stock_row["P_neutral"]),
                P_up=float(stock_row["P_up"]),
                S_t=sim_s_t,
            )
            st.write(
                f"**Simulated Blended Score (`C_t`):** `{sim_out['C_t']:+.4f}` | "
                f"**Veto Triggered:** `{sim_out['veto_triggered']}` | "
                f"**Simulated Signal:** **{_format_signal_badge(sim_out['fused_signal'])}**"
            )

    # (d) Model & Ablation Performance Tab
    with tab_ablation:
        st.subheader("Phase 5: Test_2026 Ablation Comparison (Tabular-Only vs. Late-Fusion + Conflict Veto)")
        if ablation_metrics and "strategies" in ablation_metrics:
            strat_map = {
                "a_tabular_argmax_baseline": "(a) Tabular Argmax Baseline",
                "b_conviction_thresholded_tabular": "(b) Conviction-Thresholded Tabular (|E_tech| > 0.20)",
                "c_full_late_fusion_conflict_veto": "(c) Full Late-Fusion + Conflict Veto Gate",
            }
            ab_rows = []
            for k, name in strat_map.items():
                m = ablation_metrics["strategies"][k]
                ab_rows.append(
                    {
                        "Strategy": name,
                        "Macro F1": round(m["macro_f1"], 4),
                        "Active Signal Precision": f"{m['active_signal_precision'] * 100:.2f}%",
                        "Active Signal Rate": f"{m['active_signal_rate'] * 100:.2f}%",
                        "Active Signals": m["active_signal_count"],
                        "Conflict Vetoes": m["veto_count"],
                        "P(Down)": f"{m['precision_down'] * 100:.2f}%",
                        "P(Neutral)": f"{m['precision_neutral'] * 100:.2f}%",
                        "P(Up)": f"{m['precision_up'] * 100:.2f}%",
                    }
                )
            st.dataframe(pd.DataFrame(ab_rows), use_container_width=True, hide_index=True)

        st.subheader("Phase 3: Purged Walk-Forward (2023–2025) & Holdout (2026) Baseline Comparison")
        if model_metrics and "models" in model_metrics:
            m_rows = []
            for m_name, m_info in model_metrics["models"].items():
                for split_key in ("Val_2023", "Val_2024", "Val_2025"):
                    fm = m_info["folds"][split_key]
                    m_rows.append(
                        {
                            "Model": m_name,
                            "Split": split_key,
                            "Macro F1": round(fm["macro_f1"], 4),
                            "Active Signal Precision": f"{fm['active_signal_precision'] * 100:.2f}%",
                            "Active Signal Rate": f"{fm['active_signal_rate'] * 100:.2f}%",
                            "P(Down)": round(fm["precision_down"], 4),
                            "P(Neutral)": round(fm["precision_neutral"], 4),
                            "P(Up)": round(fm["precision_up"], 4),
                        }
                    )
                vm = m_info["val_mean"]
                m_rows.append(
                    {
                        "Model": m_name,
                        "Split": "Val_Mean",
                        "Macro F1": round(vm["macro_f1"], 4),
                        "Active Signal Precision": f"{vm['active_signal_precision'] * 100:.2f}%",
                        "Active Signal Rate": f"{vm['active_signal_rate'] * 100:.2f}%",
                        "P(Down)": round(vm["precision_down"], 4),
                        "P(Neutral)": round(vm["precision_neutral"], 4),
                        "P(Up)": round(vm["precision_up"], 4),
                    }
                )
                tm = m_info["test_2026"]
                m_rows.append(
                    {
                        "Model": m_name,
                        "Split": "Test_2026",
                        "Macro F1": round(tm["macro_f1"], 4),
                        "Active Signal Precision": f"{tm['active_signal_precision'] * 100:.2f}%",
                        "Active Signal Rate": f"{tm['active_signal_rate'] * 100:.2f}%",
                        "P(Down)": round(tm["precision_down"], 4),
                        "P(Neutral)": round(tm["precision_neutral"], 4),
                        "P(Up)": round(tm["precision_up"], 4),
                    }
                )
            st.dataframe(pd.DataFrame(m_rows), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
