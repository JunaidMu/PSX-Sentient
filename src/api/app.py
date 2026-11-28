"""PSX-Sentient Streamlit dashboard (Deliverable 2) — a thin client over the FastAPI service.

Tabs: latest signals for all 30 stocks · predict & explain (SHAP, sentiment veto, live feature
inputs) · monitoring (drift, served model, last retraining decision) · model performance.
Set API_URL (default http://localhost:8000) and optionally GRAFANA_URL / MLFLOW_URL.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
GRAFANA_URL = os.environ.get("GRAFANA_URL", "")
MLFLOW_URL = os.environ.get("MLFLOW_URL", "")
SYMBOLS = ["AKBL", "ATRL", "BAFL", "BAHL", "DGKC", "FCCL", "FFC", "HBL", "HUBC", "INDU", "INIL", "KAPCO", "LUCK",
           "MARI", "MCB", "MLCF", "MTL", "NBP", "NML", "NRL", "OGDC", "PIOC", "PKGS", "POL", "PPL", "PSO", "PTC",
           "SNGP", "SSGC", "UBL"]
BADGE = {"UP": "🟢 UP", "DOWN": "🔴 DOWN", "NEUTRAL": "⚪ NEUTRAL"}


def api_get(path: str) -> dict:
    r = requests.get(f"{API_URL}{path}", timeout=120)
    r.raise_for_status()
    return r.json()


def api_post(path: str, payload: dict | None = None) -> dict:
    r = requests.post(f"{API_URL}{path}", json=payload, timeout=120)
    if r.status_code >= 400:
        raise RuntimeError(r.json().get("detail", r.text))
    return r.json()


def _load_json(path: str) -> dict:
    p = PROJECT_ROOT / path
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


st.set_page_config(page_title="PSX-Sentient", page_icon="📈", layout="wide")
st.title("📈 PSX-Sentient — next-day signals for 30 KSE-100 stocks")

try:
    health = api_get("/health")
except Exception as exc:
    st.error(f"API not reachable at {API_URL}: {exc}")
    st.stop()

m = health["model"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Served model", f"{m.get('model_type')} v{m.get('version')}", m.get("alias"))
c2.metric("Stocks tracked", health["symbols"])
c3.metric("Data through", health["data_range"][1])
c4.metric("News sentiment (FinBERT)", "on" if health["nlp_enabled"] else "manual input")
links = " · ".join(x for x in [f"[API docs]({API_URL}/docs)",
                                f"[Grafana]({GRAFANA_URL})" if GRAFANA_URL else "",
                                f"[MLflow]({MLFLOW_URL})" if MLFLOW_URL else ""] if x)
st.caption(links)

tab_sig, tab_pred, tab_mon, tab_perf = st.tabs(["Latest signals", "Predict & explain", "Monitoring", "Model performance"])

with tab_sig:
    data = api_get("/signals/latest")
    st.subheader(f"Signals for the next session after {data['date']}")
    pm = data.get("premarket")
    if pm:
        ok = sum(v == "ok" for v in pm["sources"].values())
        st.info(f"Pre-market brief for {pm['trading_day']} (collected {pm['generated_at_pkt']} PKT): "
                f"{ok}/{len(pm['sources'])} PSX files · {pm['news']} · {pm['counts']['stocks_with_news']} stocks with fresh news · "
                f"{pm['counts']['whale_blocks']} off-market whale blocks · {pm['counts']['announcements']} announcements")
    rows = [{
        "Symbol": s["symbol"], "Close": s["close"],
        "P(DOWN)": s["probabilities"]["DOWN"], "P(NEUTRAL)": s["probabilities"]["NEUTRAL"], "P(UP)": s["probabilities"]["UP"],
        "Model call": BADGE[s["model_signal"]], "Confidence": s["confidence"],
        "News S_t": s.get("sentiment_S_t"), "Whale W_ndm": s.get("whale_score"),
        "Futures OI % float": s.get("futures_oi_pct_free_float"),
        "Final signal": BADGE[s["final_signal"]] + (" (veto)" if s["veto_triggered"] else ""),
        "Top driver": s["explanation"]["top_features"][0]["feature"],
        "Latest headline": (s.get("latest_headline") or {}).get("title", ""),
    } for s in data["signals"]]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("News S_t, whale and futures columns come from the 09:00 PKT pre-market job. They are context and "
               "feed the veto/blending rule, but are not model inputs (no training history yet).")
    st.caption("Model call = most likely class. Final signal = late-fusion rule: an UP/DOWN call needs net edge "
               "|0.75·(P_up−P_down) + 0.25·S_t| above the threshold tuned on 2023–25 validation and P(NEUTRAL) < 0.50, and is vetoed to NEUTRAL when "
               "news sentiment strongly contradicts it.")

with tab_pred:
    left, right = st.columns([1, 2])
    with left:
        sym = st.selectbox("Stock", SYMBOLS, index=SYMBOLS.index("OGDC"))
        date = st.text_input("Date (YYYY-MM-DD, blank = latest)", "")
        sentiment = st.slider("News sentiment S_t (−1 very negative … +1 very positive)", -1.0, 1.0, 0.0, 0.05)
        headline = st.text_input("…or a headline to score with FinBERT", "",
                                 disabled=not health["nlp_enabled"])
        live_edit = st.checkbox("Edit the input features (live what-if input)")
        payload: dict = {"symbol": sym, "date": date or None}
        if headline:
            payload["headline"] = headline
        elif sentiment != 0.0:
            payload["sentiment"] = sentiment
        if live_edit:
            try:
                stored = api_get(f"/features/{sym}" + (f"?date={date}" if date else ""))
                st.caption(f"Feature values for {sym} on {stored['date']} — change any value:")
                edited = st.data_editor(
                    pd.DataFrame({"value": stored["features"]}), width="stretch", key=f"edit_{sym}_{date}"
                )
                payload["features"] = {k: float(v) for k, v in edited["value"].items()}
                payload["date"] = stored["date"]
            except Exception as exc:
                st.error(str(exc))
        go = st.button("Predict", type="primary")
    with right:
        if go:
            try:
                res = api_post("/predict", payload)
                res.setdefault("date", payload.get("date"))
                st.markdown(f"### {sym} · {res['date']}: **{BADGE[res['final_signal']]}**")
                if res["veto_triggered"]:
                    st.warning("Conflict veto: the news sentiment contradicts the technical model, signal forced to NEUTRAL.")
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Model call", res["model_signal"])
                k2.metric("Confidence", f"{res['confidence']:.1%}")
                k3.metric("Net technical edge E_tech", f"{res['E_tech']:+.3f}")
                k4.metric("Blended conviction C_t", f"{res['C_t']:+.3f}")
                st.bar_chart(pd.Series(res["probabilities"]), horizontal=True)
                st.markdown(f"**Why {res['explanation']['explained_class']}?** SHAP contributions (positive pushes towards it):")
                shap_df = pd.DataFrame(res["explanation"]["top_features"]).set_index("feature")
                st.bar_chart(shap_df["shap"], horizontal=True)
                st.dataframe(shap_df, width="stretch")
                st.caption(f"Sentiment used for the veto: {res.get('sentiment_S_t')} (source: {res.get('sentiment_source')})")
                ctx = res.get("premarket_context")
                if ctx:
                    st.markdown(f"**Pre-market context ({ctx['trading_day']})**")
                    for h in ctx["sentiment"].get("latest_headlines", []):
                        st.write(f"📰 {h['title']}  _(FinBERT {h['score']:+.2f})_")
                    if ctx.get("whale"):
                        w = ctx["whale"]
                        st.write(f"🐋 Off-market: {w['ndm_volume']:,} shares at {w['ndm_deal_rate']} "
                                 f"({w['ndm_spread_pct']:+.2f}% vs close, {w['ndm_vol_pct_of_sma20']:.1f}% of 20-day volume) "
                                 f"→ W_ndm {w['W_ndm']:+.2f}")
                    if ctx.get("futures_open_interest"):
                        st.write(f"📈 Futures open interest: {ctx['futures_open_interest']['pct_free_float']}% of free float")
                    for a in ctx.get("announcements", []):
                        st.write(f"📢 {a}")
            except Exception as exc:
                st.error(str(exc))

with tab_mon:
    try:
        drift = api_get("/drift")
        d1, d2, d3 = st.columns(3)
        d1.metric("Market features drifting", f"{drift['drift_share']:.0%}")
        d2.metric("Prediction class-mix PSI", drift.get("prediction_class_psi"))
        d3.metric("Retraining recommended", "YES" if drift["retrain_recommended"] else "no")
        st.caption(f"Window {drift['current_range'][0]} → {drift['current_range'][1]} ({drift['current_rows']:,} rows) vs training data.")
        feats = pd.DataFrame(drift["features"]).T.sort_values("psi", ascending=False)
        st.dataframe(feats, width="stretch")
    except Exception as exc:
        st.info(f"No drift report yet ({exc}).")
    info = api_get("/model")
    st.subheader("Served model & last retraining")
    st.json(info["served"])
    if "last_retraining" in info:
        lr = info["last_retraining"]
        st.markdown(f"**Decision:** {lr['decision']} (champion v{lr['champion_before']} → v{lr['champion_after']}, "
                    f"compared on {lr['compare_window'][0]} → {lr['compare_window'][1]})")
        st.dataframe(pd.DataFrame({"champion": lr["champion_metrics"], "challenger": lr["challenger_metrics"]}).round(4),
                     width="stretch")

with tab_perf:
    comp, base, stack = (_load_json("models/model_comparison_metrics.json"), _load_json("models/baseline_metrics.json"),
                         _load_json("models/stacking_metrics.json"))
    rows = []
    for name, r in base.get("baselines", {}).items():
        rows.append({"Model": name, "Type": "baseline", "Val F1": r["val_mean"]["macro_f1"],
                     "Test 2026 F1": r["test_2026"]["macro_f1"], "Test active precision": r["test_2026"]["active_signal_precision"]})
    for name, r in comp.get("models", {}).items():
        rows.append({"Model": name, "Type": "primary", "Val F1": r["val_mean"]["macro_f1"],
                     "Test 2026 F1": r["test_2026"]["macro_f1"], "Test active precision": r["test_2026"]["active_signal_precision"]})
    if stack:
        rows.append({"Model": "Stacking meta-learner", "Type": "late fusion", "Val F1": stack["val_mean"]["macro_f1"],
                     "Test 2026 F1": stack["test_2026"]["macro_f1"], "Test active precision": stack["test_2026"]["active_signal_precision"]})
    st.dataframe(pd.DataFrame(rows).round(4), hide_index=True, width="stretch")
    st.caption("Test 2026 = Jan–Jun 2026, never used for model selection. Active precision = share of UP/DOWN calls that were right (≈23% by chance).")
