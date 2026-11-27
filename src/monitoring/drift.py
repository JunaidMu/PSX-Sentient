"""Input & prediction drift detection: PSI + KS per feature, with an HTML report.

Reference = the champion's training window (feature store rows before 2026-01-01).
Current   = the post-cutoff new batch (or any batch of served inputs).

Decision rule (the retraining trigger):
- A market feature "drifts" when PSI >= 0.2 (significant shift) or the KS test rejects at p < 0.01
  with KS statistic >= 0.1 (large samples make p-values alone over-sensitive).
- Calendar features (month/day-of-week encodings, season flags) are reported but excluded from the
  decision: they shift by construction whenever the current window covers different months.
- Retraining is triggered when >= 25% of market features drift, or the predicted-class mix moves
  by PSI >= 0.1.
Outputs reports/drift_report.json and reports/drift_report.html.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.features.build_features import FEATURE_COLUMNS, OUTPUT_PARQUET_PATH

CALENDAR_FEATURES = {"month_sin", "month_cos", "dow_sin", "dow_cos", "is_earnings_month", "is_winter_gas", "is_monsoon"}
MARKET_FEATURES = [c for c in FEATURE_COLUMNS if c not in CALENDAR_FEATURES]

PSI_DRIFT = 0.2
KS_P_VALUE = 0.01
KS_MIN_STAT = 0.1
DRIFT_SHARE_TRIGGER = 0.25
PREDICTION_PSI_TRIGGER = 0.1
REFERENCE_END = "2025-12-31"

REPORT_JSON = Path("reports/drift_report.json")
REPORT_HTML = Path("reports/drift_report.html")


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10, eps: float = 1e-4) -> float:
    """Population Stability Index using reference-quantile bins."""
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if len(reference) == 0 or len(current) == 0:
        return float("nan")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # (near-)constant feature: compare category shares instead
        cats = np.union1d(np.unique(reference), np.unique(current))
        ref_p = np.array([(reference == c).mean() for c in cats])
        cur_p = np.array([(current == c).mean() for c in cats])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
        ref_p = np.histogram(reference, edges)[0] / len(reference)
        cur_p = np.histogram(current, edges)[0] / len(current)
    ref_p, cur_p = np.clip(ref_p, eps, None), np.clip(cur_p, eps, None)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def class_mix_psi(ref_pred: np.ndarray, cur_pred: np.ndarray) -> float:
    ref_p = np.array([(ref_pred == c).mean() for c in (0, 1, 2)])
    cur_p = np.array([(cur_pred == c).mean() for c in (0, 1, 2)])
    ref_p, cur_p = np.clip(ref_p, 1e-4, None), np.clip(cur_p, 1e-4, None)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    ref_pred: np.ndarray | None = None,
    cur_pred: np.ndarray | None = None,
) -> dict[str, Any]:
    features: dict[str, dict[str, Any]] = {}
    for col in FEATURE_COLUMNS:
        r = reference[col].to_numpy(dtype="float64")
        c = current[col].to_numpy(dtype="float64")
        p = psi(r, c)
        ks = ks_2samp(r[np.isfinite(r)], c[np.isfinite(c)])
        drifted = bool(p >= PSI_DRIFT or (ks.pvalue < KS_P_VALUE and ks.statistic >= KS_MIN_STAT))
        features[col] = {
            "psi": round(p, 4),
            "ks_stat": round(float(ks.statistic), 4),
            "ks_pvalue": float(ks.pvalue),
            "drifted": drifted,
            "calendar": col in CALENDAR_FEATURES,
        }

    drifted_market = [c for c in MARKET_FEATURES if features[c]["drifted"]]
    share = len(drifted_market) / len(MARKET_FEATURES)
    pred_psi = None
    if ref_pred is not None and cur_pred is not None:
        pred_psi = round(class_mix_psi(np.asarray(ref_pred), np.asarray(cur_pred)), 4)

    retrain = share >= DRIFT_SHARE_TRIGGER or (pred_psi is not None and pred_psi >= PREDICTION_PSI_TRIGGER)
    return {
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "current_range": [str(pd.to_datetime(current["date"]).min().date()), str(pd.to_datetime(current["date"]).max().date())],
        "drifted_market_features": drifted_market,
        "drift_share": round(share, 4),
        "prediction_class_psi": pred_psi,
        "retrain_recommended": bool(retrain),
        "thresholds": {
            "psi": PSI_DRIFT, "ks_pvalue": KS_P_VALUE, "ks_min_stat": KS_MIN_STAT,
            "drift_share_trigger": DRIFT_SHARE_TRIGGER, "prediction_psi_trigger": PREDICTION_PSI_TRIGGER,
        },
        "features": features,
    }


def write_html_report(result: dict[str, Any], path: Path = REPORT_HTML) -> bool:
    """Save a self-contained HTML drift report (Evidently is incompatible with pandas 3)."""
    rows = []
    for name, f in result["features"].items():
        color = "#fde2e1" if f["drifted"] and not f["calendar"] else ("#f3f3f3" if f["calendar"] else "white")
        rows.append(
            f"<tr style='background:{color}'><td>{name}</td><td>{f['psi']:.3f}</td><td>{f['ks_stat']:.3f}</td>"
            f"<td>{f['ks_pvalue']:.2e}</td><td>{'DRIFT' if f['drifted'] else 'ok'}"
            f"{' (calendar, ignored)' if f['calendar'] else ''}</td></tr>"
        )
    verdict = "RETRAIN RECOMMENDED" if result["retrain_recommended"] else "No retraining needed"
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>PSX-Sentient drift report</title>
<style>body{{font-family:system-ui,sans-serif;margin:2rem;max-width:960px}}table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #ccc;padding:4px 8px;text-align:left}}th{{background:#eee}}</style></head><body>
<h1>PSX-Sentient input &amp; prediction drift</h1>
<p>Current window {result['current_range'][0]} &rarr; {result['current_range'][1]} ({result['current_rows']:,} rows)
vs. training reference ({result['reference_rows']:,} rows).</p>
<p><b>{verdict}</b> &mdash; drifted market features: {len(result['drifted_market_features'])}
({result['drift_share']:.0%}), prediction class-mix PSI: {result['prediction_class_psi']}</p>
<p>Rule: feature drifts if PSI &ge; {PSI_DRIFT} or (KS p &lt; {KS_P_VALUE} and KS &ge; {KS_MIN_STAT});
retrain if &ge; {DRIFT_SHARE_TRIGGER:.0%} of market features drift or class-mix PSI &ge; {PREDICTION_PSI_TRIGGER}.</p>
<table><tr><th>feature</th><th>PSI</th><th>KS stat</th><th>KS p-value</th><th>status</th></tr>{''.join(rows)}</table>
</body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return True


def load_reference(feature_store_path: Path = OUTPUT_PARQUET_PATH) -> pd.DataFrame:
    ref = pd.read_parquet(feature_store_path)
    ref["date"] = pd.to_datetime(ref["date"])
    return ref[ref["date"] <= REFERENCE_END].reset_index(drop=True)


def run_drift_check(current: pd.DataFrame, model: Any | None = None, html: bool = True) -> dict[str, Any]:
    reference = load_reference()
    ref_pred = cur_pred = None
    if model is not None:
        ref_sample = reference.sample(n=min(len(reference), 20000), random_state=42)
        ref_pred = np.argmax(model.predict_proba(ref_sample[FEATURE_COLUMNS]), axis=1)
        cur_pred = np.argmax(model.predict_proba(current[FEATURE_COLUMNS]), axis=1)
    result = detect_drift(reference, current, ref_pred, cur_pred)
    result["html_report"] = str(REPORT_HTML) if html and write_html_report(result) else None
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    import joblib

    from src.monitoring.batch import NEW_FEATURES_PATH

    cur = pd.read_parquet(NEW_FEATURES_PATH)
    res = run_drift_check(cur, model=joblib.load("models/best_model.pkl"))
    print(f"Current window {res['current_range']} ({res['current_rows']} rows) vs reference ({res['reference_rows']} rows)")
    print(f"{'feature':<16} {'PSI':>7} {'KS':>6} drift")
    for name, f in res["features"].items():
        flag = "YES" if f["drifted"] else ""
        print(f"{name:<16} {f['psi']:>7.3f} {f['ks_stat']:>6.3f} {flag}{' (calendar)' if f['calendar'] and flag else ''}")
    print(f"\nMarket features drifted: {res['drifted_market_features']} -> share {res['drift_share']:.0%}")
    print(f"Prediction class-mix PSI: {res['prediction_class_psi']}")
    print(f"RETRAIN RECOMMENDED: {res['retrain_recommended']}   (report: {REPORT_JSON}, html: {res["html_report"]})")
