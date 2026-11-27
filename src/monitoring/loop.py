"""Closed loop in one command: new data -> drift check -> retrain if needed -> champion hot-reload.

    python -m src.monitoring.loop                 # fetch latest data, check drift, retrain on drift
    python -m src.monitoring.loop --skip-fetch    # reuse the already-fetched new batch
    python -m src.monitoring.loop --force-retrain # retrain even without drift

If API_URL is set (e.g. http://api:8000 inside docker-compose), the serving API is told to reload
the registry `champion` after a promotion, so the new model goes live without a restart.
"""

from __future__ import annotations

import argparse
import json
import os

import requests

from src.models.tracking import load_registered_model


def run(skip_fetch: bool = False, force_retrain: bool = False) -> dict:
    from src.monitoring.batch import build_new_batch_features
    from src.monitoring.drift import run_drift_check
    from src.monitoring.retrain import run_retraining

    summary: dict = {}
    if not skip_fetch:
        from src.data.fetch_new_batch import build_new_batch

        _, fetch_rep = build_new_batch()
        summary["fetch"] = {k: fetch_rep[k] for k in ("new_batch_range", "rows", "dropped_incomplete_sessions")}

    new_feats, _ = build_new_batch_features()
    champion, info = load_registered_model("champion")
    drift = run_drift_check(new_feats, model=champion)
    summary["drift"] = {k: drift[k] for k in ("current_range", "drift_share", "drifted_market_features",
                                              "prediction_class_psi", "retrain_recommended")}

    if drift["retrain_recommended"] or force_retrain:
        decision = run_retraining(trigger="drift" if drift["retrain_recommended"] else "manual")
        summary["retraining"] = {k: decision[k] for k in ("decision", "champion_before", "challenger_version",
                                                          "champion_after")}
        summary["retraining"]["champion_macro_f1"] = round(decision["champion_metrics"]["macro_f1"], 4)
        summary["retraining"]["challenger_macro_f1"] = round(decision["challenger_metrics"]["macro_f1"], 4)
        api = os.environ.get("API_URL")
        if api and decision["decision"] == "promoted":
            try:
                summary["api_reload"] = requests.post(f"{api}/model/reload", timeout=60).json()
            except Exception as exc:
                summary["api_reload"] = f"failed: {exc}"
    else:
        summary["retraining"] = {"decision": "skipped (no significant drift)", "champion": info["version"]}
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument("--force-retrain", action="store_true")
    args = ap.parse_args()
    print(json.dumps(run(args.skip_fetch, args.force_retrain), indent=2))
