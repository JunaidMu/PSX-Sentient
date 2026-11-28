"""Closed loop: new data -> drift check -> (re)train when the schedule and data allow -> hot-reload.

Retraining policy
- daily  (09:00 PKT, inside scripts/premarket.sh): drift check + alerts. Retrains only for *urgent*
  drift (>= URGENT_DRIFT_SHARE of market features), and only if there are >= MIN_NEW_DAYS new
  labelled trading days since the last retraining (a one-week cooldown).
- weekly (Saturday, scripts/weekly_retrain.sh): retrains when the drift rule fires, or as a routine
  refresh once >= REFRESH_DAYS new labelled days have accumulated, again with the cooldown.
- --force-retrain: always retrain (manual / demo).

Drift is measured against the data the *current champion* was trained on (its `train_end` tag), using
the newest data the champion has not seen (up to the last CURRENT_WINDOW_DAYS trading days). After a
promotion the reference moves forward with the new champion, so the alarm resets until the market
changes again.

    python -m src.monitoring.loop --mode weekly [--skip-fetch] [--force-retrain]

If API_URL is set (e.g. http://api:8000 in docker-compose), the API reloads the champion after a promotion.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import pandas as pd
import requests

from src.models.tracking import load_registered_model

MIN_NEW_DAYS = 5  # one trading week of new labelled data since the last retraining
URGENT_DRIFT_SHARE = 0.5  # daily runs retrain only when at least half the market features drift
REFRESH_DAYS = 20  # weekly runs refresh the model after ~4 weeks of new data even without drift
CURRENT_WINDOW_DAYS = 60  # newest trading days (unseen by the champion) used as the drift window
MIN_DRIFT_ROWS = 300


def decide_retrain(mode: str, drift: dict[str, Any] | None, new_days: int, force: bool = False) -> tuple[bool, str]:
    """Pure retraining decision (unit-tested)."""
    if force:
        return True, "forced (manual)"
    if new_days < MIN_NEW_DAYS:
        return False, f"cooldown: only {new_days} new labelled trading days since the last retraining (need {MIN_NEW_DAYS})"
    share = (drift or {}).get("drift_share", 0.0)
    recommended = bool((drift or {}).get("retrain_recommended"))
    if mode == "daily":
        if share >= URGENT_DRIFT_SHARE:
            return True, f"urgent drift: {share:.0%} of market features drifting"
        return False, "daily run: drift checked, retraining waits for the weekly run"
    if recommended:
        return True, f"weekly run: drift rule fired ({share:.0%} of market features drifting)"
    if new_days >= REFRESH_DAYS:
        return True, f"weekly run: routine refresh ({new_days} new labelled trading days)"
    return False, "weekly run: no significant drift and not enough new data for a routine refresh"


def _new_labelled_days(labelled: pd.DataFrame, since: str) -> int:
    dates = pd.to_datetime(labelled["date"]).drop_duplicates()
    return int((dates > pd.Timestamp(since)).sum())


def run(mode: str = "weekly", skip_fetch: bool = False, force_retrain: bool = False) -> dict:
    from src.monitoring.batch import build_new_batch_features
    from src.monitoring.drift import run_drift_check
    from src.monitoring.retrain import last_retrain_data_end, run_retraining

    summary: dict[str, Any] = {"mode": mode}
    if not skip_fetch:
        from src.data.fetch_new_batch import build_new_batch

        _, fetch_rep = build_new_batch()
        summary["fetch"] = {k: fetch_rep[k] for k in ("new_batch_range", "rows", "dropped_incomplete_sessions")}

    new_feats, labelled = build_new_batch_features()
    champion, info = load_registered_model("champion")
    train_end = info.get("train_end", "2025-12-31")
    summary["champion"] = {"version": info["version"], "train_end": train_end}

    # Drift: newest data the champion has never seen vs. the data it was trained on
    unseen = new_feats[pd.to_datetime(new_feats["date"]) > pd.Timestamp(train_end)]
    recent_dates = sorted(pd.to_datetime(unseen["date"]).unique())[-CURRENT_WINDOW_DAYS:]
    current = unseen[pd.to_datetime(unseen["date"]).isin(recent_dates)]
    drift = None
    if len(current) >= MIN_DRIFT_ROWS:
        drift = run_drift_check(current, model=champion, reference_end=train_end)
        summary["drift"] = {k: drift[k] for k in ("current_range", "reference_end", "drift_share",
                                                  "drifted_market_features", "prediction_class_psi",
                                                  "retrain_recommended")}
    else:
        summary["drift"] = f"skipped: only {len(current)} unseen rows (need {MIN_DRIFT_ROWS})"

    since = last_retrain_data_end() or train_end
    new_days = _new_labelled_days(labelled, since)
    do_it, reason = decide_retrain(mode, drift, new_days, force_retrain)
    summary["decision"] = {"retrain": do_it, "reason": reason, "new_labelled_days": new_days, "since": since}

    if do_it:
        decision = run_retraining(trigger=reason.split(":")[0])
        summary["retraining"] = {k: decision[k] for k in ("decision", "champion_before", "challenger_version",
                                                          "champion_after", "compare_window", "challenger_train_end")}
        summary["retraining"]["champion_macro_f1"] = round(decision["champion_metrics"]["macro_f1"], 4)
        summary["retraining"]["challenger_macro_f1"] = round(decision["challenger_metrics"]["macro_f1"], 4)
        api = os.environ.get("API_URL")
        if api and decision["decision"] == "promoted":
            try:
                summary["api_reload"] = requests.post(f"{api}/model/reload", timeout=60).json()
            except Exception as exc:
                summary["api_reload"] = f"failed: {exc}"
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["daily", "weekly"], default="weekly")
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument("--force-retrain", action="store_true")
    args = ap.parse_args()
    print(json.dumps(run(args.mode, args.skip_fetch, args.force_retrain), indent=2, default=str))
