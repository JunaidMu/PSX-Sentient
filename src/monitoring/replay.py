"""Replay the post-cutoff trading days through the live API as if they were arriving day by day.

Each (symbol, date) of the new batch is sent to POST /predict in date order, so request latency,
errors and the prediction mix show up in Prometheus/Grafana, and every served input lands in the
prediction log. Afterwards POST /drift/check compares the served inputs with the training data.

    python -m src.monitoring.replay --api http://localhost:8000 [--days 60] [--delay 0.0]
"""

from __future__ import annotations

import argparse
import json
import os
import time

import pandas as pd
import requests

from src.monitoring.batch import NEW_FEATURES_PATH


def replay(api: str, days: int | None = None, delay: float = 0.0) -> dict:
    df = pd.read_parquet(NEW_FEATURES_PATH)
    dates = sorted(df["date"].unique())
    if days:
        dates = dates[:days]
    session = requests.Session()
    sent = errors = 0
    for i, d in enumerate(dates, start=1):
        for sym in df.loc[df["date"] == d, "symbol"]:
            r = session.post(f"{api}/predict", json={"symbol": sym, "date": str(pd.Timestamp(d).date())}, timeout=30)
            sent += 1
            errors += r.status_code != 200
        if delay:
            time.sleep(delay)
        if i % 20 == 0:
            print(f"  replayed {i}/{len(dates)} trading days ({sent} predictions)")
    drift = session.post(f"{api}/drift/check", timeout=300, headers={"X-Admin-Key": os.environ.get("ADMIN_API_KEY", "")})
    return {"days": len(dates), "predictions": sent, "errors": errors,
            "drift": drift.json() if drift.status_code == 200 else drift.text}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--delay", type=float, default=0.0, help="seconds to wait between replayed days")
    a = ap.parse_args()
    print(json.dumps(replay(a.api, a.days, a.delay), indent=2))
