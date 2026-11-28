"""Daily backups to S3 so the server is disposable.

- MLflow registry/tracking database (SQLite, copied with the online-backup API so it is consistent
  while the MLflow server is running) -> s3://<bucket>/backups/mlflow/mlflow-<date>.db and latest.db
- Data-version snapshot: every DVC pointer file (*.dvc), dvc.lock and the JSON reports
  -> s3://<bucket>/backups/snapshots/<date>/...  (with these, `dvc pull` restores the data of that day)

The bucket is taken from BACKUP_BUCKET, or from MLFLOW_ARTIFACTS_DESTINATION (s3://bucket/prefix).
Model artifacts already live in S3 (MLflow artifact store), so nothing else is needed to rebuild.

    python -m src.monitoring.backup            # back up now
    python -m src.monitoring.backup --restore  # restore the latest registry DB (on a new server)
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sqlite3
import tempfile

MLFLOW_DB = Path(os.environ.get("MLFLOW_DB_PATH", "mlflow_server/mlflow.db"))
SNAPSHOT_GLOBS = ("psx_historical_data/*.dvc", "psx_historical_data/new_batch/*.dvc", "dvc.lock", "reports/*.json")


def bucket_name() -> str | None:
    if os.environ.get("BACKUP_BUCKET"):
        return os.environ["BACKUP_BUCKET"]
    dest = os.environ.get("MLFLOW_ARTIFACTS_DESTINATION", "")
    return dest[5:].split("/")[0] if dest.startswith("s3://") else None


def _s3():
    import boto3

    return boto3.client("s3")


def backup(today: str | None = None) -> dict:
    bucket = bucket_name()
    if not bucket:
        return {"status": "skipped", "reason": "no S3 bucket configured (BACKUP_BUCKET / MLFLOW_ARTIFACTS_DESTINATION)"}
    today = today or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    s3 = _s3()
    out: dict = {"bucket": bucket, "date": today, "uploaded": []}

    if MLFLOW_DB.exists():
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "mlflow.db"
            src, dst = sqlite3.connect(MLFLOW_DB), sqlite3.connect(copy)
            with dst:
                src.backup(dst)  # consistent snapshot even while MLflow is writing
            src.close()
            dst.close()
            for key in (f"backups/mlflow/mlflow-{today}.db", "backups/mlflow/latest.db"):
                s3.upload_file(str(copy), bucket, key)
                out["uploaded"].append(key)

    for pattern in SNAPSHOT_GLOBS:
        for f in sorted(Path(".").glob(pattern)):
            key = f"backups/snapshots/{today}/{f.as_posix()}"
            s3.upload_file(str(f), bucket, key)
            out["uploaded"].append(key)
    out["status"] = "ok"
    return out


def restore(force: bool = False) -> dict:
    bucket = bucket_name()
    if not bucket:
        return {"status": "skipped", "reason": "no S3 bucket configured"}
    if MLFLOW_DB.exists() and not force:
        return {"status": "skipped", "reason": f"{MLFLOW_DB} exists (use --force to overwrite)"}
    MLFLOW_DB.parent.mkdir(parents=True, exist_ok=True)
    _s3().download_file(bucket, "backups/mlflow/latest.db", str(MLFLOW_DB))
    return {"status": "restored", "from": f"s3://{bucket}/backups/mlflow/latest.db", "to": str(MLFLOW_DB)}


if __name__ == "__main__":
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--restore", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    print(json.dumps(restore(a.force) if a.restore else backup(), indent=2))
