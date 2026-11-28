"""Tests for drift detection, the retraining windows and the CI evaluation gate."""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.features.build_features import FEATURE_COLUMNS
from src.monitoring import gate
from src.monitoring.drift import detect_drift, psi
from src.monitoring.backup import bucket_name
from src.monitoring.loop import decide_retrain
from src.monitoring.retrain import COMPARE_DAYS, PURGE_DAYS, _windows


def _frame(n: int, shift: float = 0.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(shift, 1.0, n) for c in FEATURE_COLUMNS})
    df["date"] = pd.bdate_range("2025-01-01", periods=n)
    return df


def test_psi_zero_for_same_distribution_and_large_for_shift() -> None:
    rng = np.random.default_rng(1)
    a, b = rng.normal(0, 1, 20000), rng.normal(0, 1, 20000)
    assert psi(a, b) < 0.01
    assert psi(a, rng.normal(1.0, 1, 20000)) > 0.2


def test_no_drift_when_distributions_match() -> None:
    res = detect_drift(_frame(5000, seed=1), _frame(2000, seed=2))
    assert res["drift_share"] == 0.0 and res["retrain_recommended"] is False


def test_market_drift_triggers_retraining_but_calendar_drift_alone_does_not() -> None:
    ref, cur = _frame(5000, seed=1), _frame(2000, seed=2)
    cal = cur.copy()
    for c in ("month_sin", "month_cos", "is_monsoon"):
        cal[c] = cal[c] + 3.0
    assert detect_drift(ref, cal)["retrain_recommended"] is False
    assert detect_drift(ref, _frame(2000, shift=1.0, seed=3))["retrain_recommended"] is True


def test_retraining_windows_are_purged_and_disjoint() -> None:
    dates = pd.bdate_range("2025-01-01", "2026-11-20")
    df = pd.DataFrame({"date": np.repeat(dates, 2)})
    w = _windows(df)
    assert not np.any(w["inner"] & w["compare"]) and not np.any(w["es"] & w["compare"])
    d = pd.DatetimeIndex(df["date"])
    gap = pd.bdate_range(d[w["es"]].max(), d[w["compare"]].min())
    assert len(gap) - 2 >= PURGE_DAYS  # at least PURGE_DAYS trading days strictly between them
    # rolling: the comparison window is the most recent COMPARE_DAYS trading days
    assert d[w["compare"]].nunique() == COMPARE_DAYS and d[w["compare"]].max() == dates.max()
    assert w["data_end"] == str(dates.max().date())


def test_gate_blocks_worse_model_and_passes_equal(tmp_path, monkeypatch) -> None:
    cand = {"best_model": "CatBoost", "models": {"CatBoost": {"test_2026": {"macro_f1": 0.39, "active_signal_precision": 0.28}}}}
    (tmp_path / "cand.json").write_text(json.dumps(cand))
    monkeypatch.setattr(gate, "CANDIDATE_PATH", tmp_path / "cand.json")
    monkeypatch.setattr(gate, "GATE_REPORT_PATH", tmp_path / "gate.json")

    (tmp_path / "bench.json").write_text(json.dumps({"macro_f1": 0.39, "active_signal_precision": 0.28}))
    monkeypatch.setattr(gate, "BENCHMARK_PATH", tmp_path / "bench.json")
    assert gate.evaluate()["passed"] is True

    (tmp_path / "bench.json").write_text(json.dumps({"macro_f1": 0.42, "active_signal_precision": 0.28}))
    assert gate.evaluate()["passed"] is False


def test_retraining_policy_cooldown_daily_weekly_and_force() -> None:
    drift_high = {"drift_share": 0.6, "retrain_recommended": True}
    drift_mid = {"drift_share": 0.3, "retrain_recommended": True}
    drift_none = {"drift_share": 0.05, "retrain_recommended": False}
    assert decide_retrain("weekly", drift_high, new_days=3)[0] is False       # cooldown: < 1 week of new data
    assert decide_retrain("weekly", drift_mid, new_days=5)[0] is True         # weekly + drift rule
    assert decide_retrain("weekly", drift_none, new_days=8)[0] is False       # no drift, not enough for refresh
    assert decide_retrain("weekly", drift_none, new_days=25)[0] is True       # routine monthly refresh
    assert decide_retrain("daily", drift_mid, new_days=10)[0] is False        # daily waits for the weekly run
    assert decide_retrain("daily", drift_high, new_days=10)[0] is True        # ...unless drift is urgent
    assert decide_retrain("daily", None, new_days=0, force=True)[0] is True   # manual override


def test_backup_bucket_from_mlflow_destination(monkeypatch) -> None:
    monkeypatch.delenv("BACKUP_BUCKET", raising=False)
    monkeypatch.setenv("MLFLOW_ARTIFACTS_DESTINATION", "s3://psx-bucket/mlflow")
    assert bucket_name() == "psx-bucket"
    monkeypatch.setenv("MLFLOW_ARTIFACTS_DESTINATION", "/app/mlflow_server/artifacts")
    assert bucket_name() is None
