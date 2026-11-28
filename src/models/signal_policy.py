"""Tune the final-signal policy (UP/DOWN vs NEUTRAL) on the walk-forward validation years.

The late-fusion rule calls UP/DOWN only when the blended conviction |C_t| = |0.75 * E_tech + 0.25 * S_t|
exceeds a threshold (and P(NEUTRAL) < 0.50). The proposal's threshold of 0.20 was never tuned: with
this model's probability scale it fires on <1% of validation stock-days, so nearly every live signal
is NEUTRAL. Here the threshold is chosen on the 2023-2025 out-of-fold predictions only (S_t = 0, since
no historical news exists): the most precise threshold whose active-signal rate is at least
MIN_ACTIVE_RATE. Test_2026 is reported, never used for the choice.

Output: models/signal_policy.json (read by the serving API).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.train_baseline import MODELS_DIR, compute_classification_metrics

POLICY_PATH = MODELS_DIR / "signal_policy.json"
OOF_PATH = MODELS_DIR / "val_oof_predictions.parquet"
TEST_PATH = MODELS_DIR / "test_2026_predictions.parquet"
MIN_ACTIVE_RATE = 0.15  # call UP/DOWN on at least ~15% of stock-days
NEUTRAL_CAP = 0.50
GRID = np.round(np.arange(0.02, 0.26, 0.005), 3)
DEFAULT_POLICY = {"conviction_threshold": 0.20, "neutral_cap": NEUTRAL_CAP, "source": "proposal default (untuned)"}


def _signals(df: pd.DataFrame, thr: float) -> np.ndarray:
    c_t = 0.75 * (df["P_up"] - df["P_down"])
    ok = df["P_neutral"] < NEUTRAL_CAP
    return np.where((c_t > thr) & ok, 2, np.where((c_t < -thr) & ok, 0, 1))


def tune(best_model: str | None = None) -> dict[str, Any]:
    oof = pd.read_parquet(OOF_PATH)
    test = pd.read_parquet(TEST_PATH)
    model = best_model or test["model_name"].iloc[0]
    oof = oof[oof["model_name"] == model]

    rows = []
    for thr in GRID:
        m = compute_classification_metrics(oof["target_class"], _signals(oof, thr))
        rows.append({"threshold": float(thr), "rate": m["active_signal_rate"], "precision": m["active_signal_precision"]})
    grid = pd.DataFrame(rows)
    eligible = grid[grid["rate"] >= MIN_ACTIVE_RATE]
    best = eligible.sort_values(["precision", "threshold"], ascending=False).iloc[0]
    thr = float(best["threshold"])

    test_m = compute_classification_metrics(test["target_class"], _signals(test, thr))
    default_m = compute_classification_metrics(test["target_class"], _signals(test, 0.20))
    argmax_m = compute_classification_metrics(test["target_class"], test["pred_class"])
    policy = {
        "conviction_threshold": thr,
        "neutral_cap": NEUTRAL_CAP,
        "source": f"tuned on {model} walk-forward validation 2023-2025 (min active rate {MIN_ACTIVE_RATE:.0%})",
        "validation": {"active_rate": round(float(best["rate"]), 4), "active_precision": round(float(best["precision"]), 4)},
        "test_2026": {"active_rate": round(test_m["active_signal_rate"], 4),
                      "active_precision": round(test_m["active_signal_precision"], 4)},
        "test_2026_proposal_default_0.20": {"active_rate": round(default_m["active_signal_rate"], 4),
                                            "active_precision": round(default_m["active_signal_precision"], 4)},
        "test_2026_model_argmax": {"active_rate": round(argmax_m["active_signal_rate"], 4),
                                   "active_precision": round(argmax_m["active_signal_precision"], 4)},
    }
    POLICY_PATH.write_text(json.dumps(policy, indent=2), encoding="utf-8")
    return policy


def load_policy(path: Path = POLICY_PATH) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return dict(DEFAULT_POLICY)


if __name__ == "__main__":
    print(json.dumps(tune(), indent=2))
