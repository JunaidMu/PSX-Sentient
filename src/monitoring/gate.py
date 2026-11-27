"""Model-evaluation gate for CI/CD.

Compares the model just trained by the pipeline (models/model_comparison_metrics.json, best model's
Test_2026 metrics) with the production benchmark (models/production_benchmark.json) on the fixed
Test_2026 holdout (Jan-Jun 2026, never used for selection). Exits non-zero - failing the CI job and
blocking deployment - when the candidate is worse beyond the tolerances.

    python -m src.monitoring.gate             # check
    python -m src.monitoring.gate --update    # adopt the current model's metrics as the new benchmark
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

CANDIDATE_PATH = Path("models/model_comparison_metrics.json")
BENCHMARK_PATH = Path("models/production_benchmark.json")
GATE_REPORT_PATH = Path("reports/gate_result.json")
TOLERANCE = {"macro_f1": 0.005, "active_signal_precision": 0.01}


def candidate_metrics() -> dict:
    summary = json.loads(CANDIDATE_PATH.read_text(encoding="utf-8"))
    best = summary["best_model"]
    return {"model": best, **{k: summary["models"][best]["test_2026"][k] for k in TOLERANCE}}


def evaluate() -> dict:
    cand = candidate_metrics()
    bench = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    checks = {}
    for metric, tol in TOLERANCE.items():
        floor = bench[metric] - tol
        checks[metric] = {"candidate": round(cand[metric], 4), "benchmark": bench[metric], "floor": round(floor, 4),
                          "passed": cand[metric] >= floor}
    result = {"test_set": "Test_2026 (2026-01-01 -> 2026-06-29)", "candidate_model": cand["model"],
              "checks": checks, "passed": all(c["passed"] for c in checks.values())}
    GATE_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    GATE_REPORT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true")
    args = ap.parse_args()
    if args.update:
        cand = candidate_metrics()
        BENCHMARK_PATH.write_text(json.dumps({**{k: round(v, 4) for k, v in cand.items() if k != "model"},
                                              "model": cand["model"], "test_set": "Test_2026"}, indent=2), encoding="utf-8")
        print(f"Benchmark updated: {BENCHMARK_PATH.read_text()}")
        sys.exit(0)
    res = evaluate()
    for metric, c in res["checks"].items():
        print(f"[{'PASS' if c['passed'] else 'FAIL'}] {metric}: candidate {c['candidate']} vs benchmark {c['benchmark']} (floor {c['floor']})")
    if not res["passed"]:
        print("Model evaluation gate FAILED: candidate is worse than production on the fixed test set. Deployment blocked.")
        sys.exit(1)
    print("Model evaluation gate passed.")
