"""Automated schema & quality checks that run before training.

Two gates in the DVC pipeline:
- `validate_raw`      checks `stocks_with_macro.csv` before cleaning / feature building.
- `validate_features` checks `feature_store.parquet` before any model is trained.

Each gate writes a JSON report to `reports/` and exits with a non-zero status when any
check fails, so `dvc repro` stops before corrupted or missing inputs reach training.

Usage:
    python -m src.data.validate raw
    python -m src.data.validate features
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from src.features.build_features import FEATURE_COLUMNS, OUTPUT_PARQUET_PATH
from src.nlp.entity_mapper import KSE30_SYMBOLS

RAW_CSV_PATH = Path("psx_historical_data/stocks_with_macro.csv")
REPORTS_DIR = Path("reports")

RAW_SCHEMA: dict[str, str] = {
    "date": "datetime",
    "symbol": "string",
    "open": "number",
    "high": "number",
    "low": "number",
    "close": "number",
    "volume": "number",
    "adjustment_factor": "number",
    "is_ex_dividend_gap": "number",
    "brent_close": "number",
    "usdpkr_close": "number",
    "dxy_close": "number",
}

# Plausible ranges for macro series (anything outside means a broken download)
MACRO_RANGES: dict[str, tuple[float, float]] = {
    "brent_close": (5.0, 250.0),
    "usdpkr_close": (80.0, 500.0),
    "dxy_close": (70.0, 160.0),
}

MIN_ROWS_PER_SYMBOL = 1500  # ~6 years of trading days
MAX_MISSING_DAYS_PER_SYMBOL = 30
MAX_ABS_DAILY_MOVE = 0.60  # a 60% one-day move after split adjustment means a missed split / bad tick


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def _check(results: list[CheckResult], name: str, passed: bool, detail: str = "") -> None:
    results.append(CheckResult(name=name, passed=bool(passed), detail=detail))


def validate_raw_frame(df: pd.DataFrame) -> list[CheckResult]:
    """Run schema & quality checks on the raw merged stock + macro table."""
    results: list[CheckResult] = []

    missing = [c for c in RAW_SCHEMA if c not in df.columns]
    _check(results, "schema.required_columns", not missing, f"missing={missing}")
    if missing:
        return results  # Remaining checks need these columns

    dates = pd.to_datetime(df["date"], errors="coerce")
    _check(results, "schema.date_parseable", dates.notna().all(), f"unparseable={int(dates.isna().sum())}")

    numeric_cols = [c for c, t in RAW_SCHEMA.items() if t == "number"]
    non_numeric = [c for c in numeric_cols if not pd.api.types.is_numeric_dtype(df[c])]
    _check(results, "schema.numeric_dtypes", not non_numeric, f"non_numeric={non_numeric}")
    if non_numeric:
        return results

    na = df[list(RAW_SCHEMA)].isna().sum()
    _check(results, "quality.no_missing_values", int(na.sum()) == 0, f"nulls={na[na > 0].to_dict()}")

    dupes = int(df.duplicated(["symbol", "date"]).sum())
    _check(results, "quality.unique_symbol_date", dupes == 0, f"duplicates={dupes}")

    symbols = set(df["symbol"].unique())
    expected = set(KSE30_SYMBOLS)
    _check(
        results,
        "quality.expected_symbols",
        symbols == expected,
        f"missing={sorted(expected - symbols)} unexpected={sorted(symbols - expected)}",
    )

    prices = df[["open", "high", "low", "close"]]
    _check(results, "quality.positive_prices", bool((prices > 0).all().all()), f"non_positive={int((prices <= 0).sum().sum())}")
    ohlc_ok = (df["high"] >= df[["open", "close", "low"]].max(axis=1) - 1e-9) & (
        df["low"] <= df[["open", "close", "high"]].min(axis=1) + 1e-9
    )
    _check(results, "quality.ohlc_consistent", bool(ohlc_ok.all()), f"violations={int((~ohlc_ok).sum())}")
    _check(results, "quality.non_negative_volume", bool((df["volume"] >= 0).all()), f"negative={int((df['volume'] < 0).sum())}")

    flags_ok = df["is_ex_dividend_gap"].isin([0, 1]).all()
    _check(results, "quality.binary_ex_div_flag", bool(flags_ok), "")

    for col, (lo, hi) in MACRO_RANGES.items():
        bad = int(((df[col] < lo) | (df[col] > hi)).sum())
        _check(results, f"quality.{col}_in_range", bad == 0, f"range=[{lo},{hi}] out_of_range={bad}")

    tmp = df.assign(date=dates).sort_values(["symbol", "date"])
    future = int((tmp["date"] > pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).sum())
    _check(results, "quality.no_future_dates", future == 0, f"future_rows={future}")

    weekend = int(tmp["date"].dt.day_of_week.isin([5, 6]).sum())
    _check(results, "quality.no_weekend_dates", weekend == 0, f"weekend_rows={weekend}")

    counts = tmp.groupby("symbol").size()
    short = counts[counts < MIN_ROWS_PER_SYMBOL]
    _check(results, "quality.min_history_per_symbol", short.empty, f"short={short.to_dict()}")

    calendar = set(tmp["date"].unique())
    missing_days = {s: len(calendar - set(g["date"])) for s, g in tmp.groupby("symbol")}
    too_many = {s: n for s, n in missing_days.items() if n > MAX_MISSING_DAYS_PER_SYMBOL}
    _check(results, "quality.max_missing_days_per_symbol", not too_many, f"over_limit={too_many}")

    daily_move = (tmp["close"] / tmp.groupby("symbol")["close"].shift(1) - 1.0).abs()
    huge = int((daily_move > MAX_ABS_DAILY_MOVE).sum())
    _check(results, "quality.no_unadjusted_splits", huge == 0, f"moves_over_{int(MAX_ABS_DAILY_MOVE*100)}pct={huge}")

    return results


def validate_feature_frame(df: pd.DataFrame) -> list[CheckResult]:
    """Run schema & quality checks on the feature store before training."""
    results: list[CheckResult] = []
    required = FEATURE_COLUMNS + ["date", "symbol", "next_ret_1d", "B_t", "target_class"]
    missing = [c for c in required if c not in df.columns]
    _check(results, "schema.required_columns", not missing, f"missing={missing}")
    if missing:
        return results

    values = df[FEATURE_COLUMNS + ["next_ret_1d", "B_t"]].to_numpy(dtype="float64")
    _check(results, "quality.no_nan", not np.isnan(values).any(), f"nan_cells={int(np.isnan(values).sum())}")
    _check(results, "quality.no_inf", not np.isinf(values).any(), f"inf_cells={int(np.isinf(values).sum())}")

    labels = set(pd.unique(df["target_class"]))
    _check(results, "quality.labels_valid", labels == {0, 1, 2}, f"labels={sorted(labels)}")

    props = df["target_class"].value_counts(normalize=True)
    balanced = all(0.10 <= float(props.get(c, 0.0)) <= 0.70 for c in (0, 1, 2))
    _check(results, "quality.class_balance", balanced, f"proportions={props.round(4).to_dict()}")

    rsi_ok = df["rsi_14"].between(0.0, 100.0).all()
    _check(results, "quality.rsi_bounds", bool(rsi_ok), "")
    _check(results, "quality.positive_volatility", bool((df["vol_20d"] >= 0).all()), "")

    dupes = int(df.duplicated(["symbol", "date"]).sum())
    _check(results, "quality.unique_symbol_date", dupes == 0, f"duplicates={dupes}")
    _check(results, "quality.symbol_count", df["symbol"].nunique() == len(KSE30_SYMBOLS), f"symbols={df['symbol'].nunique()}")
    return results


def _write_report(stage: str, source: Path, results: list[CheckResult]) -> dict:
    report = {
        "stage": stage,
        "source": str(source),
        "passed": all(r.passed for r in results),
        "checks_total": len(results),
        "checks_failed": sum(not r.passed for r in results),
        "checks": [asdict(r) for r in results],
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / f"validation_{stage}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run(stage: str) -> dict:
    if stage == "raw":
        source = RAW_CSV_PATH
        if not source.exists():
            results = [CheckResult("input.exists", False, f"{source} not found (run `dvc pull`)")]
        else:
            results = validate_raw_frame(pd.read_csv(source))
    elif stage == "features":
        source = OUTPUT_PARQUET_PATH
        if not source.exists():
            results = [CheckResult("input.exists", False, f"{source} not found")]
        else:
            results = validate_feature_frame(pd.read_parquet(source))
    else:
        raise SystemExit(f"Unknown stage '{stage}'. Use 'raw' or 'features'.")
    return _write_report(stage, source, results)


if __name__ == "__main__":
    stage_arg = sys.argv[1] if len(sys.argv) > 1 else "raw"
    rep = run(stage_arg)
    for chk in rep["checks"]:
        mark = "PASS" if chk["passed"] else "FAIL"
        print(f"[{mark}] {chk['name']} {chk['detail']}")
    print(f"\n{rep['checks_total'] - rep['checks_failed']}/{rep['checks_total']} checks passed -> reports/validation_{stage_arg}.json")
    if not rep["passed"]:
        sys.exit(1)
