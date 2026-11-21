"""Phase 1.5: Data cleaning on top of `stocks_with_macro.csv`.

Reads `psx_historical_data/stocks_with_macro.csv` (never modified) and writes
`psx_historical_data/stocks_clean.csv` with two fixes:

1. USD/PKR bad-tick repair
   Yahoo Finance's USD/PKR series contains one-day spikes that fully reverse the next day,
   e.g. 239.9 -> 206.75 -> 239.1 (2022-08-01). A value is treated as a bad tick when it
   jumps away from the previous value by more than `threshold` and the next value returns
   to within `threshold / 2` of the pre-spike value. The bad value is replaced with the
   last good value (forward fill), so no future price is copied into the past.
   Note: spotting a spike needs the next day's value, so in live use a spike can only be
   repaired one day late.

2. Ex-dividend gap detection (heuristic, no dividend history available)
   The original flag only caught overnight drops below -10.2%. Most PSX cash dividends
   are 2-8% of price, so ordinary ex-dividend days were labelled as DOWN moves. A day is
   now flagged when the stock's overnight gap is at least `idio_gap_threshold` below
   the same-day median gap of all 30 stocks (i.e. the stock fell on its own, not with
   the market). The original flags are kept as well.
   Limitation: a genuine stock-specific bad-news gap of the same size is also flagged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_INPUT_PATH = Path("psx_historical_data/stocks_with_macro.csv")
CLEAN_OUTPUT_PATH = Path("psx_historical_data/stocks_clean.csv")

MACRO_COLUMNS: tuple[str, ...] = ("brent_close", "usdpkr_close", "dxy_close")

# Minimum one-day log move that counts as a possible bad tick, per series.
# Only USD/PKR is repaired: Brent genuinely whipsaws >8% and reverses in crises
# (Mar 2020, Mar-Apr 2026), so "repairing" it would delete real moves. DXY has no spikes.
# Only 1-day spikes are repaired; 2-day reversals include real events (e.g. Jul 2017 PKR).
MACRO_SPIKE_THRESHOLDS: dict[str, float] = {
    "usdpkr_close": 0.025,
}
MACRO_MAX_SPIKE_LEN = 1

IDIO_GAP_THRESHOLD = -0.03


def repair_macro_spikes(
    series: pd.Series,
    threshold: float,
    max_spike_len: int = 2,
) -> tuple[pd.Series, list[pd.Timestamp]]:
    """Replace short spikes that fully reverse with the last good value.

    `series` must be indexed by date and sorted ascending (one value per date).
    Returns the repaired series and the list of dates that were replaced.
    """
    values = series.to_numpy(dtype="float64").copy()
    log_v = np.log(values)
    n = len(values)
    repaired: list[int] = []

    i = 1
    while i < n:
        jump = log_v[i] - log_v[i - 1]
        if abs(jump) <= threshold:
            i += 1
            continue
        # Look for a return to the pre-spike level within `max_spike_len` days
        restored_at = None
        for k in range(1, max_spike_len + 1):
            j = i + k
            if j >= n:
                break
            if abs(log_v[j] - log_v[i - 1]) <= threshold / 2:
                restored_at = j
                break
        if restored_at is None:
            i += 1  # A persistent move (e.g. a real devaluation): keep it
            continue
        for idx in range(i, restored_at):
            values[idx] = values[i - 1]
            log_v[idx] = log_v[i - 1]
            repaired.append(idx)
        i = restored_at + 1

    out = pd.Series(values, index=series.index, name=series.name)
    return out, [series.index[k] for k in repaired]


def clean_macro_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[pd.Timestamp]]]:
    """Repair macro bad ticks on the unique-date calendar, then map back onto every row."""
    out = df.copy()
    report: dict[str, list[pd.Timestamp]] = {}
    calendar = (
        out[["date", *MACRO_COLUMNS]]
        .drop_duplicates("date")
        .sort_values("date")
        .set_index("date")
    )
    for col, threshold in MACRO_SPIKE_THRESHOLDS.items():
        fixed, dates = repair_macro_spikes(calendar[col], threshold, max_spike_len=MACRO_MAX_SPIKE_LEN)
        report[col] = dates
        out[col] = out["date"].map(fixed)
    return out, report


def flag_ex_dividend_gaps(
    df: pd.DataFrame,
    idio_gap_threshold: float = IDIO_GAP_THRESHOLD,
) -> pd.DataFrame:
    """Flag stock-specific overnight gap-downs as likely ex-dividend days."""
    out = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    prev_close = out.groupby("symbol", sort=False)["close"].shift(1)
    gap = out["open"] / prev_close - 1.0
    market_gap = gap.groupby(out["date"]).transform("median")
    idio_gap = gap - market_gap

    new_flag = (idio_gap < idio_gap_threshold) & (gap < idio_gap_threshold)
    out["is_ex_dividend_gap_original"] = out["is_ex_dividend_gap"].astype("int64")
    out["is_ex_dividend_gap"] = (
        (out["is_ex_dividend_gap_original"] == 1) | new_flag.fillna(False)
    ).astype("int64")
    return out


def build_clean_dataset(
    input_csv: Path = RAW_INPUT_PATH,
    output_csv: Path = CLEAN_OUTPUT_PATH,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Run all cleaning steps and save `stocks_clean.csv`. Returns (clean_df, report)."""
    df = pd.read_csv(input_csv, parse_dates=["date"])
    df, macro_report = clean_macro_columns(df)
    df = flag_ex_dividend_gaps(df)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    report: dict[str, object] = {
        "macro_repaired_dates": macro_report,
        "ex_div_flags_original": int(df["is_ex_dividend_gap_original"].sum()),
        "ex_div_flags_total": int(df["is_ex_dividend_gap"].sum()),
    }
    return df, report


if __name__ == "__main__":
    clean_df, rep = build_clean_dataset()
    print(f"Saved cleaned dataset to: {CLEAN_OUTPUT_PATH} ({len(clean_df):,} rows)")
    for col, dates in rep["macro_repaired_dates"].items():  # type: ignore[union-attr]
        shown = ", ".join(str(d.date()) for d in dates)
        print(f"  {col}: {len(dates)} bad ticks repaired {('-> ' + shown) if shown else ''}")
    print(f"  ex-dividend flags: {rep['ex_div_flags_original']} original -> {rep['ex_div_flags_total']} total")
    by_month = (
        clean_df.loc[clean_df["is_ex_dividend_gap"] == 1, "date"].dt.month.value_counts().sort_index()
    )
    print("  flags by month:", by_month.to_dict())
