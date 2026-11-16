import pandas as pd
import numpy as np
from pathlib import Path

INPUT = Path("psx_historical_data/all_stocks.csv")
OUTPUT = Path("psx_historical_data/adjusted_stocks.csv")

df = pd.read_csv(INPUT, parse_dates=["date"])

# 1. Guarantee strict chronological ordering and drop exact duplicates
df = df.sort_values(["symbol", "date"]).drop_duplicates(subset=["symbol", "date"]).reset_index(drop=True)

# 2. Preserve original raw values
for col in ["open", "high", "low", "close", "volume"]:
    df[f"original_{col}"] = df[col]

# 3. Fix zero/non-positive Open/High/Low glitches (e.g., PKGS 2019-10-21 & 2021-08-20)
for col in ["open", "high", "low"]:
    bad_price_mask = (df[col] <= 0) | (df[col].isna())
    df.loc[bad_price_mask, col] = df.loc[bad_price_mask, "close"]

# Ensure High >= max(Open, Close) and Low <= min(Open, Close)
df["high"] = df[["open", "high", "close"]].max(axis=1)
df["low"] = df[["open", "low", "close"]].min(axis=1)

# 4. Known Major Splits & Bonus Share Issues (Stock, ex-date, ratio)
events = [
    # Major Splits & Recent Bonuses
    ("MARI", "2024-09-16", 9.0),
    ("LUCK", "2025-04-28", 5.0),
    ("BAFL", "2026-04-20", 2.0),
    ("UBL",  "2025-06-23", 2.0),
    ("MTL",  "2026-06-22", 2.0),
    # Historical Bonus Issues (15% to 50%)
    ("ATRL", "2018-09-17", 1.20),
    ("POL",  "2018-09-17", 1.20),
    ("PSO",  "2017-10-11", 1.20),
    ("PSO",  "2018-10-05", 1.20),
    ("PSO",  "2019-10-16", 1.20),
    ("PPL",  "2018-10-12", 1.15),
    ("PPL",  "2019-10-16", 1.20),
    ("MTL",  "2019-10-17", 1.15),
    ("MTL",  "2021-03-10", 1.15),
    ("MTL",  "2021-10-21", 1.20),
    ("MTL",  "2022-03-07", 1.20),
    ("MTL",  "2022-05-13", 1.20),
    ("MTL",  "2022-10-19", 1.20),
    ("MTL",  "2023-06-22", 1.50),
    ("FCCL", "2022-10-13", 1.125),
]

df["adjustment_factor"] = 1.0

for symbol, date, ratio in events:
    ex_date = pd.Timestamp(date)
    mask = (df["symbol"] == symbol) & (df["date"] < ex_date)
    df.loc[mask, "adjustment_factor"] /= ratio

# Apply split/bonus adjustments
for col in ["open", "high", "low", "close"]:
    df[col] = (df[col] * df["adjustment_factor"]).round(5)

df["volume"] = (df["volume"] / df["adjustment_factor"]).round(0).astype("int64")

# 5. Detect & Flag Isolated Ex-Dividend Overnight Drops (KAPCO, HUBC, NBP, BAHL, etc.)
prev_close = df.groupby("symbol")["close"].shift(1)
df["overnight_gap_pct"] = (df["open"] / prev_close - 1.0) * 100

# Count how many stocks gapped down > 8.5% on the exact same date
# If >= 4 stocks gapped down together (e.g., 2026-03-02, 2026-03-09), it's a real market crash!
# If only 1-3 stocks gapped down > 10.2% overnight, it's an isolated Ex-Dividend / Bonus adjustment.
daily_crash_counts = df[df["overnight_gap_pct"] < -8.5].groupby("date")["symbol"].count()
market_crash_dates = set(daily_crash_counts[daily_crash_counts >= 4].index)

df["is_ex_dividend_gap"] = (
    (df["overnight_gap_pct"] < -10.2) & 
    (~df["date"].isin(market_crash_dates))
).astype(int)

# Save clean adjusted dataset
df.to_csv(OUTPUT, index=False)

print("Adjusted dataset saved:", OUTPUT)
print("Total records:", len(df))
print("Fixed zero-open/high/low glitches:", bad_price_mask.sum())
print("Isolated Ex-Dividend gap days flagged (to exclude from target Y):", df["is_ex_dividend_gap"].sum())