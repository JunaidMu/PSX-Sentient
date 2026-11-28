
import pandas as pd
from pathlib import Path

DATA_PATH = Path("psx_historical_data/all_stocks.csv")
OUTPUT_DIR = Path("psx_historical_data/audit")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH, parse_dates=["date"])

# Ensure numeric columns
numeric = ["open", "high", "low", "close", "volume"]
df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")

# --------------------------------------------------
# 1. MARKET DATE COVERAGE
# --------------------------------------------------

market_dates = set(df["date"].dropna().unique())

coverage = []

for symbol, stock in df.groupby("symbol"):
    stock_dates = set(stock["date"].dropna().unique())

    missing_dates = sorted(market_dates - stock_dates)

    coverage.append({
        "symbol": symbol,
        "records": len(stock),
        "missing_market_dates": len(missing_dates),
        "missing_dates_sample": ", ".join(
            str(pd.Timestamp(d).date())
            for d in missing_dates[:10]
        )
    })

coverage_df = pd.DataFrame(coverage)

# --------------------------------------------------
# 2. ZERO / NEGATIVE VOLUME
# --------------------------------------------------

zero_volume = df[df["volume"] == 0]
negative_volume = df[df["volume"] < 0]

# --------------------------------------------------
# 3. NON-POSITIVE PRICES
# --------------------------------------------------

nonpositive_prices = df[
    (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
]

# --------------------------------------------------
# 4. EXTREME DAILY PRICE MOVEMENTS
# --------------------------------------------------

df = df.sort_values(["symbol", "date"])

df["daily_return"] = (
    df.groupby("symbol")["close"].pct_change()
)

# Diagnostic threshold only; not an automatic rejection rule.
extreme_moves = df[
    df["daily_return"].abs() > 0.20
].copy()

# --------------------------------------------------
# 5. SAVE AUDIT RESULTS
# --------------------------------------------------

coverage_df.to_csv(
    OUTPUT_DIR / "trading_date_coverage.csv",
    index=False
)

zero_volume.to_csv(
    OUTPUT_DIR / "zero_volume_records.csv",
    index=False
)

negative_volume.to_csv(
    OUTPUT_DIR / "negative_volume_records.csv",
    index=False
)

nonpositive_prices.to_csv(
    OUTPUT_DIR / "nonpositive_price_records.csv",
    index=False
)

extreme_moves.to_csv(
    OUTPUT_DIR / "extreme_price_movements.csv",
    index=False
)

# --------------------------------------------------
# 6. SUMMARY
# --------------------------------------------------

print("=" * 65)
print("PSX HISTORICAL DATA AUDIT")
print("=" * 65)

print("Total records:", len(df))
print("Stocks:", df["symbol"].nunique())
print("Unique market dates:", len(market_dates))

print("\n--- TRADING DATE COVERAGE ---")
print(
    coverage_df[
        ["symbol", "records", "missing_market_dates"]
    ].to_string(index=False)
)

print("\n--- SUSPICIOUS RECORDS ---")
print("Zero volume:", len(zero_volume))
print("Negative volume:", len(negative_volume))
print("Non-positive prices:", len(nonpositive_prices))
print("Daily returns above 20% absolute:", len(extreme_moves))

print("\n--- EXTREME PRICE MOVEMENTS ---")

if not extreme_moves.empty:
    print(
        extreme_moves[
            ["symbol", "date", "close", "daily_return"]
        ].to_string(index=False)
    )
else:
    print("None detected.")

print("\nAudit files saved to:", OUTPUT_DIR.resolve())
print("=" * 65)
