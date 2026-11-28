
"""Phase 1: download daily OHLCV for the 30 stocks from the psxdata REST API (2017-01-01 -> 2026-06-30).

Run from the repo root:  python scripts/phase1/fetch_psx_data.py
Originally run against a local psxdata-api container (docker run -p 8000:8000 mtauha/psxdata-api);
set PSXDATA_API_URL to use that, otherwise the hosted API is used.
Pipeline order: fetch_psx_data.py -> audit_historical_data.py -> adjust_corporate_actions.py -> merge_macro.py
"""
import os
import requests
import pandas as pd
import json
import time
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

BASE_URL = os.environ.get("PSXDATA_API_URL", "https://psxdata-api.fastapicloud.dev")

START_DATE = "2017-01-01"
END_DATE = "2026-06-30"

STOCKS = [
    "OGDC", "PPL", "POL", "MARI",
    "HBL", "UBL", "MCB", "BAHL", "NBP", "BAFL", "AKBL",
    "FFC", "MTL", "HUBC", "KAPCO", "LUCK", "DGKC", "MLCF", "PIOC", "FCCL",
    "PSO", "SNGP", "SSGC", "ATRL", "NRL",
    "INDU", "PKGS", "PTC", "NML", "INIL"
]

OUTPUT_DIR = Path("psx_historical_data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

session = requests.Session()
session.headers["User-Agent"] = "psx-sentient/1.0 (+https://github.com/JunaidMu/PSX-Sentient)"

# ============================================================
# FETCH STOCK DATA
# ============================================================

def fetch_stock(symbol):

    url = f"{BASE_URL}/stocks/{symbol}/historical"

    params = {
        "start": START_DATE,
        "end": END_DATE
    }

    for attempt in range(3):

        try:
            response = session.get(
                url,
                params=params,
                timeout=60
            )

            response.raise_for_status()

            result = response.json()

            # Extract API response payload
            data = result.get("data", result)

            if not isinstance(data, list):
                raise ValueError(f"Unexpected API response: {data}")

            if not data:
                print(f"[EMPTY] {symbol}")
                return []

            # Save original API data as JSON
            json_path = OUTPUT_DIR / f"{symbol}.json"

            with open(json_path, "w", encoding="utf-8") as file:
                json.dump(data, file, indent=4)

            # Add stock symbol to every record
            for record in data:
                record["symbol"] = symbol

            print(f"[SUCCESS] {symbol}: {len(data)} records")

            return data

        except requests.exceptions.HTTPError as e:

            status = e.response.status_code

            if status not in (429, 502, 503):
                print(f"[ERROR] {symbol}: {e}")
                return []

            wait = 10 * (attempt + 1)

            print(f"[RETRY] {symbol}: HTTP {status}, waiting {wait}s")
            time.sleep(wait)

        except (requests.exceptions.RequestException, ValueError) as e:

            print(f"[ERROR] {symbol}, attempt {attempt + 1}: {e}")
            time.sleep(5)

    print(f"[FAILED] {symbol}: Maximum retries reached")

    return []


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():

    all_data = []
    failed_stocks = []
    empty_stocks = []

    print("=" * 60)
    print("PSX HISTORICAL DATA COLLECTION")
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Total stocks: {len(STOCKS)}")
    print("=" * 60)

    for index, symbol in enumerate(STOCKS, start=1):

        print(f"\n[{index}/{len(STOCKS)}] Fetching {symbol}...")

        records = fetch_stock(symbol)

        if records:
            all_data.extend(records)
        else:
            failed_stocks.append(symbol)

        time.sleep(1)

    # ========================================================
    # DATA PROCESSING
    # ========================================================

    if not all_data:
        print("\nNo data retrieved. Exiting.")
        return

    df = pd.DataFrame(all_data)

    required_columns = [
        "symbol",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "is_anomaly"
    ]

    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing API columns: {missing_columns}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    numeric_columns = ["open", "high", "low", "close", "volume"]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df[required_columns]

    # Sort chronologically
    df = df.sort_values(
        ["symbol", "date"]
    ).reset_index(drop=True)

    # ========================================================
    # SAVE COMBINED DATASET
    # ========================================================

    combined_path = OUTPUT_DIR / "all_stocks.csv"

    df.to_csv(combined_path, index=False)

    # ========================================================
    # SAVE INDIVIDUAL STOCK DATASETS
    # ========================================================

    for symbol, stock_df in df.groupby("symbol"):

        stock_path = OUTPUT_DIR / f"{symbol}.csv"

        stock_df.to_csv(stock_path, index=False)

    # ========================================================
    # DATA QUALITY REPORT
    # ========================================================

    duplicate_count = df.duplicated(
        subset=["symbol", "date"]
    ).sum()

    invalid_dates = df["date"].isna().sum()

    missing_values = df.isnull().sum()

    print("\n" + "=" * 60)
    print("DATA COLLECTION SUMMARY")
    print("=" * 60)

    print(f"Requested stocks: {len(STOCKS)}")
    print(f"Retrieved stocks: {df['symbol'].nunique()}")
    print(f"Failed/empty stocks: {len(failed_stocks)}")
    print(f"Total records: {len(df)}")

    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    print(f"Duplicate records: {duplicate_count}")
    print(f"Invalid dates: {invalid_dates}")

    print("\nRecords per stock:")

    print(
        df.groupby("symbol")
        .size()
        .reindex(STOCKS, fill_value=0)
        .to_string()
    )

    print("\nMissing values:")
    print(missing_values.to_string())

    if failed_stocks:
        print("\nFailed or empty stocks:")
        print(", ".join(failed_stocks))

    # Save collection report
    report = {
        "start_date": START_DATE,
        "end_date": END_DATE,
        "requested_stocks": len(STOCKS),
        "retrieved_stocks": df["symbol"].nunique(),
        "failed_or_empty_stocks": failed_stocks,
        "total_records": len(df),
        "duplicate_records": int(duplicate_count),
        "invalid_dates": int(invalid_dates),
        "missing_values": {
            key: int(value)
            for key, value in missing_values.items()
        }
    }

    with open(
        OUTPUT_DIR / "collection_report.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=4)

    print("\n" + "=" * 60)
    print("COLLECTION COMPLETED")
    print(f"Output: {OUTPUT_DIR.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
