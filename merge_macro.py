import time
import pandas as pd
import yfinance as yf
from pathlib import Path

STOCKS_FILE = Path("psx_historical_data/adjusted_stocks.csv")
OUTPUT_FILE = Path("psx_historical_data/stocks_with_macro.csv")

print("1. Loading adjusted_stocks.csv...")
df_stocks = pd.read_csv(STOCKS_FILE, parse_dates=["date"])

# Each macro variable has a primary ticker and fallback tickers in case Yahoo blocks/delists one
MACRO_SOURCES = {
    "brent_close": ["BZ=F", "BNO", "CL=F"],          # Brent Crude Futures -> Brent ETF -> WTI Crude
    "usdpkr_close": ["PKR=X"],                       # USD/PKR Spot Rate
    "dxy_close": ["DX=F", "UUP", "DX-Y.NYB"]         # Dollar Index Futures -> Dollar Bullish ETF -> Spot DXY
}

# Direct FRED (Federal Reserve) CSV fallbacks (zero blocks, no curl_cffi issues)
FRED_FALLBACKS = {
    "brent_close": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILBRENTEU",
    "dxy_close": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTWEXBGS"  # Nominal Broad US Dollar Index
}

def fetch_series(col_name, tickers):
    for ticker in tickers:
        for attempt in range(1, 4):
            try:
                print(f"   -> Fetching {col_name} via '{ticker}' (Attempt {attempt}/3)...")
                data = yf.download(
                    ticker,
                    start="2016-12-01",
                    end="2026-07-05",
                    progress=False,
                    timeout=60
                )
                if data is not None and not data.empty and "Close" in data:
                    s = data["Close"].squeeze().dropna()
                    if len(s) > 1500:  # Ensure we got the full ~10 year history
                        s.name = col_name
                        s.index = pd.to_datetime(s.index).tz_localize(None)
                        print(f"   [OK] Downloaded {len(s)} days for {col_name} using '{ticker}'")
                        return s
            except Exception as e:
                print(f"   [Warn] {ticker} attempt {attempt} failed: {e}")
            time.sleep(2)

    # If Yahoo Finance fails/times out on all tickers, fall back to FRED direct CSV
    if col_name in FRED_FALLBACKS:
        print(f"   -> Falling back to Federal Reserve (FRED) direct CSV for {col_name}...")
        fred_df = pd.read_csv(FRED_FALLBACKS[col_name])
        date_col = fred_df.columns[0]
        val_col = fred_df.columns[1]
        fred_df[date_col] = pd.to_datetime(fred_df[date_col])
        fred_df[val_col] = pd.to_numeric(fred_df[val_col], errors="coerce")
        fred_df = fred_df[(fred_df[date_col] >= "2016-12-01") & (fred_df[date_col] <= "2026-07-05")]
        s = fred_df.set_index(date_col)[val_col].dropna()
        s.name = col_name
        print(f"   [OK] Downloaded {len(s)} days for {col_name} from FRED!")
        return s

    raise RuntimeError(f"Could not fetch {col_name} from any source!")

print("2. Downloading 10-year Macro data...")
macro_dfs = []
for col_name, ticker_list in MACRO_SOURCES.items():
    s = fetch_series(col_name, ticker_list)
    macro_dfs.append(s)

# Build complete daily calendar from 2016-12-01 to 2026-07-05
full_dates = pd.date_range("2016-12-01", "2026-07-05", freq="D")
df_macro = pd.DataFrame(index=full_dates)
df_macro.index.name = "date"

for s in macro_dfs:
    # Handle any duplicate dates cleanly
    s = s[~s.index.duplicated(keep="last")]
    df_macro = df_macro.join(s)

# Forward-fill weekends/holidays strictly forward in time (bfill only for initial Dec 2016 start)
df_macro = df_macro.ffill().bfill().reset_index()

print("3. Merging macro data with PSX stocks...")
df_merged = pd.merge(df_stocks, df_macro, on="date", how="left")

df_merged.to_csv(OUTPUT_FILE, index=False)
print("\nDone! Saved to:", OUTPUT_FILE)
print("Total rows:", len(df_merged))
print("Missing macro values:", df_merged[["brent_close", "usdpkr_close", "dxy_close"]].isna().sum().to_dict())
print("\nSample merged row:")
print(df_merged[["date", "symbol", "close", "brent_close", "usdpkr_close", "dxy_close"]].tail(3).to_string(index=False))