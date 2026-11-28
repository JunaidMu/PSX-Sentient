import pandas as pd
import numpy as np

df = pd.read_csv("psx_historical_data/adjusted_stocks.csv", parse_dates=["date"])
df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

# Calculate simple and log returns per stock
df["pct_return"] = df.groupby("symbol")["close"].pct_change() * 100
df["overnight_gap"] = (df["open"] / df.groupby("symbol")["close"].shift(1) - 1) * 100

# PSX normal daily circuit breaker is +/- 7.5% (occasionally ~8-10% during 2020 COVID changes).
# Any overnight open gap worse than -8.5% is almost certainly an unadjusted bonus/dividend/split!
suspect_events = df[
    (df["pct_return"] < -8.5) | (df["pct_return"] > 12.0) | (df["overnight_gap"] < -8.0)
][["symbol", "date", "close", "pct_return", "overnight_gap", "adjustment_factor"]]

print(f"Found {len(suspect_events)} potential unadjusted ex-bonus/ex-dividend dates:")
print(suspect_events.sort_values("pct_return").to_string(index=False))