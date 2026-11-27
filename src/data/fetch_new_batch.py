"""Fetch the post-cutoff "new batch" (2026-07-01 -> today) in the same schema as the history.

Sources
- Prices: psxdata REST API (same source as the 2017-2026 history), all 30 symbols.
- Macro:  Brent / USD-PKR / dollar index from Yahoo Finance or FRED. For each series every candidate
          source is compared with the values already in `stocks_with_macro.csv` over the overlap
          window (June 2026); the source that reproduces the history is used, so the new batch is
          on the same scale (e.g. the history's `dxy_close` is FRED's broad dollar index, ~118, not DXY).

Output: `psx_historical_data/new_batch/stocks_with_macro_new.csv` with the exact columns of
`stocks_with_macro.csv`, only rows dated after the historical cutoff.

Usage: python -m src.data.fetch_new_batch [--end YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
from io import StringIO
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests

from src.nlp.entity_mapper import KSE30_SYMBOLS

HISTORY_PATH = Path("psx_historical_data/stocks_with_macro.csv")
NEW_BATCH_DIR = Path("psx_historical_data/new_batch")
NEW_BATCH_PATH = NEW_BATCH_DIR / "stocks_with_macro_new.csv"
REPORT_PATH = Path("reports/new_batch_fetch.json")

PSXDATA_API_URL = os.environ.get("PSXDATA_API_URL", "https://psxdata-api.fastapicloud.dev")
HTTP_HEADERS = {"User-Agent": "psx-sentient/1.0 (+https://github.com/JunaidMu/PSX-Sentient)"}
OVERLAP_START = "2026-06-01"

MACRO_CANDIDATES: dict[str, list[tuple[str, str]]] = {
    "brent_close": [("yahoo", "BZ=F"), ("fred", "DCOILBRENTEU"), ("yahoo", "BNO")],
    "usdpkr_close": [("yahoo", "PKR=X")],
    "dxy_close": [("fred", "DTWEXBGS"), ("yahoo", "DX-Y.NYB"), ("yahoo", "DX=F")],
}
MAX_OVERLAP_MISMATCH = 0.01  # mean absolute relative difference allowed vs. history
MIN_SYMBOLS_PER_SESSION = 27  # a complete PSX session has data for (almost) all 30 symbols
MIN_VOLUME_RATIO = 0.2  # median volume below 20% of the trailing norm = partial/intraday snapshot


RAW_CACHE_DIR = NEW_BATCH_DIR / "raw"


def fetch_symbol_prices(symbol: str, start: str, end: str, session: requests.Session) -> pd.DataFrame:
    """Fetch one symbol, retrying rate limits, 5xx errors and timeouts; cache the raw JSON per run."""
    cache = RAW_CACHE_DIR / f"{symbol}_{start}_{end}.json"
    if cache.exists():
        rows = json.loads(cache.read_text(encoding="utf-8"))
    else:
        url = f"{PSXDATA_API_URL}/stocks/{symbol}/historical"
        rows = None
        for attempt in range(6):
            try:
                resp = session.get(url, params={"start": start, "end": end}, headers=HTTP_HEADERS, timeout=90)
            except requests.RequestException:
                time.sleep(5 * (attempt + 1))
                continue
            if resp.status_code == 429 or resp.status_code >= 500:  # rate limit or transient upstream error
                time.sleep(int(resp.headers.get("Retry-After", 5)) * (attempt + 1))
                continue
            resp.raise_for_status()
            rows = resp.json().get("data", [])
            break
        if rows is None:
            raise RuntimeError(f"psxdata API failed repeatedly for {symbol}")
        RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(rows), encoding="utf-8")
        time.sleep(1.1)  # stay under the API's 60 requests/minute limit
    df = pd.DataFrame(rows)
    df["symbol"] = symbol
    return df


def fetch_all_prices(start: str, end: str) -> pd.DataFrame:
    session = requests.Session()
    frames = []
    for sym in KSE30_SYMBOLS:
        frames.append(fetch_symbol_prices(sym, start, end, session))
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df[["symbol", "date", "open", "high", "low", "close", "volume", "is_anomaly"]]


def _fetch_macro_series(kind: str, ticker: str, start: str, end: str) -> pd.Series:
    if kind == "fred":
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={ticker}"
        raw = pd.read_csv(StringIO(requests.get(url, headers=HTTP_HEADERS, timeout=60).text))
        raw.columns = ["date", "value"]
        s = pd.Series(pd.to_numeric(raw["value"], errors="coerce").to_numpy(), index=pd.to_datetime(raw["date"]))
    else:
        import yfinance as yf

        # Ticker.history is more reliable than yf.download on throttled networks
        end_excl = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        data = yf.Ticker(ticker).history(start=start, end=end_excl, auto_adjust=False, timeout=60)
        s = data["Close"]
        s.index = pd.to_datetime(s.index.date)
    s = s.dropna()
    return s[(s.index >= start) & (s.index <= end)].astype("float64")


def choose_macro_sources(history: pd.DataFrame, end: str) -> tuple[dict[str, pd.Series], dict[str, dict]]:
    """For each macro column pick the candidate source that reproduces the history on the overlap."""
    hist_cal = history.drop_duplicates("date").set_index("date").sort_index()
    overlap = hist_cal.loc[OVERLAP_START:]
    chosen: dict[str, pd.Series] = {}
    report: dict[str, dict] = {}
    for col, candidates in MACRO_CANDIDATES.items():
        tried = {}
        for kind, ticker in candidates:
            try:
                s = _fetch_macro_series(kind, ticker, "2026-05-01", end)
            except Exception as exc:
                tried[f"{kind}:{ticker}"] = f"error: {str(exc)[:80]}"
                continue
            aligned = s.reindex(pd.date_range("2026-05-01", end, freq="D")).ffill().reindex(overlap.index)
            mismatch = float((aligned / overlap[col] - 1.0).abs().mean())
            tried[f"{kind}:{ticker}"] = round(mismatch, 5)
            if np.isfinite(mismatch) and mismatch <= MAX_OVERLAP_MISMATCH and col not in chosen:
                chosen[col] = s
                report[col] = {"source": f"{kind}:{ticker}", "overlap_mismatch": round(mismatch, 5)}
        report.setdefault(col, {"source": None})
        report[col]["candidates"] = tried
        if col not in chosen:
            raise RuntimeError(f"No macro source matches the history for {col}: {tried}")
    return chosen, report


def build_new_batch(end: str | None = None) -> tuple[pd.DataFrame, dict]:
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    history = pd.read_csv(HISTORY_PATH, parse_dates=["date"])
    cutoff = history["date"].max()

    prices = fetch_all_prices(OVERLAP_START, end)

    # 1. Consistency check: the API must reproduce the historical rows on the overlap window
    hist_overlap = history[history["date"] >= OVERLAP_START][["symbol", "date", "original_close"]]
    chk = hist_overlap.merge(prices[["symbol", "date", "close"]], on=["symbol", "date"], how="left")
    price_mismatch = float((chk["close"] / chk["original_close"] - 1.0).abs().max())

    # 2. Drop incomplete sessions (e.g. an intraday snapshot cached before the close): too few symbols,
    #    or a median volume far below the recent norm.
    new = prices[prices["date"] > cutoff].copy()
    coverage = new.groupby("date")["symbol"].nunique()
    med_vol = new.groupby("date")["volume"].median()
    norm_vol = med_vol.rolling(20, min_periods=5).median().shift(1)
    incomplete = coverage.index[(coverage < MIN_SYMBOLS_PER_SESSION) | (med_vol < MIN_VOLUME_RATIO * norm_vol)]
    new = new[~new["date"].isin(incomplete)]

    # 3. Same raw schema as stocks_with_macro.csv (no splits after the cutoff -> factor 1.0)
    for col in ("open", "high", "low", "close", "volume"):
        new[f"original_{col}"] = new[col]
    new["adjustment_factor"] = 1.0

    tail = history[history["date"] == cutoff][["symbol", "close"]].rename(columns={"close": "prev_hist_close"})
    new = new.sort_values(["symbol", "date"]).merge(tail, on="symbol", how="left")
    prev_close = new.groupby("symbol")["close"].shift(1).fillna(new["prev_hist_close"])
    new["overnight_gap_pct"] = (new["open"] / prev_close - 1.0) * 100
    crash_counts = new[new["overnight_gap_pct"] < -8.5].groupby("date")["symbol"].count()
    crash_dates = set(crash_counts[crash_counts >= 4].index)
    new["is_ex_dividend_gap"] = ((new["overnight_gap_pct"] < -10.2) & ~new["date"].isin(crash_dates)).astype(int)

    # 3. Macro from the source that matches the history, forward-filled over a daily calendar
    macro, macro_report = choose_macro_sources(history, end)
    cal = pd.DataFrame(index=pd.date_range("2026-05-01", end, freq="D"))
    for col, s in macro.items():
        cal[col] = s[~s.index.duplicated(keep="last")]
    cal = cal.ffill()
    new = new.merge(cal, left_on="date", right_index=True, how="left")

    new = new[history.columns.tolist()].sort_values(["symbol", "date"]).reset_index(drop=True)
    NEW_BATCH_DIR.mkdir(parents=True, exist_ok=True)
    new.to_csv(NEW_BATCH_PATH, index=False)

    report = {
        "history_cutoff": str(cutoff.date()),
        "new_batch_range": [str(new["date"].min().date()), str(new["date"].max().date())],
        "rows": int(len(new)),
        "symbols": int(new["symbol"].nunique()),
        "overlap_max_price_mismatch": price_mismatch,
        "dropped_incomplete_sessions": [str(d.date()) for d in incomplete],
        "macro_sources": macro_report,
        "missing_macro_cells": int(new[list(MACRO_CANDIDATES)].isna().sum().sum()),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return new, report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", default=None, help="last date to fetch (default: today)")
    args = ap.parse_args()
    _, rep = build_new_batch(args.end)
    print(json.dumps(rep, indent=2))
    print(f"Saved new batch to {NEW_BATCH_PATH}")
