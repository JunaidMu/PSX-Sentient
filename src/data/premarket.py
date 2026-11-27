"""Pre-market job (run ~09:00 PKT, 30 minutes before the PSX open, Monday-Friday).

For the previous trading day it:
1. downloads the PSX daily files (market summary, off-market deals, futures open interest,
   corporate announcements, short-sell volume) into psx_historical_data/daily/<date>/
2. scrapes overnight news (Business Recorder, Google News, PSX notices; no fallback headlines) and
   turns that day's official announcements into factual headlines
3. scores all headlines with CPU FinBERT (+ PSX shock keywords), appends them to the news archive and
   computes each stock's 3-day EWMA sentiment S_t
4. computes each stock's off-market "whale" score W_ndm, futures open interest and short-sell volume
5. writes psx_historical_data/daily/<date>/premarket_snapshot.json and reports/premarket_latest.json,
   which the API uses for the conflict veto and the dashboard shows next to each prediction.

None of these sources is a model input (no training history yet); they are archived daily so they can
become features once enough history exists.

    python -m src.data.premarket [--date YYYY-MM-DD] [--no-news]
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from src.data.live_updater import compute_whale_block_metric
from src.data.psx_files import download_day, parse_day
from src.nlp.entity_mapper import KSE30_SYMBOLS, compute_title_md5

DAILY_DIR = Path("psx_historical_data/daily")
NEWS_ARCHIVE = Path("psx_historical_data/news/news_archive.csv")
LATEST_SNAPSHOT = Path("reports/premarket_latest.json")
VOLUME_HISTORY = Path("psx_historical_data/combined_clean.csv")
PKT = timezone(timedelta(hours=5))
SENTIMENT_STALE_DAYS = 3


def previous_trading_day(today: datetime | None = None) -> str:
    """Most recent weekday before today (PKT); the file download falls back further if it is a holiday."""
    d = (today or datetime.now(PKT)).date() - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d.isoformat()


def _num(token: str) -> float | None:
    token = token.replace(",", "")
    neg = token.startswith("(") and token.endswith(")")
    try:
        v = float(token.strip("()"))
    except ValueError:
        return None
    return -v if neg else v


def announcement_headline(symbol: str, text: str) -> str:
    """Turn a tabular announcement row into a plain factual sentence for sentiment scoring."""
    company = re.split(r"\s(?:1st|2nd|3rd|Half|Annual|Nine|-)(?=\s|$)", text, maxsplit=1)[0].strip()
    company = re.sub(r"^\([A-Z0-9]+\)\s*", "", company) or symbol
    parts = []
    period = re.search(r"(1st Quarter|2nd Quarter|3rd Quarter|Half Year|Annual|Nine Months)", text)
    body = re.sub(r"\d{2}-\d{2}-\d{4}|\d{2}-[A-Za-z]{3}-\d{2,4}", " ", text)  # drop dates
    nums = [n for n in (_num(t) for t in re.findall(r"\(?-?[\d,]+\.\d+\)?", body)) if n is not None]
    if period and len(nums) >= 2:
        after_tax = nums[1]
        parts.append(f"{company} reports {period.group(1).lower()} after-tax "
                     f"{'loss' if after_tax < 0 else 'profit'} of PKR {abs(after_tax):,.1f} million")
        if len(nums) >= 3:
            parts.append(f"EPS {nums[2]:+.2f}")
    ent = re.search(r"(\d+(?:\.\d+)?%\s*\(?[A-Z]?\)?\s*\((?:D|B|R)\))", text)
    if ent:
        kind = {"D": "cash dividend", "B": "bonus shares", "R": "right shares"}[ent.group(1)[-2]]
        parts.append(f"announces {kind} of {ent.group(1).split('(')[0].strip()}")
    if "book closure" in text.lower():
        parts.append("announces book closure" + (" for dividend entitlement" if ent else ""))
    if re.search(r"\b(AGM|EOGM)\b", text):
        parts.append("calls a general meeting")
    if not parts:
        return f"{company} corporate announcement"
    if not parts[0].startswith(company):
        parts[0] = f"{company} {parts[0]}"
    return ", ".join(parts)


def _score_news(news: pd.DataFrame) -> pd.DataFrame:
    """FinBERT + shock-keyword scores; returns an empty frame if the NLP extras are not installed."""
    if news.empty:
        return news.assign(sentiment_score=pd.Series(dtype="float64"))
    try:
        from src.nlp.finbert_engine import FinBERTSentimentEngine

        return FinBERTSentimentEngine(device="cpu").score_headlines_df(news)
    except ImportError:
        return news.iloc[0:0].assign(sentiment_score=pd.Series(dtype="float64"))


def update_news_archive(scored: pd.DataFrame) -> pd.DataFrame:
    keep = ["date", "source", "symbol", "title", "title_md5", "sentiment_score", "collected_at"]
    if not scored.empty:
        scored = scored.assign(collected_at=datetime.now(timezone.utc).isoformat())[keep]
    archive = pd.read_csv(NEWS_ARCHIVE) if NEWS_ARCHIVE.exists() else pd.DataFrame(columns=keep)
    merged = pd.concat([archive, scored], ignore_index=True).drop_duplicates(["title_md5", "symbol"], keep="first")
    NEWS_ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(NEWS_ARCHIVE, index=False)
    return merged


def sentiment_by_symbol(archive: pd.DataFrame, as_of: str) -> dict[str, dict[str, Any]]:
    """3-day EWMA of daily mean headline sentiment per symbol, as of `as_of` (stale -> None)."""
    from src.nlp.finbert_engine import compute_daily_ewma_sentiment

    if archive.empty:
        return {}
    df = archive.dropna(subset=["sentiment_score"]).copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"] <= pd.Timestamp(as_of) + pd.Timedelta(days=1)]  # include overnight news
    if df.empty:
        return {}
    daily = compute_daily_ewma_sentiment(df, span=3)
    out = {}
    for sym, g in daily.groupby("symbol"):
        last = g.sort_values("date").iloc[-1]
        age = (pd.Timestamp(as_of) - last["date"]).days
        heads = df[df["symbol"] == sym].sort_values("date").tail(3)
        out[sym] = {
            "S_t": None if age > SENTIMENT_STALE_DAYS else round(float(last["S_t"]), 4),
            "last_news_date": str(last["date"].date()),
            "headline_count_3d": int(len(df[(df["symbol"] == sym) & (df["date"] >= pd.Timestamp(as_of) - pd.Timedelta(days=3))])),
            "latest_headlines": [{"title": t, "score": round(float(s), 3)} for t, s in zip(heads["title"], heads["sentiment_score"])][::-1],
        }
    return out


def _avg_volume_20(symbols: list[str]) -> dict[str, float]:
    if not VOLUME_HISTORY.exists():
        return {}
    hist = pd.read_csv(VOLUME_HISTORY, usecols=["symbol", "date", "volume"], parse_dates=["date"])
    hist = hist[hist["symbol"].isin(symbols)].sort_values("date")
    return hist.groupby("symbol")["volume"].apply(lambda s: float(s.tail(20).mean())).to_dict()


def run(date: str | None = None, with_news: bool = True) -> dict[str, Any]:
    # 1. PSX files for the previous trading day (step back over holidays until the files exist)
    target = date or previous_trading_day()
    for _ in range(7):
        day_dir = DAILY_DIR / target
        paths = download_day(target, day_dir)
        if paths["market_summary"] is not None:
            break
        if date:  # an explicit date was requested: do not search further back
            break
        target = (pd.Timestamp(target) - pd.offsets.BDay(1)).date().isoformat()
    parsed = parse_day(paths)
    ours = list(KSE30_SYMBOLS)

    prices = parsed["market_summary"]
    prices = prices[prices["symbol"].isin(ours)] if not prices.empty else prices
    if not prices.empty:
        prices.to_csv(day_dir / "prices_kse30.csv", index=False)
    close = dict(zip(prices["symbol"], prices["close"])) if not prices.empty else {}

    # 2. Off-market whale score per stock (volume-weighted deal rate vs. the regular-market close)
    vol20 = _avg_volume_20(ours)
    whales: dict[str, dict[str, Any]] = {}
    om = parsed["off_market"]
    if not om.empty:
        om = om[om["symbol"].isin(ours)]
        for sym, g in om.groupby("symbol"):
            vwap = float((g["volume"] * g["rate"]).sum() / g["volume"].sum())
            if sym in close and sym in vol20:
                whales[sym] = compute_whale_block_metric(float(g["volume"].sum()), vwap, float(close[sym]), vol20[sym])
                whales[sym]["deals"] = int(len(g))

    fut = parsed["futures_oi"]
    fut = fut.set_index("symbol").to_dict("index") if not fut.empty else {}
    ss = parsed["short_sell"]
    ss = ss.groupby("symbol")[["short_volume", "short_value"]].sum().to_dict("index") if not ss.empty else {}
    ann = parsed["announcements"]
    ann_ours = ann[ann["symbol"].isin(ours)] if not ann.empty else ann

    # 3. News: overnight scrape + official announcements as factual headlines -> FinBERT -> archive
    sentiment: dict[str, dict[str, Any]] = {}
    news_status = "skipped"
    if with_news:
        from src.nlp.entity_mapper import map_headlines_df
        from src.nlp.scrape_news import fetch_all_psx_news

        news = fetch_all_psx_news(timeout=15, max_items_per_source=40, allow_fallback=False)
        ann_rows = [{"date": target, "source": "PSX Daily Announcements", "title": announcement_headline(r.symbol, r.text)}
                    for r in ann_ours.itertuples()] if not ann_ours.empty else []
        if ann_rows:
            ann_df = map_headlines_df(pd.DataFrame(ann_rows))
            news = pd.concat([news, ann_df], ignore_index=True)
        scored = _score_news(news)
        news_status = f"{len(news)} headlines, {len(scored)} scored" + ("" if len(scored) or news.empty else " (FinBERT not installed)")
        archive = update_news_archive(scored)
        sentiment = sentiment_by_symbol(archive, target)

    # 4. Snapshot per stock
    stocks = {}
    for sym in ours:
        f = fut.get(sym, {})
        stocks[sym] = {
            "close": close.get(sym),
            "sentiment": sentiment.get(sym, {"S_t": None, "headline_count_3d": 0, "latest_headlines": []}),
            "whale": whales.get(sym),
            "futures_open_interest": {"contracts": f.get("oi_contracts"), "volume": f.get("oi_volume"),
                                      "pct_free_float": None if f.get("oi_pct_free_float") is None
                                      else round(float(f["oi_pct_free_float"]), 4)} if f else None,
            "short_sell": ss.get(sym),
            "announcements": ann_ours[ann_ours["symbol"] == sym]["text"].tolist() if not ann_ours.empty else [],
        }
    snapshot = {
        "trading_day": target,
        "generated_at_pkt": datetime.now(PKT).strftime("%Y-%m-%d %H:%M"),
        "sources": {k: ("ok" if v is not None else "unavailable") for k, v in paths.items()},
        "news": news_status,
        "macro_sentiment": sentiment.get("MACRO"),
        "counts": {"whale_blocks": sum(1 for w in whales.values() if w and w.get("is_whale_block")),
                   "stocks_with_news": sum(1 for s in stocks.values() if s["sentiment"].get("S_t") is not None),
                   "announcements": int(len(ann_ours)) if not ann_ours.empty else 0},
        "stocks": stocks,
    }
    text = json.dumps(snapshot, indent=2, default=lambda o: None if isinstance(o, float) and np.isnan(o) else str(o))
    (day_dir / "premarket_snapshot.json").write_text(text, encoding="utf-8")
    LATEST_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    LATEST_SNAPSHOT.write_text(text, encoding="utf-8")
    return snapshot


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="trading day to collect (default: previous trading day)")
    ap.add_argument("--no-news", action="store_true")
    a = ap.parse_args()
    snap = run(a.date, with_news=not a.no_news)
    print(f"Pre-market snapshot for {snap['trading_day']} (generated {snap['generated_at_pkt']} PKT)")
    print("sources:", snap["sources"], "| news:", snap["news"], "| counts:", snap["counts"])
    for sym, s in snap["stocks"].items():
        w = s["whale"] or {}
        st = s["sentiment"].get("S_t")
        fo = (s["futures_open_interest"] or {}).get("pct_free_float")
        if st is not None or w.get("is_whale_block") or s["announcements"]:
            print(f"  {sym:<6} close={s['close']} S_t={st} whale={w.get('W_ndm')} fut_oi%={fo} ann={len(s['announcements'])}")
