"""Download and parse the PSX Data Portal daily files (https://dps.psx.com.pk/downloads).

Files (plain GET, one per trading day):
- mkt_summary/<date>.Z         Market Summary (Closing): a ZIP holding a pipe-separated .lis file
                               date|symbol|sector|name|open|high|low|close|volume|prev_close|||
- omts/<date>.csv              Off Market Transaction Summary (block deals: symbol, volume, rate)
- fut_opn_int/<date>.xls       Symbol-wise open interest in deliverable futures (DFC), sheet 2
- announce/<date>.pdf          Daily corporate announcements (results, dividends, AGMs, book closures)
- short_sell_vol/<date>.pdf    Ready-market short-sell volume per symbol
"""

from __future__ import annotations

from io import BytesIO, StringIO
from pathlib import Path
import re
import time
import zipfile

import pandas as pd
import requests

BASE_URL = "https://dps.psx.com.pk/download"
HEADERS = {"User-Agent": "psx-sentient/1.0 (+https://github.com/JunaidMu/PSX-Sentient)"}
FILES = {
    "market_summary": ("mkt_summary", "Z"),
    "off_market": ("omts", "csv"),
    "futures_oi": ("fut_opn_int", "xls"),
    "announcements": ("announce", "pdf"),
    "short_sell": ("short_sell_vol", "pdf"),
}


def download_day(date: str, out_dir: Path, session: requests.Session | None = None) -> dict[str, Path | None]:
    """Download every daily file for `date` (YYYY-MM-DD) into out_dir; missing files map to None."""
    http = session or requests.Session()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path | None] = {}
    for key, (folder, ext) in FILES.items():
        target = out_dir / f"{key}.{ext}"
        if target.exists() and target.stat().st_size > 0:
            paths[key] = target
            continue
        try:
            resp = http.get(f"{BASE_URL}/{folder}/{date}.{ext}", headers=HEADERS, timeout=60)
            ok = resp.status_code == 200 and len(resp.content) > 100 and not resp.content.lstrip().startswith(b"<")
        except requests.RequestException:
            ok = False
        if ok:
            target.write_bytes(resp.content)
            paths[key] = target
        else:
            paths[key] = None
        time.sleep(1.0)
    return paths


def download_file(date: str, key: str, out_dir: Path, session: requests.Session | None = None) -> Path | None:
    """Download one daily file (e.g. key='market_summary') for `date`; cached, None if unavailable."""
    folder, ext = FILES[key]
    target = out_dir / f"{key}.{ext}"
    if target.exists() and target.stat().st_size > 0:
        return target
    try:
        resp = (session or requests).get(f"{BASE_URL}/{folder}/{date}.{ext}", headers=HEADERS, timeout=60)
    except requests.RequestException:
        return None
    if resp.status_code != 200 or len(resp.content) <= 100 or resp.content.lstrip().startswith(b"<"):
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    target.write_bytes(resp.content)
    return target


def parse_market_summary(path: Path) -> pd.DataFrame:
    """Official closing prices for every listed equity (futures contracts like OGDC-DECB dropped)."""
    with zipfile.ZipFile(path) as zf:
        text = zf.read(zf.namelist()[0]).decode("latin-1")
    cols = ["date", "symbol", "sector", "name", "open", "high", "low", "close", "volume", "prev_close"]
    rows = [line.split("|")[:10] for line in text.splitlines() if line.count("|") >= 9]
    df = pd.DataFrame(rows, columns=cols)
    df = df[~df["symbol"].str.contains("-")].copy()
    for c in ("open", "high", "low", "close", "volume", "prev_close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], format="%d%b%Y")
    return df.reset_index(drop=True)


def parse_off_market(path: Path) -> pd.DataFrame:
    """Every off-market block deal (both sections of the CSV) as symbol, volume, rate, value."""
    text = path.read_text(encoding="latin-1")
    deal_lines = [ln for ln in text.splitlines() if re.match(r"^\d{2}-[A-Za-z]{3}-\d{2},", ln.strip())]
    if not deal_lines:
        return pd.DataFrame(columns=["date", "settlement", "symbol", "company", "volume", "rate", "value"])
    df = pd.read_csv(StringIO("\n".join(deal_lines)), header=None,
                     names=["date", "settlement", "members", "symbol", "company", "volume", "rate", "value"])
    df["symbol"] = df["symbol"].astype(str).str.strip()
    df["date"] = df["date"].astype(str).str.strip()
    df["settlement"] = df["settlement"].astype(str).str.strip()
    for c in ("volume", "rate", "value"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["date", "settlement", "symbol", "company", "volume", "rate", "value"]].dropna(subset=["volume", "rate"])


def parse_futures_oi(path: Path) -> pd.DataFrame:
    """Open interest per underlying stock, summed over its contract months."""
    sheets = pd.read_excel(path, header=None, sheet_name=None)
    df = max(sheets.values(), key=len)  # the data sheet is the large one
    df = df[pd.to_numeric(df[0], errors="coerce").notna()].copy()
    df = df.rename(columns={1: "contract", 3: "oi_contracts", 4: "oi_volume", 5: "oi_value", 6: "free_float", 7: "oi_pct_free_float"})
    df["symbol"] = df["contract"].astype(str).str.split("-").str[0].str.strip()
    for c in ("oi_contracts", "oi_volume", "oi_value", "oi_pct_free_float"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.groupby("symbol", as_index=False)[["oi_contracts", "oi_volume", "oi_value", "oi_pct_free_float"]].sum()


def _pdf_text(path: Path) -> str:
    from pypdf import PdfReader

    return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)


def parse_announcements(path: Path) -> pd.DataFrame:
    """One row per announcement: symbol + the raw line (results, entitlements, AGM, book closure)."""
    rows = []
    for line in _pdf_text(path).splitlines():
        m = re.match(r"^\s*\d+\s+([A-Z][A-Z0-9]+)\s+(.*)$", line)
        if m:
            text = re.sub(r"\s+", " ", m.group(2)).strip()
            rows.append({"symbol": m.group(1), "text": text,
                         "dividend_or_bonus": bool(re.search(r"\(D\)|\(B\)|dividend|bonus", text, re.I)),
                         "book_closure": "book closure" in text.lower()})
    return pd.DataFrame(rows, columns=["symbol", "text", "dividend_or_bonus", "book_closure"])


def parse_short_sell(path: Path) -> pd.DataFrame:
    """Short-sell volume per symbol (empty when PSX reports no short sales that day)."""
    rows = []
    for line in _pdf_text(path).splitlines():
        m = re.match(r"^\s*([A-Z][A-Z0-9]+)\s+.*?\s([\d,]+)\s+([\d,]+(?:\.\d+)?)\s*$", line)
        if m and m.group(1) not in {"NOTE", "DATE", "TOTAL"}:
            rows.append({"symbol": m.group(1), "short_volume": float(m.group(2).replace(",", "")),
                         "short_value": float(m.group(3).replace(",", ""))})
    return pd.DataFrame(rows, columns=["symbol", "short_volume", "short_value"])


def parse_day(paths: dict[str, Path | None]) -> dict[str, pd.DataFrame]:
    parsers = {"market_summary": parse_market_summary, "off_market": parse_off_market,
               "futures_oi": parse_futures_oi, "announcements": parse_announcements, "short_sell": parse_short_sell}
    out = {}
    for key, fn in parsers.items():
        p = paths.get(key)
        try:
            out[key] = fn(p) if p else pd.DataFrame()
        except Exception:
            out[key] = pd.DataFrame()
    return out
