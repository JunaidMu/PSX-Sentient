"""Unit tests for Phase 4: NLP Pipeline (News Scraping, Entity Mapping, CPU FinBERT & 3-Day EWMA)."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest

from src.nlp.entity_mapper import (
    KSE30_SYMBOLS,
    compute_title_md5,
    deduplicate_headlines,
    map_headline_entities,
    map_headlines_df,
)
from src.nlp.finbert_engine import (
    FinBERTSentimentEngine,
    apply_psx_shock_multiplier,
    compute_daily_ewma_sentiment,
)
from src.nlp.scrape_news import (
    fetch_all_psx_news,
    parse_psx_announcements_html,
    parse_rss_xml,
)


class _BrokenSession:
    """Simulated HTTP session that always raises a timeout/connection error."""

    def get(self, *args, **kwargs):
        raise TimeoutError("Simulated network timeout")


def test_entity_mapper_30_stocks_and_macro() -> None:
    """Verify headlines map accurately to the 30 KSE-100 symbols or 'MACRO'."""
    assert len(KSE30_SYMBOLS) == 30

    assert map_headline_entities("Lucky Cement and HBL post strong quarterly earnings") == ["HBL", "LUCK"]
    assert map_headline_entities("OGDCL discovers new hydrocarbon reserves in Sindh") == ["OGDC"]
    assert map_headline_entities("Millat Tractors announces 50% bonus issue") == ["MTL"]
    assert map_headline_entities("SBP keeps policy rate unchanged amid IMF review") == ["MACRO"]


def test_md5_title_deduplication() -> None:
    """Verify normalized MD5 hashing deduplicates identical or whitespace-varied titles."""
    raw = pd.DataFrame(
        [
            {"date": "2026-06-29", "source": "BR", "title": "UBL declares 110% interim cash dividend"},
            {"date": "2026-06-29", "source": "GNews", "title": "  ubl declares 110% interim cash dividend  "},
            {"date": "2026-06-29", "source": "PSX", "title": "ATRL reports temporary refinery shutdown"},
        ]
    )
    deduped = deduplicate_headlines(raw)
    assert len(deduped) == 2
    assert deduped["title_md5"].nunique() == 2
    assert compute_title_md5("Hello PSX") == compute_title_md5("  hello   psx ")


def test_scraper_parsers_and_graceful_fallback() -> None:
    """Verify RSS XML parser, PSX HTML parser, and resilient network failure fallback."""
    sample_rss = """<?xml version="1.0" encoding="UTF-8" ?>
    <rss version="2.0"><channel>
        <item>
            <title>Pakistan State Oil (PSO) receivables decline after circular debt settlement</title>
            <pubDate>Mon, 29 Jun 2026 08:30:00 GMT</pubDate>
        </item>
    </channel></rss>"""
    rss_items = parse_rss_xml(sample_rss, source_name="Business Recorder RSS")
    assert len(rss_items) == 1
    assert rss_items[0]["date"] == "2026-06-29"
    assert "PSO" in rss_items[0]["title"]

    sample_html = """
    <html><body><table><tbody>
        <tr>
            <td>2026-06-29</td><td>10:15 AM</td><td>FFC</td>
            <td>Fauji Fertilizer Company</td><td>Board Meeting Results &amp; Dividend</td>
        </tr>
    </tbody></table></body></html>
    """
    html_items = parse_psx_announcements_html(sample_html)
    assert len(html_items) == 1
    assert "FFC" in html_items[0]["title"]

    # Network failure must NEVER crash fetch_all_psx_news; it must return fallback mapped headlines
    fallback_df = fetch_all_psx_news(timeout=0.1, allow_fallback=True, session=_BrokenSession())
    assert not fallback_df.empty
    assert {"date", "source", "title", "title_md5", "symbol"}.issubset(fallback_df.columns)


def test_psx_corporate_shock_multipliers() -> None:
    """Verify PSX corporate shock keywords (dividend, bonus, profit, shutdown, tax, IMF) amplify scores."""
    pos_adj, pos_kws, pos_mult = apply_psx_shock_multiplier(
        "MARI announces record profit and 200% bonus dividend", base_score=0.50
    )
    assert set(pos_kws) == {"dividend", "bonus", "profit"}
    assert pos_mult > 1.20
    assert pos_adj > 0.50
    assert -1.0 <= pos_adj <= 1.0

    neg_adj, neg_kws, neg_mult = apply_psx_shock_multiplier(
        "ATRL announces partial plant shutdown after new super tax", base_score=-0.50
    )
    assert set(neg_kws) == {"shutdown", "tax"}
    assert neg_mult > 1.20
    assert neg_adj < -0.50
    assert -1.0 <= neg_adj <= 1.0


def test_finbert_cpu_enforcement_and_scoring() -> None:
    """Verify FinBERT strictly enforces device='cpu' and scores positive vs negative PSX headlines."""
    with pytest.raises(ValueError, match="strictly on CPU"):
        FinBERTSentimentEngine(device="cuda")

    # FinBERT scoring needs the optional NLP extras (requirements-nlp.txt)
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    engine = FinBERTSentimentEngine(device="cpu")
    sample_df = pd.DataFrame(
        [
            {
                "date": "2026-06-27",
                "source": "PSX",
                "title": "United Bank Limited (UBL) reports record quarterly profit and hefty cash dividend",
            },
            {
                "date": "2026-06-28",
                "source": "BR",
                "title": "Attock Refinery (ATRL) suffers losses following plant shutdown and heavy tax",
            },
            {
                "date": "2026-06-29",
                "source": "BR",
                "title": "IMF approves tranche for Pakistan boosting KSE-100 market outlook",
            },
        ]
    )
    mapped = map_headlines_df(sample_df)
    scored = engine.score_headlines_df(mapped)

    assert len(scored) == 3
    assert ((scored["sentiment_score"] >= -1.0) & (scored["sentiment_score"] <= 1.0)).all()

    ubl_score = float(scored.loc[scored["symbol"] == "UBL", "sentiment_score"].iloc[0])
    atrl_score = float(scored.loc[scored["symbol"] == "ATRL", "sentiment_score"].iloc[0])
    assert ubl_score > 0.25, f"Expected strongly positive FinBERT score for UBL dividend news, got {ubl_score}"
    assert atrl_score < -0.25, f"Expected strongly negative FinBERT score for ATRL shutdown news, got {atrl_score}"


def test_3day_ewma_daily_sentiment_bounds_and_causality() -> None:
    """Verify 3-day EWMA sentiment S_t stays in [-1, 1] and is strictly causal per symbol."""
    synthetic_scored = pd.DataFrame(
        [
            {"symbol": "LUCK", "date": "2026-06-25", "sentiment_score": 0.80},
            {"symbol": "LUCK", "date": "2026-06-26", "sentiment_score": 0.40},
            {"symbol": "LUCK", "date": "2026-06-29", "sentiment_score": -0.60},
            {"symbol": "OGDC", "date": "2026-06-25", "sentiment_score": -0.50},
        ]
    )
    ewma_df = compute_daily_ewma_sentiment(synthetic_scored, span=3)
    assert ((ewma_df["S_t"] >= -1.0) & (ewma_df["S_t"] <= 1.0)).all()

    luck = ewma_df[ewma_df["symbol"] == "LUCK"].sort_values("date").reset_index(drop=True)
    # For span=3, alpha = 2 / (3 + 1) = 0.5
    # t=0: 0.80
    # t=1: 0.5 * 0.40 + 0.5 * 0.80 = 0.60
    # t=2: 0.5 * (-0.60) + 0.5 * 0.60 = 0.00
    np.testing.assert_allclose(luck["S_t"].to_numpy(), np.array([0.80, 0.60, 0.00]), rtol=1e-12, atol=1e-12)
