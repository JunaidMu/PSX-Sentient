"""Phase 4: NLP Pipeline — News Scraping, Entity Mapping, and CPU FinBERT Sentiment Scoring."""

from src.nlp.entity_mapper import (
    KSE30_SYMBOLS,
    SYMBOL_ALIASES,
    compute_title_md5,
    deduplicate_headlines,
    map_headline_entities,
    map_headlines_df,
)
from src.nlp.finbert_engine import (
    PSX_SHOCK_KEYWORDS,
    FinBERTSentimentEngine,
    apply_psx_shock_multiplier,
    compute_daily_ewma_sentiment,
)
from src.nlp.scrape_news import (
    BUSINESS_RECORDER_RSS_URL,
    GOOGLE_NEWS_PSX_RSS_URL,
    PSX_ANNOUNCEMENTS_URL,
    fetch_all_psx_news,
)

__all__ = [
    "BUSINESS_RECORDER_RSS_URL",
    "GOOGLE_NEWS_PSX_RSS_URL",
    "KSE30_SYMBOLS",
    "PSX_ANNOUNCEMENTS_URL",
    "PSX_SHOCK_KEYWORDS",
    "SYMBOL_ALIASES",
    "FinBERTSentimentEngine",
    "apply_psx_shock_multiplier",
    "compute_daily_ewma_sentiment",
    "compute_title_md5",
    "deduplicate_headlines",
    "fetch_all_psx_news",
    "map_headline_entities",
    "map_headlines_df",
]
