"""CPU FinBERT Sentiment Engine with PSX Corporate Shock Multipliers and 3-Day EWMA.

Strict Engineering Rule:
- Always loads and runs `ProsusAI/finbert` strictly on CPU (`device="cpu"`).
"""

from __future__ import annotations

from pathlib import Path
import re
import sys
import types
from typing import Sequence
import numpy as np
import pandas as pd

try:
    import numpy.fft  # noqa: F401
except ImportError:
    _fft_stub = types.ModuleType("numpy.fft")
    sys.modules["numpy.fft"] = _fft_stub
    np.fft = _fft_stub  # type: ignore[attr-defined]

try:
    import regex  # noqa: F401
except ImportError:
    sys.modules["regex"] = re  # type: ignore[assignment]

MODEL_NAME = "ProsusAI/finbert"
LOCAL_FINBERT_CACHE = Path("models/finbert_cache")

# PSX corporate & macro shock keywords and their directional/magnitude multipliers
PSX_SHOCK_KEYWORDS: dict[str, tuple[float, float]] = {
    # keyword: (magnitude_multiplier, directional_nudge_if_neutral)
    "dividend": (1.30, +0.20),
    "bonus": (1.30, +0.20),
    "profit": (1.25, +0.15),
    "shutdown": (1.35, -0.30),
    "tax": (1.25, -0.15),
    "imf": (1.25, +0.10),
}

_SHOCK_PATTERNS: dict[str, re.Pattern[str]] = {
    kw: re.compile(rf"\b{re.escape(kw)}\w*\b", flags=re.IGNORECASE)
    for kw in PSX_SHOCK_KEYWORDS
}


def apply_psx_shock_multiplier(title: str, base_score: float) -> tuple[float, list[str], float]:
    """Apply PSX corporate shock keyword multipliers and clamp to [-1.0, 1.0].

    Returns
    -------
    tuple[float, list[str], float]
        (adjusted_score, matched_keywords, combined_multiplier)
    """
    text = str(title or "")
    matched_kws: list[str] = []
    combined_mult = 1.0
    nudge_sum = 0.0

    for kw, (mult, nudge) in PSX_SHOCK_KEYWORDS.items():
        if _SHOCK_PATTERNS[kw].search(text):
            matched_kws.append(kw)
            combined_mult *= mult
            nudge_sum += nudge

    if not matched_kws:
        return float(np.clip(base_score, -1.0, 1.0)), [], 1.0

    # Cap combined multiplier to avoid extreme distortion when multiple keywords co-occur
    combined_mult = min(combined_mult, 1.60)

    # If base FinBERT score is very small in magnitude (< 0.12), add domain directional nudge
    score = float(base_score)
    if abs(score) < 0.12:
        score = score + nudge_sum
    adjusted = float(np.clip(score * combined_mult, -1.0, 1.0))
    return adjusted, matched_kws, float(round(combined_mult, 4))


class FinBERTSentimentEngine:
    """CPU-only FinBERT (`ProsusAI/finbert`) inference engine for PSX financial headlines."""

    def __init__(self, model_name: str = MODEL_NAME, device: str = "cpu") -> None:
        if device.lower() != "cpu":
            raise ValueError("Strict Engineering Rule Violation: FinBERT must run strictly on CPU (device='cpu').")
        self.model_name = model_name
        self.device = "cpu"
        self._tokenizer = None
        self._model = None

    def _ensure_loaded(self) -> None:
        """Lazy-load tokenizer and model strictly onto CPU (using local workspace cache if present)."""
        if self._model is not None and self._tokenizer is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        load_source = (
            str(LOCAL_FINBERT_CACHE)
            if (LOCAL_FINBERT_CACHE / "config.json").exists()
            else self.model_name
        )
        self._tokenizer = AutoTokenizer.from_pretrained(load_source)
        self._model = AutoModelForSequenceClassification.from_pretrained(load_source)
        if load_source == self.model_name:
            LOCAL_FINBERT_CACHE.mkdir(parents=True, exist_ok=True)
            self._tokenizer.save_pretrained(LOCAL_FINBERT_CACHE)
            self._model.save_pretrained(LOCAL_FINBERT_CACHE)
        self._model.to(torch.device("cpu"))
        self._model.eval()

    def score_texts(self, texts: Sequence[str], batch_size: int = 16) -> pd.DataFrame:
        """Score a sequence of headlines on CPU and apply PSX shock multipliers.

        Returns DataFrame with columns:
        `title`, `p_positive`, `p_negative`, `p_neutral`, `finbert_raw_score`,
        `shock_keywords`, `shock_multiplier`, `sentiment_score`.
        """
        if not texts:
            return pd.DataFrame(
                columns=[
                    "title",
                    "p_positive",
                    "p_negative",
                    "p_neutral",
                    "finbert_raw_score",
                    "shock_keywords",
                    "shock_multiplier",
                    "sentiment_score",
                ]
            )

        self._ensure_loaded()
        import torch

        # Determine label indices from model config (ProsusAI/finbert: positive, negative, neutral)
        id2label = {int(k): str(v).lower() for k, v in self._model.config.id2label.items()}
        pos_idx = next(i for i, lbl in id2label.items() if "pos" in lbl)
        neg_idx = next(i for i, lbl in id2label.items() if "neg" in lbl)
        neu_idx = next(i for i, lbl in id2label.items() if "neu" in lbl)

        rows: list[dict[str, object]] = []
        text_list = [str(t) for t in texts]

        with torch.no_grad():
            for start in range(0, len(text_list), batch_size):
                batch_texts = text_list[start : start + batch_size]
                encoded = self._tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                )
                encoded = {k: v.to(torch.device("cpu")) for k, v in encoded.items()}
                logits = self._model(**encoded).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

                for text_item, prob_vec in zip(batch_texts, probs):
                    p_pos = float(prob_vec[pos_idx])
                    p_neg = float(prob_vec[neg_idx])
                    p_neu = float(prob_vec[neu_idx])
                    raw_score = p_pos - p_neg
                    adj_score, kws, mult = apply_psx_shock_multiplier(text_item, raw_score)
                    rows.append(
                        {
                            "title": text_item,
                            "p_positive": round(p_pos, 6),
                            "p_negative": round(p_neg, 6),
                            "p_neutral": round(p_neu, 6),
                            "finbert_raw_score": round(raw_score, 6),
                            "shock_keywords": ",".join(kws),
                            "shock_multiplier": mult,
                            "sentiment_score": round(adj_score, 6),
                        }
                    )

        return pd.DataFrame(rows)

    def score_headlines_df(self, df: pd.DataFrame, batch_size: int = 16) -> pd.DataFrame:
        """Score a mapped headlines DataFrame (deduplicating unique titles during CPU inference)."""
        if df.empty:
            out = df.copy()
            for col in (
                "p_positive",
                "p_negative",
                "p_neutral",
                "finbert_raw_score",
                "shock_keywords",
                "shock_multiplier",
                "sentiment_score",
            ):
                out[col] = pd.Series(dtype="float64" if col != "shock_keywords" else "object")
            return out

        unique_titles = df["title"].drop_duplicates().tolist()
        scored_unique = self.score_texts(unique_titles, batch_size=batch_size)
        merged = pd.merge(df, scored_unique, on="title", how="left")
        return merged


def compute_daily_ewma_sentiment(
    scored_df: pd.DataFrame,
    span: int = 3,
) -> pd.DataFrame:
    """Aggregate daily headline scores per `(symbol, date)` and compute 3-day EWMA $S_t \\in [-1, 1]$.

    Strictly groups by `symbol` after sorting chronologically by `(symbol, date)`.
    """
    if scored_df.empty:
        return pd.DataFrame(columns=["symbol", "date", "daily_raw_sentiment", "S_t", "headline_count"])

    df = scored_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    daily = (
        df.groupby(["symbol", "date"], as_index=False)
        .agg(
            daily_raw_sentiment=("sentiment_score", "mean"),
            headline_count=("sentiment_score", "count"),
        )
        .sort_values(["symbol", "date"])
        .reset_index(drop=True)
    )

    daily["S_t"] = (
        daily.groupby("symbol", sort=False)["daily_raw_sentiment"]
        .transform(lambda s: s.ewm(span=span, adjust=False).mean())
        .clip(lower=-1.0, upper=1.0)
    )
    return daily


def run_live_nlp_pipeline(
    max_items_per_source: int = 20,
    timeout: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch live PSX headlines, map to 30 stocks/MACRO, score with CPU FinBERT, and compute 3-day EWMA."""
    from src.nlp.scrape_news import fetch_all_psx_news

    news_df = fetch_all_psx_news(
        timeout=timeout,
        max_items_per_source=max_items_per_source,
        allow_fallback=True,
    )
    engine = FinBERTSentimentEngine(device="cpu")
    scored_df = engine.score_headlines_df(news_df)
    ewma_df = compute_daily_ewma_sentiment(scored_df, span=3)
    return scored_df, ewma_df


if __name__ == "__main__":
    scored, ewma = run_live_nlp_pipeline(max_items_per_source=15, timeout=10.0)
    print(f"\nScraped, deduplicated & mapped rows: {len(scored)} (Unique headlines: {scored['title_md5'].nunique()})")
    display_cols = [
        "date",
        "source",
        "symbol",
        "finbert_raw_score",
        "shock_keywords",
        "sentiment_score",
        "title",
    ]
    stock_rows = scored[scored["symbol"] != "MACRO"]
    macro_rows = scored[scored["symbol"] == "MACRO"]
    sample_view = pd.concat([stock_rows.head(10), macro_rows.head(6)], ignore_index=True)
    print("\nSample Live FinBERT-Scored PSX Headlines (Stock-Specific & MACRO on CPU):")
    print(sample_view[display_cols].to_string(index=False))
    print("\n3-Day EWMA Daily Sentiment (S_t in [-1, 1]):")
    print(ewma.tail(15).to_string(index=False))

