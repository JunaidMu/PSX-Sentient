"""Entity Mapper & MD5 Title Deduplication for the 30 KSE-100 Stocks and MACRO news."""

from __future__ import annotations

import hashlib
import re
import pandas as pd

KSE30_SYMBOLS: tuple[str, ...] = (
    "AKBL",
    "ATRL",
    "BAFL",
    "BAHL",
    "DGKC",
    "FCCL",
    "FFC",
    "HBL",
    "HUBC",
    "INDU",
    "INIL",
    "KAPCO",
    "LUCK",
    "MARI",
    "MCB",
    "MLCF",
    "MTL",
    "NBP",
    "NML",
    "NRL",
    "OGDC",
    "PIOC",
    "PKGS",
    "POL",
    "PPL",
    "PSO",
    "PTC",
    "SNGP",
    "SSGC",
    "UBL",
)

SYMBOL_ALIASES: dict[str, tuple[str, ...]] = {
    "AKBL": ("akbl", "askari bank", "askari commercial bank"),
    "ATRL": ("atrl", "attock refinery"),
    "BAFL": ("bafl", "bank alfalah", "alfalah bank"),
    "BAHL": ("bahl", "bank al habib", "bank al-habib"),
    "DGKC": ("dgkc", "d.g. khan cement", "dg khan cement"),
    "FCCL": ("fccl", "fauji cement"),
    "FFC": ("ffc", "fauji fertilizer company", "fauji fertilizer"),
    "HBL": ("hbl", "habib bank limited", "habib bank"),
    "HUBC": ("hubc", "hub power", "hubco"),
    "INDU": ("indu", "indus motor", "toyota pakistan", "pak suzuki indus"),
    "INIL": ("inil", "international industries"),
    "KAPCO": ("kapco", "kot addu power"),
    "LUCK": ("luck", "lucky cement", "lucky core"),
    "MARI": ("mari", "mari petroleum", "mari energies", "mari gas"),
    "MCB": ("mcb", "mcb bank", "muslim commercial bank"),
    "MLCF": ("mlcf", "maple leaf cement"),
    "MTL": ("mtl", "millat tractors", "millat tractor"),
    "NBP": ("nbp", "national bank of pakistan", "national bank"),
    "NML": ("nml", "nishat mills", "nishat group"),
    "NRL": ("nrl", "national refinery"),
    "OGDC": ("ogdc", "ogdcl", "oil and gas development", "oil & gas development"),
    "PIOC": ("pioc", "pioneer cement"),
    "PKGS": ("pkgs", "packages limited", "packages ltd", "packages group"),
    "POL": ("pol", "pakistan oilfields", "pakistan oilfield"),
    "PPL": ("ppl", "pakistan petroleum"),
    "PSO": ("pso", "pakistan state oil"),
    "PTC": ("ptc", "ptcl", "pakistan telecommunication"),
    "SNGP": ("sngp", "sngpl", "sui northern gas", "sui northern"),
    "SSGC": ("ssgc", "ssgcl", "sui southern gas", "sui southern"),
    "UBL": ("ubl", "united bank limited", "united bank"),
}

_COMPILED_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    sym: [re.compile(rf"\b{re.escape(alias)}\b", flags=re.IGNORECASE) for alias in aliases]
    for sym, aliases in SYMBOL_ALIASES.items()
}


def normalize_title(title: str) -> str:
    """Normalize headline text for deterministic MD5 hashing."""
    cleaned = re.sub(r"\s+", " ", str(title).strip().lower())
    return cleaned


def compute_title_md5(title: str) -> str:
    """Compute MD5 hex digest of normalized headline text."""
    norm = normalize_title(title)
    return hashlib.md5(norm.encode("utf-8")).hexdigest()


def map_headline_entities(title: str) -> list[str]:
    """Map a headline to one or more of the 30 KSE-100 symbols, or ['MACRO'] if none match."""
    text = str(title or "")
    matched: list[str] = []
    for sym, patterns in _COMPILED_PATTERNS.items():
        if any(p.search(text) for p in patterns):
            matched.append(sym)
    return matched if matched else ["MACRO"]


def deduplicate_headlines(df: pd.DataFrame, title_col: str = "title") -> pd.DataFrame:
    """Add `title_md5` column and remove duplicate headlines deterministically."""
    if df.empty:
        out = df.copy()
        out["title_md5"] = pd.Series(dtype="object")
        return out

    out = df.copy()
    out["title"] = out[title_col].astype(str).str.strip()
    out = out[out["title"].str.len() > 0].copy()
    out["title_md5"] = out["title"].map(compute_title_md5)
    out = out.drop_duplicates(subset=["title_md5"], keep="first").reset_index(drop=True)
    return out


def map_headlines_df(df: pd.DataFrame, explode_multi_symbol: bool = True) -> pd.DataFrame:
    """Deduplicate headlines by MD5 and map each headline to the 30 symbols or 'MACRO'."""
    deduped = deduplicate_headlines(df, title_col="title")
    if deduped.empty:
        deduped["symbol"] = pd.Series(dtype="object")
        return deduped

    mapped_lists = deduped["title"].map(map_headline_entities)
    if explode_multi_symbol:
        deduped["symbol"] = mapped_lists
        deduped = deduped.explode("symbol", ignore_index=True)
    else:
        deduped["symbol"] = mapped_lists.map(lambda items: items[0])

    return deduped
