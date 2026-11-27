"""Build features for the post-cutoff new batch with exactly the same code as the history.

History and new batch are concatenated *before* cleaning and feature building so the rolling
windows (20/50-day) at the start of July are warmed up with real June history, and the same
cleaning rules (USD/PKR bad ticks, ex-dividend gaps) apply to both.

Outputs
- psx_historical_data/combined_clean.csv          history + new batch, cleaned
- psx_historical_data/new_batch/features_new.parquet  feature rows dated after the cutoff
- psx_historical_data/combined_features.parquet   all labelled feature rows (history + new batch)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.clean_data import clean_macro_columns, flag_ex_dividend_gaps
from src.data.fetch_new_batch import HISTORY_PATH, NEW_BATCH_PATH
from src.features.build_features import FEATURE_COLUMNS, compute_features_and_labels

COMBINED_CLEAN_PATH = Path("psx_historical_data/combined_clean.csv")
NEW_FEATURES_PATH = Path("psx_historical_data/new_batch/features_new.parquet")
COMBINED_FEATURES_PATH = Path("psx_historical_data/combined_features.parquet")


def build_new_batch_features(
    history_path: Path = HISTORY_PATH,
    new_batch_path: Path = NEW_BATCH_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (new-batch feature rows, all labelled feature rows) and save both."""
    history = pd.read_csv(history_path, parse_dates=["date"])
    new = pd.read_csv(new_batch_path, parse_dates=["date"])
    cutoff = history["date"].max()

    combined = pd.concat([history, new], ignore_index=True)
    combined, _ = clean_macro_columns(combined)
    combined = flag_ex_dividend_gaps(combined)
    COMBINED_CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(COMBINED_CLEAN_PATH, index=False)

    feats = compute_features_and_labels(combined, drop_Incomplete=False)
    feats = feats.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)

    new_feats = feats[feats["date"] > cutoff].reset_index(drop=True)
    labelled = feats.dropna(subset=["target_class"]).copy()
    labelled["target_class"] = labelled["target_class"].astype("int64")

    NEW_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_feats.to_parquet(NEW_FEATURES_PATH, index=False)
    labelled.to_parquet(COMBINED_FEATURES_PATH, index=False)
    return new_feats, labelled


if __name__ == "__main__":
    nf, lab = build_new_batch_features()
    print(f"New-batch feature rows: {len(nf):,} ({nf['date'].min().date()} -> {nf['date'].max().date()}), "
          f"labelled: {int(nf['target_class'].notna().sum()):,}")
    print(f"Combined labelled rows: {len(lab):,} ({lab['date'].min().date()} -> {lab['date'].max().date()})")
