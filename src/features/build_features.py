"""Phase 2: Feature Engineering & Triple-Barrier 3-Class Labeling.

Reads `psx_historical_data/stocks_clean.csv` (output of `src.data.clean_data`, which repairs
USD/PKR bad ticks and flags ex-dividend gaps on top of `stocks_with_macro.csv`), computes strictly causal per-symbol
technical, macro, and seasonality features along with the dynamic volatility-adjusted
Triple-Barrier 3-class target (`target_class`), and writes `psx_historical_data/feature_store.parquet`.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

INPUT_CSV_PATH = Path("psx_historical_data/stocks_clean.csv")
OUTPUT_PARQUET_PATH = Path("psx_historical_data/feature_store.parquet")

# Trading days by which macro closes are lagged so only values known at the PSX close are used
MACRO_LAG_DAYS = 1

FEATURE_COLUMNS: list[str] = [
    # 1. Daily & lagged log returns
    "ret_1d",
    "ret_2d",
    "ret_3d",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    # 2. Volatility & technical indicators
    "vol_20d",
    "rsi_14",
    "macd_hist",
    "atr_14_pct",
    "bb_width_20",
    "close_to_sma20",
    "close_to_sma50",
    "vol_shock_20",
    # 3. Macro features (1d and 5d log returns)
    "brent_ret_1d",
    "brent_ret_5d",
    "usdpkr_ret_1d",
    "usdpkr_ret_5d",
    "dxy_ret_1d",
    "dxy_ret_5d",
    # 4. Seasonality features
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos",
    "is_earnings_month",
    "is_winter_gas",
    "is_monsoon",
]


def _compute_rsi_14(df: pd.DataFrame) -> pd.Series:
    """Compute 14-day RSI per symbol without ex-dividend distortion or look-ahead leakage."""
    prev_close = df.groupby("symbol", sort=False)["close"].shift(1)
    # On isolated ex-dividend days, use intraday change (close - open) to avoid fake dividend drops
    delta = np.where(
        df["is_ex_dividend_gap"] == 1,
        df["close"] - df["open"],
        df["close"] - prev_close,
    )
    delta_s = pd.Series(delta, index=df.index)
    gain = delta_s.clip(lower=0.0)
    loss = (-delta_s).clip(lower=0.0)

    avg_gain = (
        gain.groupby(df["symbol"], sort=False)
        .rolling(window=14, min_periods=14)
        .mean()
        .reset_index(level=0, drop=True)
    )
    avg_loss = (
        loss.groupby(df["symbol"], sort=False)
        .rolling(window=14, min_periods=14)
        .mean()
        .reset_index(level=0, drop=True)
    )

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # Handle edge cases where avg_loss == 0 or both == 0 after warmup
    rsi = np.where((avg_loss == 0.0) & (avg_gain > 0.0), 100.0, rsi)
    rsi = np.where((avg_loss == 0.0) & (avg_gain == 0.0), 50.0, rsi)
    return pd.Series(rsi, index=df.index, dtype="float64")


def _compute_macd_hist(df: pd.DataFrame) -> pd.Series:
    """Compute MACD(12, 26, 9) histogram per symbol."""
    def _symbol_macd_hist(close_s: pd.Series) -> pd.Series:
        ema_12 = close_s.ewm(span=12, adjust=False, min_periods=12).mean()
        ema_26 = close_s.ewm(span=26, adjust=False, min_periods=26).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
        return macd_line - signal_line

    return df.groupby("symbol", sort=False)["close"].transform(_symbol_macd_hist)


def _compute_atr_14_pct(df: pd.DataFrame) -> pd.Series:
    """Compute ATR_14 / close per symbol, guarding against ex-dividend overnight drops."""
    prev_close = df.groupby("symbol", sort=False)["close"].shift(1)
    hl = df["high"] - df["low"]
    hpc = (df["high"] - prev_close).abs()
    lpc = (df["low"] - prev_close).abs()

    tr = np.where(
        df["is_ex_dividend_gap"] == 1,
        hl,
        np.maximum(hl, np.maximum(hpc, lpc)),
    )
    tr_s = pd.Series(tr, index=df.index, dtype="float64")
    atr_14 = (
        tr_s.groupby(df["symbol"], sort=False)
        .rolling(window=14, min_periods=14)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return atr_14 / df["close"]


def compute_features_and_labels(
    df_raw: pd.DataFrame,
    drop_Incomplete: bool = True,
) -> pd.DataFrame:
    """Compute all Phase 2 features and Triple-Barrier 3-class labels per symbol.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Input DataFrame with columns from `stocks_with_macro.csv`.
    drop_Incomplete : bool, default True
        If True, drops rows with NaN in any engineered feature (rolling warmup)
        or in `next_ret_1d` (final row per symbol).

    Returns
    -------
    pd.DataFrame
        DataFrame with all engineered features, `next_ret_1d`, `B_t`, and `target_class`.
    """
    df = df_raw.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Enforce strict chronological ordering within each symbol (Never shuffle)
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    g = df.groupby("symbol", sort=False)

    # 1. Daily Log Return (ret_1d)
    # np.log(close_t / close_{t-1}) per symbol; if is_ex_dividend_gap == 1, use np.log(close_t / open_t)
    prev_close = g["close"].shift(1)
    standard_ret_1d = np.log(df["close"] / prev_close)
    intraday_ret_1d = np.log(df["close"] / df["open"])
    df["ret_1d"] = np.where(
        df["is_ex_dividend_gap"] == 1,
        intraday_ret_1d,
        standard_ret_1d,
    )

    # 2. Technical Indicators (per symbol)
    # Multi-day log returns computed via rolling sum of ex-dividend-adjusted ret_1d per symbol
    ret_1d_grp = df.groupby("symbol", sort=False)["ret_1d"]
    for horizon in (2, 3, 5, 10, 20):
        df[f"ret_{horizon}d"] = (
            ret_1d_grp.rolling(window=horizon, min_periods=horizon)
            .sum()
            .reset_index(level=0, drop=True)
        )

    # Rolling volatility: 20-day rolling std of ret_1d per symbol
    df["vol_20d"] = (
        ret_1d_grp.rolling(window=20, min_periods=20)
        .std()
        .reset_index(level=0, drop=True)
    )

    # RSI(14), MACD histogram, ATR(14)/close
    df["rsi_14"] = _compute_rsi_14(df)
    df["macd_hist"] = _compute_macd_hist(df)
    df["atr_14_pct"] = _compute_atr_14_pct(df)

    # Bollinger Band width (20-day, 2 std), close_to_sma20, close_to_sma50, vol_shock_20
    sma_20 = g["close"].rolling(window=20, min_periods=20).mean().reset_index(level=0, drop=True)
    std_20 = g["close"].rolling(window=20, min_periods=20).std().reset_index(level=0, drop=True)
    sma_50 = g["close"].rolling(window=50, min_periods=50).mean().reset_index(level=0, drop=True)
    vol_sma_20 = g["volume"].rolling(window=20, min_periods=20).mean().reset_index(level=0, drop=True)

    df["bb_width_20"] = (4.0 * std_20) / sma_20
    df["close_to_sma20"] = df["close"] / sma_20
    df["close_to_sma50"] = df["close"] / sma_50
    df["vol_shock_20"] = df["volume"] / vol_sma_20.replace(0.0, np.nan)

    # 3. Macro Features: 1-day and 5-day log returns of brent_close, usdpkr_close, dxy_close per symbol.
    # Macro closes are lagged by one trading day (MACRO_LAG_DAYS): Brent/DXY settle in New York
    # (~21:00 UTC) and USD/PKR's daily close is also after the PSX close (~10:30 UTC), so the
    # value stamped on date t is not yet known when the PSX day-t close is observed.
    for macro_prefix, col_name in (
        ("brent", "brent_close"),
        ("usdpkr", "usdpkr_close"),
        ("dxy", "dxy_close"),
    ):
        known = g[col_name].shift(MACRO_LAG_DAYS)
        prev_1 = g[col_name].shift(MACRO_LAG_DAYS + 1)
        prev_5 = g[col_name].shift(MACRO_LAG_DAYS + 5)
        df[f"{macro_prefix}_ret_1d"] = np.log(known / prev_1)
        df[f"{macro_prefix}_ret_5d"] = np.log(known / prev_5)

    # 4. Seasonality Features
    month = df["date"].dt.month
    dow = df["date"].dt.day_of_week
    df["month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
    df["month_cos"] = np.cos(2.0 * np.pi * month / 12.0)
    df["dow_sin"] = np.sin(2.0 * np.pi * dow / 5.0)
    df["dow_cos"] = np.cos(2.0 * np.pi * dow / 5.0)
    df["is_earnings_month"] = month.isin([1, 4, 7, 10]).astype("int64")
    df["is_winter_gas"] = month.isin([11, 12, 1, 2]).astype("int64")
    df["is_monsoon"] = month.isin([7, 8]).astype("int64")

    # 5. Triple-Barrier 3-Class Target (target_class)
    df["next_ret_1d"] = df.groupby("symbol", sort=False)["ret_1d"].shift(-1)
    df["B_t"] = np.maximum(0.0075, 0.60 * df["vol_20d"])

    target = np.where(
        df["next_ret_1d"].isna() | df["B_t"].isna(),
        np.nan,
        np.where(
            df["next_ret_1d"] > df["B_t"],
            2,
            np.where(df["next_ret_1d"] < -df["B_t"], 0, 1),
        ),
    )
    df["target_class"] = pd.Series(target, index=df.index)

    if drop_Incomplete:
        required_cols = FEATURE_COLUMNS + ["next_ret_1d", "B_t", "target_class"]
        df = df.dropna(subset=required_cols).reset_index(drop=True)
        df["target_class"] = df["target_class"].astype("int64")

    return df


def build_feature_store(
    input_csv: Path = INPUT_CSV_PATH,
    output_parquet: Path = OUTPUT_PARQUET_PATH,
) -> pd.DataFrame:
    """Read `stocks_with_macro.csv`, engineer features & labels, and save `feature_store.parquet`."""
    df_raw = pd.read_csv(input_csv, parse_dates=["date"])
    df_features = compute_features_and_labels(df_raw, drop_Incomplete=True)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_parquet(output_parquet, index=False)
    return df_features


if __name__ == "__main__":
    df_out = build_feature_store()
    counts = df_out["target_class"].value_counts().sort_index()
    pcts = (df_out["target_class"].value_counts(normalize=True).sort_index() * 100).round(2)
    label_names = {0: "DOWN (0)", 1: "NEUTRAL (1)", 2: "UP (2)"}
    print(f"Saved feature store to: {OUTPUT_PARQUET_PATH}")
    print(f"Total clean rows: {len(df_out):,} across {df_out['symbol'].nunique()} symbols")
    print(f"Date range: {df_out['date'].min().date()} to {df_out['date'].max().date()}")
    print("\nTriple-Barrier 3-Class Distribution:")
    for cls_id in (0, 1, 2):
        print(f"  {label_names[cls_id]:<12}: {counts.get(cls_id, 0):>6,} rows ({pcts.get(cls_id, 0.0):>5.2f}%)")
