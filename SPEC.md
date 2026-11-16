# PSX-Sentient: Master Project Specification for Antigravity

## 1. Current State (Phase 1 Completed)
The human supervisor has already completed Phase 1. 
The clean dataset is located at: `psx_historical_data/stocks_with_macro.csv` (70,474 rows, 30 KSE-100 stocks from 2017-01-02 to 2026-06-30).
Columns in `stocks_with_macro.csv`:
`date, symbol, open, high, low, close, volume, original_open, original_high, original_low, original_close, original_volume, adjustment_factor, overnight_gap_pct, is_ex_dividend_gap, brent_close, usdpkr_close, dxy_close`

DO NOT re-download or overwrite `stocks_with_macro.csv`.

## 2. Strict Engineering Rules
1. Work one Phase at a time. Write unit tests (`pytest`) for each Phase and run them before moving to the next Phase.
2. Zero data leakage: Always group by `symbol` (`df.groupby("symbol")`) before computing lags, rolling windows, or target shifts. Never use `shuffle=True`.
3. Run FinBERT (`ProsusAI/finbert`) strictly on CPU (`device="cpu"`).

## 3. Implementation Phases for Antigravity to Build

### PHASE 2 — Feature Engineering (`src/features/`)
Read `psx_historical_data/stocks_with_macro.csv` and generate `psx_historical_data/feature_store.parquet`:
1. **Daily Log Return (`ret_1d`):** `np.log(close_t / close_{t-1})` per symbol. If `is_ex_dividend_gap == 1`, use intraday return `np.log(close_t / open_t)` so dividend drops do not distort features.
2. **Technical Indicators (per symbol):**
   - Lagged log returns: `ret_2d`, `ret_3d`, `ret_5d`, `ret_10d`, `ret_20d`
   - Rolling volatility: `vol_20d` = 20-day rolling std of `ret_1d`
   - `rsi_14`, `macd_hist`, `atr_14_pct` (`ATR_14 / close`), `bb_width_20`, `close_to_sma20`, `close_to_sma50`, `vol_shock_20` (`volume / SMA_20(volume)`)
3. **Macro Features:** 1-day and 5-day log returns of `brent_close`, `usdpkr_close`, and `dxy_close`.
4. **Optional Seasonality Features:** `month_sin`, `month_cos`, `dow_sin`, `dow_cos`, plus binary flags for `is_earnings_month` (`[1,4,7,10]`), `is_winter_gas` (`[11,12,1,2]`), and `is_monsoon` (`[7,8]`).
5. **Triple-Barrier 3-Class Target (`target_class`):**
   - `next_ret_1d` = `df.groupby("symbol")["ret_1d"].shift(-1)`
   - Barrier `B_t = np.maximum(0.0075, 0.60 * vol_20d)`
   - Label `2` (UP) if `next_ret_1d > B_t`; `0` (DOWN) if `next_ret_1d < -B_t`; `1` (NEUTRAL) otherwise.
   - Write unit tests in `tests/test_features.py` verifying zero look-ahead leakage and class balance.

### PHASE 3 — Baseline ML (`src/models/`)
1. Implement Purged Walk-Forward validation across dates (5-day purge gap between train and val folds; final test holdout = `2026-01-01` to `2026-06-30`).
2. Train and compare `LightGBM` (`objective="multiclass"`, `num_class=3`) and `CatBoost` (`loss_function="MultiClass"`).
3. Log `macro_f1` and `active_signal_precision` (precision on classes 0 and 2) to `MLflow`.
4. Output calibrated probabilities `[P_down, P_neutral, P_up]` and Net Technical Edge `E_tech = P_up - P_down`.

### PHASE 4 — NLP Pipeline (`src/nlp/`)
1. Scrape daily news from Business Recorder RSS (`https://www.brecorder.com/feeds/latest-news`), Google News PSX RSS, and PSX notices (`https://dps.psx.com.pk/announcements/companies`).
2. Map headlines to the 30 stock symbols or `"MACRO"`. Deduplicate titles.
3. Score headlines on CPU using `ProsusAI/finbert` (`P(pos) - P(neg)`) and compute 3-day EWMA daily sentiment score `S_t` in `[-1, 1]`.

### PHASE 5 — Late-Fusion Engine (`src/fusion/`)
1. Conflict Veto Gate: If `(E_tech > 0.20 and S_t < -0.50)` or `(E_tech < -0.20 and S_t > 0.50)`, force signal to `"NEUTRAL"`.
2. Blended Score: `C_t = 0.75 * E_tech + 0.25 * S_t`. Output `"UP"` if `C_t > 0.20` and `P_neutral < 0.50`; `"DOWN"` if `C_t < -0.20` and `P_neutral < 0.50`; else `"NEUTRAL"`.
3. Run an ablation script comparing Tabular-Only vs. Late-Fusion.

### PHASE 6 & 7 — MLOps & Deployment (`src/api/`, Docker, CI/CD)
1. Configure DVC, FastAPI (`src/api/main.py`), and Streamlit (`src/api/app.py`).
2. Create `Dockerfile`, `docker-compose.yml`, and `.github/workflows/daily_pipeline.yml`.