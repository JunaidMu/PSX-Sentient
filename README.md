# PSX-Sentient: Stock Movement & Sentiment Engine for the Pakistan Stock Exchange

**PSX-Sentient** is an MLOps pipeline that predicts the next-day move — **DOWN (0) / NEUTRAL (1) / UP (2)** — of
**30 KSE-100 stocks** on the Pakistan Stock Exchange (PSX), using split-adjusted prices, technical indicators,
macro data (Brent crude, USD/PKR, DXY) and calendar seasonality. Historical data covers **2017-01-02 → 2026-06-30**
(70,474 rows), sourced from the [psxdata](https://github.com/psxdata/psxdata-api) API, Yahoo Finance and FRED.

---

## Deliverable 1 — Reproducible Training & Experimentation Pipeline

### Reproduce everything with one command

```bash
pip install -r requirements-train.txt
export DAGSHUB_USER=<your DagsHub username> DAGSHUB_TOKEN=<your DagsHub token>
sh scripts/reproduce.sh          # dvc pull  ->  dvc repro --force  ->  dvc metrics show
```

or with Docker (no local Python needed):

```bash
docker build -f Dockerfile.train -t psx-sentient-train .
docker run --rm -e DAGSHUB_USER=... -e DAGSHUB_TOKEN=... -v "$PWD/mlruns:/app/mlruns" psx-sentient-train
```

Re-running the pipeline produces **identical metrics** (fixed seeds, deterministic LightGBM, no shuffling).

### Pipeline stages (`dvc.yaml`)

```
stocks_with_macro.csv  (raw, DVC-tracked, stored on the DagsHub remote)
   │
   ├─ validate_raw       18 schema & quality checks  -> reports/validation_raw.json      (stops the pipeline on failure)
   ├─ clean              USD/PKR bad-tick repair + ex-dividend gap flags -> stocks_clean.csv
   ├─ features           27 stationary features + triple-barrier labels   -> feature_store.parquet
   ├─ validate_features  9 checks (NaN/inf, labels, class balance, bounds) -> reports/validation_features.json
   ├─ baselines          AlwaysNeutral rule, Momentum rule, Logistic Regression -> models/baseline_metrics.json
   └─ train              LightGBM vs CatBoost, purged walk-forward, MLflow tracking + Model Registry
                         -> models/best_model.pkl, models/model_comparison_metrics.json
```

| Requirement | Where |
| :--- | :--- |
| Versioned raw + processed data | `*.dvc` files, `dvc.yaml`, `dvc.lock`, `.dvc/config` (DagsHub remote) |
| Schema & quality checks before training | `src/data/validate.py`, stages `validate_raw` and `validate_features` |
| Leakage-free time-ordered splits | `generate_purged_walk_forward_splits` in `src/models/train_baseline.py` |
| Baselines before the primary model | `src/models/baselines.py`, stage `baselines` |
| Experiment tracking (params, dataset versions, metrics) | MLflow, `mlruns/mlflow.db`, `src/models/tracking.py` |
| Registered model | MLflow Model Registry: `psx-sentient-direction`, alias `champion` |
| Reproducible execution | `scripts/reproduce.sh`, `Dockerfile.train` |

### Leakage controls

- **Time-ordered splits only.** Three expanding walk-forward folds (validate on 2023, 2024, 2025) and a final
  holdout **Test_2026 (Jan–Jun 2026)**, each with a **5-trading-day purge gap** between training and evaluation.
- **Early stopping never sees the evaluation data.** Every fold early-stops on the last 60 trading days of its
  own training window (with another 5-day purge), never on the fold being scored.
- **Model selection uses validation only.** The best model is picked on the walk-forward validation mean;
  Test_2026 is only reported.
- **Macro timing.** Brent, DXY and USD/PKR closes settle after the PSX close, so macro features use the
  previous day's values.
- **Per-symbol computation.** All lags, rolling windows and target shifts are grouped by symbol (tested for
  look-ahead and cross-symbol leakage in `tests/test_features.py`).

### Data cleaning (`src/data/clean_data.py`)

- **Splits & bonus issues** (e.g. MARI 1→9, LUCK 1→5, UBL/BAFL/MTL 1→2, plus 15 bonus issues) are
  back-adjusted in Phase 1 (`adjust_corporate_actions.py`); `validate_raw` fails if any >60% unadjusted
  jump remains.
- **USD/PKR bad ticks:** 8 one-day spikes from Yahoo Finance that fully reverse the next day
  (e.g. 239.9 → 206.75 → 239.1 on 2022-08-01) are replaced with the previous value.
- **Ex-dividend days:** a stock gapping down ≥3% overnight more than the market median is flagged
  (19 → 526 flags) so dividend drops are not labelled as DOWN moves. *Limitation:* this is a heuristic,
  not real dividend history.

### Results (Test_2026 = Jan–Jun 2026, 3,570 stock-days, never used for model selection)

| Model | Type | Val mean Macro F1 | Val mean Active Precision | **Test Macro F1** | **Test Active Precision** |
| :--- | :--- | ---: | ---: | ---: | ---: |
| AlwaysNeutral | rule baseline | 0.2398 | – | 0.2341 | – |
| MomentumRule | rule baseline | 0.3583 | 24.87% | 0.3593 | 25.29% |
| LogisticRegression | classical baseline | 0.3766 | 25.84% | 0.3817 | 27.51% |
| LightGBM | primary | 0.3843 | 26.47% | 0.3848 | 26.62% |
| **CatBoost** (registered champion) | primary | **0.3870** | **27.28%** | **0.3972** | **28.53%** |

*Active precision* = how often an UP or DOWN call is correct (random would be ~23%).
Class balance: DOWN 23.4%, NEUTRAL 53.4%, UP 23.2%.

> **Note on earlier numbers.** A previous version reported Test Macro F1 0.4282 / 32.65%. Those runs picked the
> model using the test set, early-stopped on the validation folds and used same-day macro closes; the numbers
> above are the corrected, leakage-free results.

---

## Later phases (Deliverable 2, in progress)

- `src/nlp/` — news scraping (Business Recorder, Google News, PSX notices), entity mapping to the 30 stocks,
  CPU FinBERT sentiment with 3-day EWMA.
- `src/fusion/late_fusion.py` — conflict-veto / blended-conviction rules on top of model probabilities.
  **Caveat:** the Test_2026 ablation in `models/ablation_metrics.json` uses *simulated* sentiment (template
  headlines assigned from price patterns), because no historical news archive exists for 2017–2026. It is not
  evidence that news improves predictions; real news is collected going forward.
- `src/data/live_updater.py`, `src/api/` — FastAPI service and Streamlit dashboard. The current
  `models/latest_signals.json` is a demo snapshot (hand-picked dates, sample off-market data), to be replaced
  by a daily data collector.

### Run tests

```bash
pytest -v
```

### Serve locally

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
uvicorn src.api.main:app --port 8000          # http://localhost:8000/docs
streamlit run src/api/app.py --server.port 8501
```
