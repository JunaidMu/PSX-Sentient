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

## Deliverable 2 — End-to-End Production & Monitoring System

**Live on AWS (EC2, ap-southeast-2):** [web app](http://3.104.86.72) · [classic Streamlit dashboard](http://3.104.86.72:8501) · [API docs](http://3.104.86.72:8000/docs) · [Grafana](http://3.104.86.72:3000/d/psx-sentient) · [MLflow](http://3.104.86.72:5000) · [Prometheus](http://3.104.86.72:9090)

```
psxdata API + Yahoo/FRED ──► fetch_new_batch (Jul–Nov 2026, DVC data v2) ──► same cleaning + features
                                                                                   │
   ┌──────────────── docker compose on AWS EC2 ───────────────────────────────────┼──────────────┐
   │  FastAPI :8000  ◄── champion from MLflow registry (:5000, artifacts in S3)    │              │
   │   /predict  probabilities · confidence · SHAP · news-sentiment veto           ▼              │
   │   /metrics ──► Prometheus :9090 ──► Grafana :3000 (latency, errors, drift, alerts)            │
   │  Streamlit :8501 (signals, predict & explain, monitoring)                                     │
   │  trainer job: replay traffic → PSI/KS drift → Optuna challenger → compare → promote/rollback  │
   └──────────────────────── container logs ──► CloudWatch Logs ─────────────────────────────────┘
GitHub Actions: unit tests → DVC pull + validation + retrain → evaluation gate (Test_2026) → deploy to EC2
```

| Requirement | Implementation |
| :--- | :--- |
| Serving API + UI | `src/api/main.py` (FastAPI), `src/api/app.py` (Streamlit); live inputs = stock/date or an editable 27-feature vector |
| Explanations | Per-prediction SHAP (native CatBoost/LightGBM), class probabilities and confidence |
| Late fusion (proposal) | `src/models/stacking.py` meta-learner over LightGBM + CatBoost probabilities; conflict-veto gate on live FinBERT/entered sentiment |
| Eval-gated CI/CD | `.github/workflows/ci-cd.yml` + `src/monitoring/gate.py`: blocks deploy if Test_2026 metrics fall below `models/production_benchmark.json` |
| Live observability | Prometheus metrics (latency, error rate, prediction mix, confidence, PSI per feature), Grafana dashboard + alert rules (`monitoring/`), CloudWatch logs |
| Drift detection | `src/monitoring/drift.py`: PSI + KS per feature and predicted-class PSI, on the new batch or on served inputs |
| Closed-loop retraining | `src/monitoring/loop.py` / `retrain.py`: weekly (Saturday) retraining when drift vs. the champion's own training data fires or after ~4 weeks of new data, urgent mid-week retrain for severe drift, one-week cooldown; Optuna-tuned challenger trained on everything up to ~6 weeks ago, compared with the champion on the latest ~6 weeks → promote, `previous_champion` kept for `--rollback` |
| Backups & data versions | `src/monitoring/backup.py`: daily MLflow registry backup + data-version snapshot to S3; `scripts/commit_data_pointers.sh`: DVC pointers committed to the git `data` branch; Phase-1 download scripts in `scripts/phase1/` |
| Deployment | `Dockerfile`, `docker-compose.yml`, `docker-compose.aws.yml`, `deploy/AWS_DEPLOY.md` |

### Results on the new batch (Jul–Nov 2026)

- **Drift:** 10 of 20 market features drifted (volatility: ATR, Bollinger width, 20-day vol; Brent and USD/PKR returns) → retraining triggered.
- **Retraining:** on 2026-10-01 – 2026-11-24 (unseen by both), champion (trained to 2025) Macro F1 **0.377** / active precision 24.3%
  vs Optuna-tuned challenger (trained to 2026-09-23) **0.393** / 25.9% → challenger promoted, API hot-reloaded; `--rollback` restores the previous champion.
- **Stacking meta-learner:** Test_2026 Macro F1 0.397 / active precision 28.7% vs CatBoost 0.397 / 28.5% (leave-one-year-out validation 0.381 vs CatBoost 0.387) — no meaningful gain yet, so the single CatBoost champion is served.

### Pre-market job (09:00 PKT, 30 minutes before the open) — `scripts/premarket.sh`

1. Latest prices + macro → features (`src/data/fetch_new_batch.py`, `src/monitoring/batch.py`)
2. PSX Data Portal daily files (`src/data/psx_files.py`): Market Summary (closing prices), Off-Market Transaction
   Summary (block deals → whale score W_ndm), futures open interest (DFC), corporate announcements (results,
   dividends, book closures → factual headlines) and short-sell volume
3. Overnight news (Business Recorder, Google News, PSX notices) + announcements → CPU FinBERT + PSX shock
   keywords → news archive → per-stock 3-day EWMA sentiment S_t (`src/data/premarket.py`)
4. Snapshot used by the API: S_t feeds the conflict veto automatically; whales, futures OI, headlines and
   announcements are shown next to each prediction. Raw files and the news archive are versioned with DVC.
5. Drift check → retraining loop if the drift rule fires.

These sources are **context, not model inputs** — there is no history to train on yet; the daily archive builds it.

### Honest limitations

- **News sentiment and the veto run live only.** No 2017–2026 news archive exists, so their effect on accuracy cannot be
  measured yet; the old 69.77% "fusion" figure in `models/ablation_metrics.json` came from simulated sentiment and is not a result.
- Dividend days are detected heuristically, not from real dividend history.
- Off-market (NDM), futures, short-sell and news data are not model inputs (no training history yet).
- The target is close-to-close, but a 09:00 PKT signal can only be acted on at the 09:30 open, so part of the
  predicted move (the overnight gap) is not capturable; an open-to-close target is future work.

### Run locally

```bash
pip install -r requirements.txt                      # + requirements-nlp.txt for FinBERT
pytest -q
uvicorn src.api.main:app --port 8000                 # http://localhost:8000/docs
API_URL=http://localhost:8000 streamlit run src/api/app.py
python -m src.monitoring.loop --skip-fetch           # drift -> retrain -> promote
```
