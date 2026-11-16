# PSX-Sentient: Regime-Adaptive Late-Fusion Stock Movement & Sentiment Engine

**PSX-Sentient** is an end-to-end MLOps pipeline for **30 KSE-100 equities** on the **Pakistan Stock Exchange (PSX)** (`2017-01-02` to `2026-06-30`). It fuses calibrated **CatBoost / LightGBM** probabilities trained on stationary technical, macro (`Brent`, `USD/PKR`, `DXY`), and PSX seasonality indicators with a **CPU-only `ProsusAI/finbert` NLP Engine**, a **Hard Conflict Veto Gate**, and an **Off-Market (NDM) Institutional Whale Block Detector**.

---

## 1. Architecture & Pipeline Phases

```
psx_historical_data/stocks_with_macro.csv (70,474 rows, 30 KSE-100 stocks)
   │
   ├──► Phase 2: Feature Engineering & Triple-Barrier Labeling (src/features/build_features.py)
   │       • 27 stationary technical, macro return, and seasonality features
   │       • Ex-dividend intraday return adjustment (is_ex_dividend_gap == 1)
   │       • Volatility-scaled Triple-Barrier target: B_t = max(0.0075, 0.60 * vol_20d)
   │       • Output: psx_historical_data/feature_store.parquet (68,974 clean rows)
   │
   ├──► Phase 3: Purged Walk-Forward Baseline ML & MLflow (src/models/train_baseline.py)
   │       • 3 expanding validation folds (Val_2023, Val_2024, Val_2025) + Test_2026 Holdout
   │       • Strict 5-trading-day purge gap after each training window
   │       • Balanced class weights (CatBoost vs. LightGBM) -> Best Model: CatBoost
   │       • Outputs: models/best_model.pkl, models/model_comparison_metrics.json
   │
   ├──► Phase 4: NLP Pipeline & CPU FinBERT (src/nlp/)
   │       • Business Recorder RSS, Google News PSX RSS, and PSX Notices scraper
   │       • 30-stock Entity Mapper + MD5 title deduplication
   │       • ProsusAI/finbert strictly on CPU (device="cpu") + PSX corporate shock multipliers
   │       • 3-day EWMA daily sentiment score S_t in [-1, 1]
   │
   ├──► Phase 5: Late-Fusion Engine & Conflict Veto Gate (src/fusion/late_fusion.py)
   │       • Conflict Veto Gate: If (E_tech > 0.20 & S_t < -0.50) or (E_tech < -0.20 & S_t > 0.50) -> NEUTRAL
   │       • Blended Score: C_t = 0.75 * E_tech + 0.25 * S_t (Threshold |C_t| > 0.20 & P_neutral < 0.50)
   │       • Test_2026 Active Signal Precision lifted from 32.65% (Argmax) -> 69.77% (Late-Fusion)
   │
   └──► Phase 6: Live Updater, NDM Whale Parser, FastAPI, Streamlit, Docker & CI/CD
           • Off-Market Whale Block Detector (W_ndm) in src/data/live_updater.py
           • FastAPI Service (src/api/main.py) & Streamlit Dashboard (src/api/app.py)
           • DVC (dvc.yaml), Docker (Dockerfile, docker-compose.yml), GitHub Actions (.github/workflows/)
```

---

## 2. Key Experimental Results

### Phase 3: Purged Walk-Forward (`2023–2025`) & Holdout (`Test_2026`)

| Model | Split | Macro F1 | Active Signal Precision | Active Signal Rate | P(Down) | P(Neutral) | P(Up) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **LightGBM** | `Val_Mean (2023-25)` | 0.3903 | 26.75% | 51.50% | 0.2526 | 0.6423 | 0.2868 |
| **LightGBM** | `Test_2026` | 0.4204 | 31.57% | 49.86% | 0.3260 | 0.6274 | 0.3059 |
| **CatBoost** | `Val_Mean (2023-25)` | **0.3923** | **27.18%** | 51.03% | 0.2522 | 0.6393 | **0.2920** |
| **CatBoost** | **`Test_2026`** | **0.4282** | **32.65%** | 57.48% | **0.3721** | **0.6318** | 0.2942 |

### Phase 5: `Test_2026` Ablation Study (`3,570` Stock-Days)

| Strategy | Macro F1 | Active Signal Precision | Active Signal Rate | Active Signals | Conflict Vetoes | P(Down) | P(Up) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **(a) Tabular Argmax Baseline** | **0.4282** | 32.65% | 57.48% | 2,052 | 0 | 37.21% | 29.42% |
| **(b) Conviction-Thresholded Tabular** | 0.3032 | 54.75% | 5.01% | 179 | 0 | 70.25% | 22.41% |
| **(c) Full Late-Fusion + Conflict Veto** | 0.3008 | **69.77%** | 3.61% | 129 | **71** | **73.39%** | **50.00%** |

---

## 3. Local Quickstart

### 1. Install Dependencies
```bash
python -m venv venv
.\venv\Scripts\activate
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
```

### 2. Run the Full Reproducible Pipeline
```bash
python -m src.features.build_features
python -m src.models.train_baseline
python -m src.nlp.finbert_engine
python -m src.fusion.late_fusion
python -m src.data.live_updater
```

### 3. Run Unit Tests (`pytest`)
```bash
pytest -v
```

### 4. Launch FastAPI & Streamlit Locally
```bash
# Terminal 1: FastAPI Server (http://localhost:8000/docs)
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Terminal 2: Streamlit Institutional Dashboard (http://localhost:8501)
streamlit run src/api/app.py --server.port 8501
```

### 5. Launch via Docker Compose
```bash
docker compose up --build
```
