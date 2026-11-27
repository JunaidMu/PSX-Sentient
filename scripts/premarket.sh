#!/usr/bin/env sh
# Daily pre-market run, 09:00 PKT (04:00 UTC) Monday-Friday, 30 minutes before the PSX open.
#   docker compose run --rm trainer sh scripts/premarket.sh
# EC2 crontab:  0 4 * * 1-5 cd ~/PSX-Sentient && docker compose run --rm trainer sh scripts/premarket.sh >> logs/premarket.log 2>&1
set -u
export PYTHONPATH="${PYTHONPATH:-/app}"

echo "==> $(date -u) pre-market run"
echo "==> 1. latest prices + macro (psxdata API, Yahoo/FRED) and features"
python -m src.data.fetch_new_batch || echo "price fetch failed, keeping the previous batch"
python -m src.monitoring.batch

echo "==> 2. PSX daily files + overnight news -> FinBERT -> snapshot"
python -m src.data.premarket

echo "==> 3. version today's raw files and the news archive with DVC"
dvc add psx_historical_data/daily psx_historical_data/news psx_historical_data/new_batch/stocks_with_macro_new.csv \
  && dvc push || echo "DVC versioning skipped (no remote credentials?)"

echo "==> 4. drift check on the latest batch (retrains and promotes only if the drift rule fires)"
python -m src.monitoring.loop --skip-fetch || echo "loop failed"
echo "==> done"
