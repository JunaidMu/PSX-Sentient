#!/usr/bin/env sh
# Daily pre-market run, 09:00 PKT (04:00 UTC) Monday-Friday, 30 minutes before the PSX open.
#   docker compose run --rm trainer sh scripts/premarket.sh
# EC2 crontab:  0 4 * * 1-5 cd ~/PSX-Sentient && docker compose run --rm trainer sh scripts/premarket.sh >> logs/premarket.log 2>&1
set -u
export PYTHONPATH="${PYTHONPATH:-/app}"
git config --global --add safe.directory "$(pwd)" 2>/dev/null

echo "==> $(date -u) pre-market run"
echo "==> 1. latest prices + macro (psxdata API, Yahoo/FRED) and features"
python -m src.data.fetch_new_batch || echo "price fetch failed, keeping the previous batch"
python -m src.monitoring.batch

echo "==> 2. PSX daily files + overnight news -> FinBERT -> snapshot"
python -m src.data.premarket

echo "==> 3. version today's data: DVC -> DagsHub, pointers -> git 'data' branch"
dvc add psx_historical_data/daily psx_historical_data/news psx_historical_data/new_batch/stocks_with_macro_new.csv \
  && dvc push || echo "DVC versioning skipped (no remote credentials?)"
sh scripts/commit_data_pointers.sh

echo "==> 4. daily drift check (retrains only for urgent drift; routine retraining is weekly)"
python -m src.monitoring.loop --mode daily --skip-fetch || echo "loop failed"

echo "==> 5. back up the MLflow registry and today's data-version snapshot to S3"
python -m src.monitoring.backup || echo "backup failed"
echo "==> done"
