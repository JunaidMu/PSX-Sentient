#!/usr/bin/env sh
# Weekly retraining check, Saturday 07:00 PKT (02:00 UTC), when the market is closed.
#   docker compose run --rm trainer sh scripts/weekly_retrain.sh
# EC2 crontab:  0 2 * * 6 cd ~/PSX-Sentient && docker compose run --rm trainer sh scripts/weekly_retrain.sh >> logs/weekly_retrain.log 2>&1
#
# Retrains when the drift rule fires (vs. the current champion's training data) or after ~4 weeks of new
# data, never more than once per trading week; promotes the challenger only if it beats the champion on
# the most recent weeks, then backs up the registry.
set -u
export PYTHONPATH="${PYTHONPATH:-/app}"
echo "==> $(date -u) weekly retraining check"
python -m src.monitoring.loop --mode weekly --skip-fetch || echo "loop failed"
python -m src.monitoring.backup || echo "backup failed"
echo "==> done"
