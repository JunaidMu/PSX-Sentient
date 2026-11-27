#!/usr/bin/env sh
# First-time setup of a server (run inside the trainer container):
#   docker compose run --rm trainer sh scripts/bootstrap.sh
# 1. pull versioned data from DagsHub (DAGSHUB_USER / DAGSHUB_TOKEN from .env)
# 2. run the D1 pipeline -> trains + registers the champion in this server's MLflow
# 3. fetch the post-cutoff batch, build its features, train the stacking meta-learner
set -e
export PYTHONPATH=/app

if [ -n "$DAGSHUB_USER" ] && [ -n "$DAGSHUB_TOKEN" ]; then
  dvc remote modify origin --local auth basic
  dvc remote modify origin --local user "$DAGSHUB_USER"
  dvc remote modify origin --local password "$DAGSHUB_TOKEN"
fi

echo "==> dvc pull (raw data + new batch)"
dvc pull psx_historical_data/stocks_with_macro.csv.dvc
dvc pull psx_historical_data/new_batch/stocks_with_macro_new.csv.dvc || \
  python -m src.data.fetch_new_batch

echo "==> D1 pipeline: validate -> clean -> features -> validate -> baselines -> train/register"
dvc repro --force

echo "==> new-batch features"
python -m src.monitoring.batch

echo "==> done. Start the stack with: docker compose up -d"
