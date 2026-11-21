#!/usr/bin/env sh
# One command to reproduce the registered metrics:  sh scripts/reproduce.sh
# Requires DagsHub read credentials in DAGSHUB_USER / DAGSHUB_TOKEN (any DagsHub account token works
# for a public repo). Credentials are written to .dvc/config.local, which is never committed.
set -e

export PYTHONPATH="${PYTHONPATH:-.}"
export MLFLOW_DISABLE_AGENT_HINT=1

if [ -n "$DAGSHUB_USER" ] && [ -n "$DAGSHUB_TOKEN" ]; then
  dvc remote modify origin --local auth basic
  dvc remote modify origin --local user "$DAGSHUB_USER"
  dvc remote modify origin --local password "$DAGSHUB_TOKEN"
fi

echo "==> Pulling versioned data from the DVC remote"
dvc pull psx_historical_data/stocks_with_macro.csv.dvc

echo "==> Running the pipeline (validate -> clean -> features -> validate -> baselines -> train)"
dvc repro --force

echo "==> Metrics"
dvc metrics show
