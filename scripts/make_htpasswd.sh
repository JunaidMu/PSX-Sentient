#!/usr/bin/env sh
# Create deploy/htpasswd (login for MLflow :5000 and Prometheus :9090 behind nginx) from .env:
#   ADMIN_USER=admin
#   ADMIN_PASSWORD=<strong password>
# Run once on each machine before `docker compose up` (the deploy job runs it if the file is missing).
set -e
cd "$(dirname "$0")/.."
USER_NAME="$(grep -E '^ADMIN_USER=' .env 2>/dev/null | cut -d= -f2-)"
PASS="$(grep -E '^ADMIN_PASSWORD=' .env 2>/dev/null | cut -d= -f2-)"
USER_NAME="${USER_NAME:-admin}"
if [ -z "$PASS" ]; then
  echo "ADMIN_PASSWORD is missing from .env" >&2
  exit 1
fi
printf '%s:%s\n' "$USER_NAME" "$(openssl passwd -apr1 "$PASS")" > deploy/htpasswd
chmod 644 deploy/htpasswd
echo "deploy/htpasswd written for user '$USER_NAME'"
