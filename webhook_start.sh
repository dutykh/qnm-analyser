#!/bin/bash
# QNM Analyser - webhook receiver start script
# Author: Dr. Denys Dutykh (https://www.denys-dutykh.com/)
set -euo pipefail
cd "$(dirname "$0")"

# Load env if present
set -a
[ -f .env ] && . ./.env
set +a

if [ ! -x venv/bin/gunicorn ]; then
  echo "ERROR: venv/bin/gunicorn missing. Run ./deploy.sh (or recreate the venv) first." >&2
  exit 127
fi

if [ -z "${WEBHOOK_SECRET:-}" ]; then
  echo "WARNING: WEBHOOK_SECRET is unset; every webhook request will be rejected." >&2
fi

ulimit -c 0

PORT="${WEBHOOK_PORT:-9050}"

# Bind localhost only (Traefik proxies to it). One worker keeps the deploy
# trigger single-threaded.
exec venv/bin/gunicorn --bind "127.0.0.1:${PORT}" --workers 1 webhook:app
