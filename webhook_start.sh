#!/bin/bash
cd "$(dirname "$0")"

# Load env if present
set -a
[ -f .env ] && . ./.env
set +a

source venv/bin/activate

PORT="${WEBHOOK_PORT:-9050}"

# Bind localhost only (Traefik can proxy to it)
exec gunicorn --bind 127.0.0.1:${PORT} --workers 1 webhook:app
