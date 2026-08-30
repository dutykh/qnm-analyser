#!/bin/bash
# QNM Analyser - Auto-deployment script
# Triggered by GitHub webhook on push
set -euo pipefail

APP_DIR="${APP_DIR:-/home/dds/www/qnm-analyser}"
VENV_DIR="${VENV_DIR:-$APP_DIR/venv}"
GIT_REMOTE="${GIT_REMOTE:-origin}"
GIT_BRANCH="${GIT_BRANCH:-main}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$APP_DIR" || exit 1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting deployment in $APP_DIR"

git fetch "$GIT_REMOTE" "$GIT_BRANCH" --prune
git pull --ff-only "$GIT_REMOTE" "$GIT_BRANCH"

if [ ! -x "$VENV_DIR/bin/python" ]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Creating virtualenv at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/pip" install -q --upgrade pip
"$VENV_DIR/bin/pip" install -q -r requirements.txt

pm2 restart qnm-analyser --update-env
# Wait briefly and require a local 200 so a dead backend is obvious in webhook logs
sleep 2
code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 http://127.0.0.1:8050/ || true)"
if [ "$code" != "200" ]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: local health check returned HTTP ${code:-none}" >&2
  exit 1
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Deployment complete (local HTTP $code)"
