#!/bin/bash
# QNM Analyser - Auto-deployment script
# Triggered by GitHub webhook on push

APP_DIR="${APP_DIR:-/home/dds/www/qnm-analyser}"
VENV_DIR="${VENV_DIR:-$APP_DIR/venv}"
GIT_REMOTE="${GIT_REMOTE:-origin}"
GIT_BRANCH="${GIT_BRANCH:-main}"

cd "$APP_DIR" || exit 1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting deployment in $APP_DIR"

git fetch "$GIT_REMOTE" "$GIT_BRANCH" --prune

git pull --ff-only "$GIT_REMOTE" "$GIT_BRANCH"

"$VENV_DIR/bin/pip" install -q -r requirements.txt

pm2 restart qnm-analyser

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Deployment complete"
