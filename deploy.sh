#!/bin/bash
# QNM Analyser - Auto-deployment script
# Triggered by GitHub webhook on push to the deploy branch.
#
# Author: Dr. Denys Dutykh (https://www.denys-dutykh.com/)
set -euo pipefail

APP_DIR="${APP_DIR:-/home/dds/www/qnm-analyser}"
VENV_DIR="${VENV_DIR:-$APP_DIR/venv}"
GIT_REMOTE="${GIT_REMOTE:-origin}"
GIT_BRANCH="${GIT_BRANCH:-main}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PM2_APP="${PM2_APP:-qnm-analyser}"
HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:8050/health}"
LOCK_FILE="${LOCK_FILE:-/tmp/qnm-analyser-deploy.lock}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# Serialise deploys: two pushes in quick succession would otherwise race on
# git's index.lock and on the venv.
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  log "Another deployment is in progress; aborting."
  exit 0
fi

# Never leave a core dump in the checkout: it is a full memory image and would
# be picked up by the next 'git add'.
ulimit -c 0

cd "$APP_DIR"

log "Starting deployment in $APP_DIR"

git pull --ff-only "$GIT_REMOTE" "$GIT_BRANCH"

if [ ! -x "$VENV_DIR/bin/python" ]; then
  log "Creating virtualenv at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install -r requirements.txt

# Expose the deployed commit in `pm2 status`.
COMMIT="$(git rev-parse --short HEAD)"
STAMP="0.0.0-$(date '+%Y%m%d').${COMMIT}"
"$PYTHON_BIN" - "$STAMP" <<'PY'
import json, sys, pathlib
p = pathlib.Path("package.json")
data = json.loads(p.read_text())
data["version"] = sys.argv[1]
p.write_text(json.dumps(data, indent=2) + "\n")
PY
log "Stamped package.json version $STAMP"

pm2 restart "$PM2_APP" --update-env

# Require a local 200 so a dead backend is obvious in the webhook log rather
# than only surfacing as a public 503.
sleep 2
code=""
for _ in 1 2 3 4 5; do
  code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$HEALTH_URL" || true)"
  [ "$code" = "200" ] && break
  sleep 2
done

if [ "$code" != "200" ]; then
  log "ERROR: local health check returned HTTP ${code:-none}"
  exit 1
fi

log "Deployment complete ($COMMIT, local HTTP $code)"
