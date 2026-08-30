#!/bin/bash
# QNM Analyser - PM2 start script
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x venv/bin/gunicorn ]; then
  echo "ERROR: venv/bin/gunicorn missing. Run ./deploy.sh (or recreate the venv) first." >&2
  exit 127
fi
exec venv/bin/gunicorn --config gunicorn_conf.py app:server
