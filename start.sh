#!/bin/bash
# QNM Analyser - PM2 start script
# Author: Dr. Denys Dutykh (https://www.denys-dutykh.com/)
set -euo pipefail
cd "$(dirname "$0")"

# Load environment if present (WEBHOOK_SECRET, GUNICORN_* overrides, ...)
set -a
[ -f .env ] && . ./.env
set +a

if [ ! -x venv/bin/gunicorn ]; then
  echo "ERROR: venv/bin/gunicorn missing. Run ./deploy.sh (or recreate the venv) first." >&2
  exit 127
fi

# A core dump is a full memory image, including the environment block; never
# write one into the checkout.
ulimit -c 0

# NumPy/SciPy would otherwise start a BLAS thread pool per worker sized to the
# CPU count, so workers x threads oversubscribes a small VPS.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

exec venv/bin/gunicorn --config gunicorn_conf.py app:server
