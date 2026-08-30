#!/bin/bash
# Install QNM Analyser Traefik dynamic config into the live file-provider dir.
# Author: Dr. Denys Dutykh (https://www.denys-dutykh.com/)
set -euo pipefail

SRC="$(cd "$(dirname "$0")" && pwd)/traefik-qnm-analyser.yml"
DEST_DIR="${TRAEFIK_DYNAMIC_DIR:-/etc/traefik/dynamic}"
DEST="$DEST_DIR/qnm-analyser.yml"
TS="$(date +%Y%m%d%H%M%S)"

if [ ! -f "$SRC" ]; then
  echo "ERROR: missing $SRC" >&2
  exit 1
fi

if [ "$(id -u)" -ne 0 ]; then
  echo "Re-running under sudo..."
  exec sudo -E env TRAEFIK_DYNAMIC_DIR="$DEST_DIR" bash "$0"
fi

mkdir -p "$DEST_DIR"
if [ -f "$DEST" ]; then
  cp -a "$DEST" "${DEST}.bak.${TS}"
  echo "Backed up existing config to ${DEST}.bak.${TS}"
fi
install -m 0644 -o root -g root "$SRC" "$DEST"
echo "Installed $DEST"
ls -la "$DEST"

# Traefik file provider watches the directory; no restart required.
# Still surface recent errors if journal is available.
if command -v journalctl >/dev/null 2>&1; then
  sleep 1
  journalctl -u traefik --no-pager -n 15 2>/dev/null || true
fi

echo
echo "Verify:"
echo "  curl -sI https://qnm-anal.denys-dutykh.com/ | grep -i cache-control"
echo "  # expect: no-store, must-revalidate"
