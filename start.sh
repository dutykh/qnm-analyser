#!/bin/bash
# QNM Analyser - PM2 start script
cd "$(dirname "$0")"
source venv/bin/activate
exec gunicorn --config gunicorn_conf.py app:server
