"""Gunicorn configuration for QNM Analyser.

Author: Dr. Denys Dutykh
        Khalifa University of Science and Technology, Abu Dhabi, UAE
        https://www.denys-dutykh.com/

Every value can be overridden from the environment, so the VPS can be tuned
without editing a tracked file.
"""

import os

# Bind IPv4 loopback only. Traefik must target http://127.0.0.1:8050 —
# plain "localhost" prefers [::1] on this host and marks the backend down.
bind = os.environ.get("GUNICORN_BIND", "127.0.0.1:8050")

# Threaded workers: a PNG/PDF export drives Kaleido, which blocks for seconds.
# With the default synchronous worker those exports block the whole process.
worker_class = "gthread"
workers = int(os.environ.get("GUNICORN_WORKERS", "3"))
threads = int(os.environ.get("GUNICORN_THREADS", "4"))

timeout = int(os.environ.get("GUNICORN_TIMEOUT", "120"))
graceful_timeout = 30
keepalive = 5

# Recycle workers so that memory held by the headless browser Kaleido spawns
# cannot accumulate for the lifetime of the process.  The jitter keeps the
# workers from restarting in lockstep.
max_requests = 500
max_requests_jitter = 50

# Heartbeat file on tmpfs; on a disk-backed /tmp a busy or throttled volume can
# stall the heartbeat and have the arbiter kill a healthy worker.
worker_tmp_dir = "/dev/shm" if os.path.isdir("/dev/shm") else None

# Bound the request line and headers; the body is capped by Flask's
# MAX_CONTENT_LENGTH in app.py and should also be capped at the proxy.
limit_request_line = 8190
limit_request_fields = 100
limit_request_field_size = 8190

accesslog = "-"
errorlog = "-"
loglevel = os.environ.get("GUNICORN_LOGLEVEL", "info")
