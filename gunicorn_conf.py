"""Gunicorn configuration for QNM Analyser.

Author: Dr. Denys Dutykh
        Khalifa University of Science and Technology, Abu Dhabi, UAE
        https://www.denys-dutykh.com/
"""

# Bind IPv4 loopback only. Traefik must target http://127.0.0.1:8050 —
# plain "localhost" prefers [::1] on this host and marks the backend down.
bind = "127.0.0.1:8050"
workers = 4
timeout = 120
accesslog = "-"
errorlog = "-"
loglevel = "info"
