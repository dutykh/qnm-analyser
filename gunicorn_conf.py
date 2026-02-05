"""Gunicorn configuration for QNM Analyser.

Author: Dr. Denys Dutykh
        Khalifa University of Science and Technology, Abu Dhabi, UAE
        https://www.denys-dutykh.com/
"""

bind = "127.0.0.1:8050"
workers = 4
timeout = 120
accesslog = "-"
errorlog = "-"
loglevel = "info"
