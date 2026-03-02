"""GitHub Webhook Receiver for QNM Analyser auto-deployment.

Author: Dr. Denys Dutykh
"""

import hashlib
import hmac
import os
import subprocess
from flask import Flask, request, abort

app = Flask(__name__)

WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "")
from pathlib import Path

APP_DIR = Path(os.environ.get("APP_DIR", str(Path(__file__).resolve().parent))).resolve()
DEPLOY_SCRIPT = APP_DIR / "deploy.sh"


def verify_signature(payload, signature):
    """Verify GitHub webhook signature."""
    if not WEBHOOK_SECRET:
        return False
    expected = "sha256=" + hmac.new(
        WEBHOOK_SECRET.encode(), payload, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


@app.route("/webhook", methods=["POST"])
def webhook():
    signature = request.headers.get("X-Hub-Signature-256", "")
    if not verify_signature(request.data, signature):
        abort(403)

    event = request.headers.get("X-GitHub-Event", "")
    if event == "push":
        subprocess.Popen([str(DEPLOY_SCRIPT)], cwd=str(APP_DIR))
        return "Deployment triggered", 200
    elif event == "ping":
        return "Pong", 200

    return "Event ignored", 200


@app.route("/webhook/health", methods=["GET"])
def health():
    return "OK", 200


@app.route("/health", methods=["GET"])
def health_root():
    return "OK", 200


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=9050)
