"""GitHub Webhook Receiver for QNM Analyser auto-deployment.

Author: Dr. Denys Dutykh
        Khalifa University of Science and Technology, Abu Dhabi, UAE
        https://www.denys-dutykh.com/

Listens on loopback only; Traefik proxies /webhook to it.  A push to the
deploy branch, carrying a valid HMAC-SHA256 signature, runs deploy.sh.
"""

import hashlib
import hmac
import json
import logging
import os
import subprocess
from pathlib import Path

from flask import Flask, abort, request

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "")
APP_DIR = Path(
    os.environ.get("APP_DIR", str(Path(__file__).resolve().parent))
).resolve()
DEPLOY_SCRIPT = APP_DIR / "deploy.sh"
DEPLOY_BRANCH = os.environ.get("GIT_BRANCH", "main")
DEPLOY_REF = f"refs/heads/{DEPLOY_BRANCH}"

# Deploys launched by this process, reaped opportunistically so that finished
# children do not linger as zombies in a long-running worker.
_children = []


def verify_signature(payload, signature):
    """Verify the GitHub HMAC-SHA256 signature over the raw request body.

    Fails closed when no secret is configured.  Both operands are compared as
    bytes: hmac.compare_digest raises TypeError on a non-ASCII str, which an
    attacker could otherwise trigger with a crafted header.
    """
    if not WEBHOOK_SECRET:
        return False
    expected = "sha256=" + hmac.new(
        WEBHOOK_SECRET.encode(), payload, hashlib.sha256
    ).hexdigest()
    if isinstance(signature, str):
        signature = signature.encode("utf-8", "replace")
    return hmac.compare_digest(expected.encode(), signature)


def _reap():
    """Drop references to deploys that have finished."""
    for proc in list(_children):
        if proc.poll() is not None:
            _children.remove(proc)


def extract_ref(req):
    """Return the pushed git ref, or None if it cannot be determined.

    GitHub can deliver either content type configured on the webhook:
    application/json, or application/x-www-form-urlencoded with the JSON in a
    "payload" field.  Both are handled so that changing that setting does not
    silently stop deployments.
    """
    payload = req.get_json(silent=True)
    if payload is None:
        form_payload = req.form.get("payload")
        if form_payload:
            try:
                payload = json.loads(form_payload)
            except ValueError:
                payload = None
    if payload is None:
        # Signature already verified, so the body is genuinely from GitHub;
        # parse it directly rather than trusting the Content-Type header.
        try:
            payload = json.loads(
                req.get_data(cache=True, parse_form_data=False).decode("utf-8")
            )
        except (ValueError, UnicodeDecodeError):
            return None
    if not isinstance(payload, dict):
        return None
    ref = payload.get("ref")
    return ref if isinstance(ref, str) else None


@app.route("/webhook", methods=["POST"])
def webhook():
    # Read and cache the raw body BEFORE anything touches request.form.  For a
    # form-encoded delivery Werkzeug consumes the stream while parsing, after
    # which request.data is empty and the HMAC would never match.
    raw_body = request.get_data(cache=True, parse_form_data=False)

    signature = request.headers.get("X-Hub-Signature-256", "")
    if not verify_signature(raw_body, signature):
        logger.warning(
            "Rejected webhook: bad or missing signature (from %s)",
            request.headers.get("X-Forwarded-For", request.remote_addr),
        )
        abort(403)

    event = request.headers.get("X-GitHub-Event", "")
    if event == "ping":
        return "Pong", 200
    if event != "push":
        return "Event ignored", 200

    ref = extract_ref(request)
    if ref != DEPLOY_REF:
        logger.info("Ignoring push to %s (deploy branch is %s)", ref, DEPLOY_REF)
        return f"Ignored: not {DEPLOY_REF}", 200

    _reap()
    if _children:
        logger.info("Deployment already running; skipping")
        return "Deployment already running", 202

    logger.info("Push to %s accepted; starting deployment", ref)
    # argv is a fixed, module-level path; no request data reaches it, and no
    # shell is involved.
    _children.append(
        subprocess.Popen([str(DEPLOY_SCRIPT)], cwd=str(APP_DIR))  # noqa: S603
    )
    return "Deployment triggered", 200


@app.route("/webhook/health", methods=["GET"])
def health():
    return "OK", 200


@app.route("/health", methods=["GET"])
def health_root():
    return "OK", 200


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("WEBHOOK_PORT", "9050")))
