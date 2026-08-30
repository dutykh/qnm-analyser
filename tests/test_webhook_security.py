"""Tests for the deploy webhook's signature check.

`verify_signature` is the entire security boundary of the auto-deploy
pipeline: anything that passes it runs deploy.sh on the server.

Author: Dr. Denys Dutykh (https://www.denys-dutykh.com/)
"""

import hashlib
import hmac
import importlib

import pytest


def _load_webhook(monkeypatch, secret):
    """Import webhook.py with WEBHOOK_SECRET set to *secret*."""
    if secret is None:
        monkeypatch.delenv("WEBHOOK_SECRET", raising=False)
    else:
        monkeypatch.setenv("WEBHOOK_SECRET", secret)
    import webhook

    return importlib.reload(webhook)


def _sign(secret, payload):
    return "sha256=" + hmac.new(
        secret.encode(), payload, hashlib.sha256
    ).hexdigest()


def test_valid_signature_accepted(monkeypatch):
    wh = _load_webhook(monkeypatch, "s3cr3t")
    payload = b'{"ref": "refs/heads/main"}'
    assert wh.verify_signature(payload, _sign("s3cr3t", payload)) is True


def test_wrong_signature_rejected(monkeypatch):
    wh = _load_webhook(monkeypatch, "s3cr3t")
    payload = b'{"ref": "refs/heads/main"}'
    assert wh.verify_signature(payload, _sign("wrong-secret", payload)) is False


def test_tampered_payload_rejected(monkeypatch):
    wh = _load_webhook(monkeypatch, "s3cr3t")
    signature = _sign("s3cr3t", b'{"ref": "refs/heads/main"}')
    assert wh.verify_signature(b'{"ref": "refs/heads/evil"}', signature) is False


def test_missing_signature_rejected(monkeypatch):
    wh = _load_webhook(monkeypatch, "s3cr3t")
    assert wh.verify_signature(b"{}", "") is False


def test_unset_secret_fails_closed(monkeypatch):
    """With no secret configured, nothing may authenticate."""
    wh = _load_webhook(monkeypatch, None)
    payload = b"{}"
    assert wh.verify_signature(payload, _sign("", payload)) is False
    assert wh.verify_signature(payload, "sha256=" + "0" * 64) is False


@pytest.mark.parametrize(
    "signature",
    ["sha256=café", "sha256=ÿþ", "sha256=ünicode", "🔓"],
)
def test_non_ascii_signature_does_not_raise(monkeypatch, signature):
    """A non-ASCII header must be rejected, not raise TypeError into a 500."""
    wh = _load_webhook(monkeypatch, "s3cr3t")
    assert wh.verify_signature(b"{}", signature) is False


def test_deploy_ref_is_branch_scoped(monkeypatch):
    wh = _load_webhook(monkeypatch, "s3cr3t")
    assert wh.DEPLOY_REF.startswith("refs/heads/")


def test_non_push_event_does_not_deploy(monkeypatch):
    """Only a push to the deploy branch may spawn deploy.sh."""
    wh = _load_webhook(monkeypatch, "s3cr3t")
    spawned = []
    monkeypatch.setattr(
        wh.subprocess, "Popen", lambda *a, **k: spawned.append(a) or _FakeProc()
    )
    client = wh.app.test_client()

    payload = b'{"ref": "refs/heads/main"}'
    sig = _sign("s3cr3t", payload)

    # Wrong event type.
    r = client.post(
        "/webhook", data=payload,
        headers={"X-Hub-Signature-256": sig, "X-GitHub-Event": "issues"},
    )
    assert r.status_code == 200
    assert spawned == []

    # Right event, wrong branch.
    other = b'{"ref": "refs/heads/experiment"}'
    r = client.post(
        "/webhook", data=other,
        headers={"X-Hub-Signature-256": _sign("s3cr3t", other),
                 "X-GitHub-Event": "push"},
    )
    assert r.status_code == 200
    assert spawned == []

    # Bad signature on an otherwise valid push.
    r = client.post(
        "/webhook", data=payload,
        headers={"X-Hub-Signature-256": _sign("nope", payload),
                 "X-GitHub-Event": "push"},
    )
    assert r.status_code == 403
    assert spawned == []


def test_valid_push_to_main_deploys(monkeypatch):
    wh = _load_webhook(monkeypatch, "s3cr3t")
    spawned = []
    monkeypatch.setattr(
        wh.subprocess, "Popen", lambda *a, **k: spawned.append(a) or _FakeProc()
    )
    wh._children.clear()
    client = wh.app.test_client()

    payload = b'{"ref": "refs/heads/main"}'
    r = client.post(
        "/webhook", data=payload,
        content_type="application/json",
        headers={"X-Hub-Signature-256": _sign("s3cr3t", payload),
                 "X-GitHub-Event": "push"},
    )
    assert r.status_code == 200
    assert len(spawned) == 1
    # deploy.sh is invoked as a fixed argv list, never through a shell.
    assert spawned[0][0] == [str(wh.DEPLOY_SCRIPT)]
    wh._children.clear()


def test_deploys_for_both_github_content_types(monkeypatch):
    """GitHub sends JSON or form-encoded depending on webhook settings.

    Both must be understood, or flipping that setting silently stops deploys.
    """
    import urllib.parse

    body = b'{"ref": "refs/heads/main"}'
    form_body = urllib.parse.urlencode({"payload": body.decode()}).encode()

    cases = [
        (body, "application/json"),
        (form_body, "application/x-www-form-urlencoded"),
        (body, "text/plain"),  # no declared JSON type: parse the raw body
    ]
    for payload, content_type in cases:
        wh = _load_webhook(monkeypatch, "s3cr3t")
        spawned = []
        monkeypatch.setattr(
            wh.subprocess, "Popen",
            lambda *a, _s=spawned, **k: _s.append(a) or _FakeProc(),
        )
        wh._children.clear()
        r = wh.app.test_client().post(
            "/webhook", data=payload,
            content_type=content_type,
            headers={"X-Hub-Signature-256": _sign("s3cr3t", payload),
                     "X-GitHub-Event": "push"},
        )
        assert r.status_code == 200, f"{content_type}: {r.status_code}"
        assert len(spawned) == 1, f"{content_type} did not trigger a deploy"
        wh._children.clear()


def test_concurrent_deploy_is_skipped(monkeypatch):
    """A second push while a deploy runs must not start a racing deploy."""
    wh = _load_webhook(monkeypatch, "s3cr3t")
    spawned = []
    monkeypatch.setattr(
        wh.subprocess, "Popen", lambda *a, **k: spawned.append(a) or _FakeProc()
    )
    wh._children.clear()
    client = wh.app.test_client()
    payload = b'{"ref": "refs/heads/main"}'
    headers = {"X-Hub-Signature-256": _sign("s3cr3t", payload),
               "X-GitHub-Event": "push"}

    first = client.post("/webhook", data=payload,
                        content_type="application/json", headers=headers)
    second = client.post("/webhook", data=payload,
                         content_type="application/json", headers=headers)
    assert first.status_code == 200
    assert second.status_code == 202
    assert len(spawned) == 1
    wh._children.clear()


class _FakeProc:
    """A deploy that never finishes, so poll() keeps returning None."""

    def poll(self):
        return None
