<!-- QNM Analyser: Deployment Guide
     Author: Dr. Denys Dutykh
             Khalifa University of Science and Technology, Abu Dhabi, UAE
             https://www.denys-dutykh.com/ -->

# QNM Analyser: Deployment Guide

**Author:** Dr. Denys Dutykh, Khalifa University of Science and Technology,
Abu Dhabi, UAE, [denys-dutykh.com](https://www.denys-dutykh.com/)

How the QNM Analyser actually runs in production, and how to operate it.

```text
   Internet
      |
      v
   Traefik           TLS termination, security headers, rate limiting
      |                (routers + middlewares in deploy/traefik-qnm-analyser.yml)
      +--> 127.0.0.1:8050   gunicorn -> app:server        (PM2: qnm-analyser)
      +--> 127.0.0.1:9050   gunicorn -> webhook:app       (PM2: qnm-webhook)
                                          |
                                          v
                                     deploy.sh  (git pull, pip install, pm2 restart)
```

- **Install root:** `/home/dds/www/qnm-analyser`
- **Supervisor:** PM2, running as the `dds` user (not root, not `www-data`)
- **Ingress:** Traefik, with its own ACME/Let's Encrypt resolver
- **Updates:** a GitHub webhook triggers `deploy.sh`

Both processes bind loopback only. Nothing but Traefik is reachable from the
internet.

## Prerequisites

- Ubuntu 22.04+ with sudo access
- Python 3.11+
- Node.js and PM2 (`npm install -g pm2`)
- Traefik already running, with a file provider watching a dynamic-config
  directory
- DNS: `qnm-anal.denys-dutykh.com` and `www.qnm-anal.denys-dutykh.com` both
  pointing at the VPS

## 1. First-time install

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git curl

mkdir -p ~/www && cd ~/www
git clone https://github.com/dutykh/qnm-analyser.git
cd qnm-analyser

python3 -m venv venv
venv/bin/pip install --upgrade pip
venv/bin/pip install -r requirements.txt
```

## 2. Configure the environment

```bash
cp .env.example .env
chmod 600 .env
openssl rand -hex 32          # paste into WEBHOOK_SECRET
${EDITOR:-nano} .env
```

`.env` is gitignored and is sourced as shell by the start scripts, so avoid
unquoted backticks and `$(...)` in values.

## 3. Start under PM2

```bash
pm2 start ecosystem.config.js
pm2 save
pm2 startup                   # run the command it prints, once
```

Verify both processes locally before touching the proxy:

```bash
curl -i http://127.0.0.1:8050/health          # expect 200 OK
curl -i http://127.0.0.1:9050/webhook/health  # expect 200 OK
```

## 4. Configure Traefik

Copy `deploy/traefik-qnm-analyser.yml` into Traefik's dynamic-config directory
(commonly `/etc/traefik/dynamic/`) and let the file provider pick it up.

Two points in that file matter enough to repeat here.

**Use `127.0.0.1`, never `localhost`, in the service URL.** On this host
`localhost` resolves to `[::1]` first, gunicorn listens on IPv4 only, and
Traefik marks the backend down and serves a public 503.

**Never attach a long-lived cache middleware to the application router.** A
`Cache-Control` middleware also stamps Traefik's own 503 page. A visitor who
arrives during an outage then caches that error for the full `max-age`, and
`immutable` stops the browser revalidating even on an ordinary reload, so the
site stays broken for that person long after the backend recovers. The config
splits this into two routers: `qnm-anal-assets` caches hard, `qnm-anal` sets
`no-store`.

Check the result from outside:

```bash
curl -sI https://www.qnm-anal.denys-dutykh.com/ | grep -i cache-control
# expect: no-store, must-revalidate   (NOT max-age=31536000)
```

## 5. Configure the GitHub webhook

In the repository settings, add a webhook:

| Field | Value |
| ----- | ----- |
| Payload URL | `https://www.qnm-anal.denys-dutykh.com/webhook` |
| Content type | `application/json` (form-encoded also works) |
| Secret | the `WEBHOOK_SECRET` value from `.env` |
| Events | Just the push event |

Only a push to the branch named by `GIT_BRANCH` (default `main`) triggers a
deploy. Requests without a valid HMAC-SHA256 signature are rejected with 403,
and if `WEBHOOK_SECRET` is unset **every** request is rejected.

## Updating

Normally: push to `main` and the webhook runs `deploy.sh`, which pulls,
installs dependencies, stamps the version into `package.json`, restarts PM2,
and then requires a local `200` from `/health` before reporting success.

Manually, or to recover:

```bash
cd ~/www/qnm-analyser
./deploy.sh
```

`deploy.sh` takes an exclusive `flock`, so two deploys cannot race.

## Operations

```bash
pm2 status                             # both processes
pm2 logs qnm-analyser --lines 100      # application log
pm2 logs qnm-webhook --lines 100       # deploy log, including rejected hooks
pm2 restart qnm-analyser --update-env  # after editing .env
pm2 describe qnm-analyser              # shows the deployed version stamp
```

### Tuning workers

`gunicorn_conf.py` reads its values from the environment, so tune in `.env` and
restart rather than editing a tracked file:

```bash
GUNICORN_WORKERS=3
GUNICORN_THREADS=4
```

Threaded workers matter here: a PNG or PDF export drives Kaleido, which blocks
for seconds. Keep `OMP_NUM_THREADS=1` and its siblings at 1, or every worker
starts a BLAS thread pool sized to the CPU count and oversubscribes the box.

### Core dumps

`start.sh`, `webhook_start.sh` and `deploy.sh` all set `ulimit -c 0`. A core
dump is a full image of process memory, including the environment block, so one
landing in the checkout is both a secret leak and something a careless
`git add` will commit. Belt and braces, set a system-wide path outside any
repository:

```bash
echo 'kernel.core_pattern=/var/lib/systemd/coredump/core.%e.%p' \
  | sudo tee /etc/sysctl.d/50-coredump.conf
sudo sysctl --system
```

## Troubleshooting

### Public 503 "no available server"

Traefik has no healthy backend. In order:

```bash
pm2 status                                    # is qnm-analyser online?
curl -i http://127.0.0.1:8050/health          # does the app answer locally?
pm2 logs qnm-analyser --lines 50              # why did it exit?
ls -la venv/bin/gunicorn                      # venv intact?
```

The usual causes are a missing or half-built venv (`start.sh` exits 127 with an
explicit message), and a Traefik service pointing at `localhost` rather than
`127.0.0.1`.

### The site looks down in a browser but `curl` returns 200

The browser is serving a cached error page. This happens if a cache middleware
was ever attached to a response that was not a 200. Hard-reload with
`Ctrl+Shift+R` (`Cmd+Shift+R` on macOS), and check §4 above.

### A push did not deploy

```bash
pm2 logs qnm-webhook --lines 50
```

- `Rejected webhook: bad or missing signature`: the secret in `.env` and the
  one in the GitHub webhook settings disagree.
- `Ignoring push to refs/heads/...`: the push was to another branch.
- `Deployment already running`: a previous deploy still holds the lock.

### PDF or PNG export fails

Kaleido renders through a headless Chromium that it downloads separately;
`deploy.sh` does not do that step. If exports fail after rebuilding the venv:

```bash
venv/bin/python -c "import kaleido; kaleido.get_chrome_sync()"
```

The failure is logged and returns no file rather than taking the worker down.

## Rolling back

```bash
cd ~/www/qnm-analyser
git log --oneline -10
git checkout <good-commit>
venv/bin/pip install -r requirements.txt
pm2 restart qnm-analyser --update-env
curl -i http://127.0.0.1:8050/health
```

Return to the branch tip with `git checkout main` once the cause is fixed.

## Security notes

- Both application processes listen on loopback only; Traefik is the sole
  public entry point.
- Uploads are capped twice: `MAX_CONTENT_LENGTH` (10 MB) in `app.py` and a
  `buffering` middleware at Traefik. Neither alone is sufficient, since the
  Traefik limit does not exist if the middleware is not attached.
- No uploaded data is written to disk or persisted server-side.
- `WEBHOOK_SECRET` is the only secret. Rotate it by generating a new value,
  updating `.env` and the GitHub webhook settings, then
  `pm2 restart qnm-webhook --update-env`.
- Firewall: allow only 22, 80 and 443. Ports 8050 and 9050 must never be
  exposed.

```bash
sudo ufw allow OpenSSH
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```
