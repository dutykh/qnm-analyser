# QNM Analyser — Deployment Guide

**Author:** Dr. Denys Dutykh — Khalifa University of Science and Technology,
Abu Dhabi, UAE — [denys-dutykh.com](https://www.denys-dutykh.com/)

Step-by-step instructions for deploying the QNM Analyser on an Ubuntu VPS
behind Nginx with HTTPS.

Target URL: `https://www.qnm-anal.denys-dutykh.com/`

## Prerequisites

- Ubuntu 22.04+ VPS with root/sudo access
- Domain DNS configured: both `qnm-anal.denys-dutykh.com` and
  `www.qnm-anal.denys-dutykh.com` A records pointing to the VPS IP
- Python 3.10+

## 1. Install System Packages

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip nginx certbot python3-certbot-nginx
```

## 2. Create Application Directory

```bash
sudo mkdir -p /opt/qnm-analyser
sudo chown $USER:$USER /opt/qnm-analyser
```

## 3. Deploy Application Files

Copy the project files to the server. From your local machine:

```bash
rsync -avz --exclude '.git' --exclude '.tmp' --exclude '__pycache__' \
    ./ user@your-vps:/opt/qnm-analyser/
```

Or clone from Git:

```bash
cd /opt/qnm-analyser
git clone https://github.com/your-repo/qnm-analyser.git .
```

## 4. Set Up Python Virtual Environment

```bash
cd /opt/qnm-analyser
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Verify the Application Runs

```bash
cd /opt/qnm-analyser
source venv/bin/activate
python app.py
```

Open `http://your-vps-ip:8050` in a browser. Upload a `.dat` file to confirm
it works. Press `Ctrl+C` to stop.

## 6. Obtain SSL Certificate

Before configuring Nginx with SSL, obtain the certificate:

```bash
sudo certbot certonly --standalone \
    -d www.qnm-anal.denys-dutykh.com \
    -d qnm-anal.denys-dutykh.com
```

If Nginx is already running on port 80, use the Nginx plugin instead:

```bash
sudo certbot --nginx \
    -d www.qnm-anal.denys-dutykh.com \
    -d qnm-anal.denys-dutykh.com
```

Certbot automatically sets up certificate renewal via a systemd timer.

## 7. Configure Nginx

```bash
sudo cp deploy/nginx-qnm-analyser.conf /etc/nginx/sites-available/qnm-analyser
sudo ln -sf /etc/nginx/sites-available/qnm-analyser /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default   # optional: remove default site
sudo nginx -t
sudo systemctl reload nginx
```

## 8. Set Up systemd Service

```bash
sudo cp deploy/qnm-analyser.service /etc/systemd/system/
sudo chown -R www-data:www-data /opt/qnm-analyser
sudo systemctl daemon-reload
sudo systemctl enable --now qnm-analyser
```

## 9. Verify Deployment

```bash
# Check service status
sudo systemctl status qnm-analyser

# Check Nginx
sudo nginx -t

# Test HTTPS
curl -I https://www.qnm-anal.denys-dutykh.com/
```

Open `https://www.qnm-anal.denys-dutykh.com/` in a browser.

## Maintenance

### View Logs

```bash
# Application logs
sudo journalctl -u qnm-analyser -f

# Nginx access/error logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

### Restart After Code Update

```bash
cd /opt/qnm-analyser
git pull   # or rsync updated files
sudo chown -R www-data:www-data /opt/qnm-analyser
sudo systemctl restart qnm-analyser
```

### Renew SSL Certificate

Certbot sets up automatic renewal. To test:

```bash
sudo certbot renew --dry-run
```

### Change Number of Workers

Edit `gunicorn_conf.py` and adjust `workers`, then:

```bash
sudo systemctl restart qnm-analyser
```

## Firewall (Optional)

If `ufw` is enabled:

```bash
sudo ufw allow 'Nginx Full'
sudo ufw allow OpenSSH
sudo ufw enable
```

## Troubleshooting

| Symptom | Check |
|---------|-------|
| 502 Bad Gateway | `systemctl status qnm-analyser` -- is Gunicorn running? |
| Certificate error | `certbot certificates` -- is the cert valid and not expired? |
| Slow image export | Ensure `kaleido` is installed: `pip list \| grep kaleido` |
| Upload fails | Check `client_max_body_size` in Nginx config (default: 10 MB) |
