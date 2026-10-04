#!/bin/bash
# DietPi first-boot installer for the vehicle tracker.
# Place at /boot/Automation_Custom_Script.sh with AUTO_SETUP_CUSTOM_SCRIPT_EXEC=0.
LOG=/var/log/tracker-install.log
exec >>"$LOG" 2>&1
echo "==== tracker install start: $(date) ===="
export DEBIAN_FRONTEND=noninteractive

set -e
apt-get update
apt-get install -y --no-install-recommends python3-venv python3-pip ca-certificates

# Dedicated unprivileged service user + persistent (offline-first) storage
id tracker >/dev/null 2>&1 || useradd -r -s /usr/sbin/nologin -d /opt/tracker tracker
install -d -o tracker -g tracker /var/lib/tracker \
    /var/lib/tracker/captures /var/lib/tracker/captures/thumb

cd /opt/tracker
python3 -m venv venv
./venv/bin/pip install --upgrade pip wheel
./venv/bin/pip install -r requirements.txt
# Pure-Python but declares torch as a hard dep; skip it.
./venv/bin/pip install --no-deps bytetracker==0.3.2

chown -R tracker:tracker /opt/tracker

install -m 0644 /opt/tracker/deploy/tracker.service /etc/systemd/system/tracker.service
install -m 0644 /opt/tracker/deploy/sync.service /etc/systemd/system/sync.service
systemctl daemon-reload
systemctl enable tracker.service sync.service

echo "==== tracker install done: $(date) ===="
