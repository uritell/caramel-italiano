#!/bin/bash
# Lean data pod. /start.sh stays PID 1. This job downloads, speed-tests, encodes.
set -euo pipefail
log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
mkdir -p /workspace/pilot-out/logs /workspace/work
exec >>/workspace/pilot-out/logs/bootstrap.log 2>&1
log "lean bootstrap"

repo="${CARAMEL_REPO_URL:-https://github.com/uritell/caramel-italiano.git}"
ref="${CARAMEL_REPO_REF:-main}"
dest=/workspace/caramel-italiano
git config --global --add safe.directory '*' || true
if ! git clone --recurse-submodules --branch "$ref" "$repo" "$dest"; then
  log "clone failed"
  exit 1
fi
export PILOT_WATCH_JOB=1
export PILOT_LEAN_DATA=1
if command -v setsid >/dev/null 2>&1; then
  setsid bash "$dest/runpod/watchdog.sh" </dev/null >>/workspace/pilot-out/logs/watchdog.log 2>&1 &
else
  nohup bash "$dest/runpod/watchdog.sh" </dev/null >>/workspace/pilot-out/logs/watchdog.log 2>&1 &
fi
log "watchdog started"
cd "$dest"
exec python3 runpod/lean_prep.py --pod --out /workspace/pilot-out --work /workspace/work/lean \
  --hourly-cents "${PILOT_HOURLY_CENTS:-59}" --cap-cents "${PILOT_MAX_USD_CENTS:-300}"
