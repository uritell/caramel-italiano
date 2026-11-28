#!/bin/bash
# Lean data pod. /start.sh stays PID 1. This job downloads, speed-tests, encodes.
set -euo pipefail
log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
mkdir -p /workspace/pilot-out/logs /workspace/work
# Tee to both logs. A plain redirect to bootstrap.log left job.log empty.
exec > >(tee -a /workspace/pilot-out/logs/bootstrap.log /workspace/pilot-out/logs/job.log >/dev/null) 2>&1
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
# Same pull server as pod_run.sh, before any audio download, so the launcher
# can read speed_test.json and the token shards on port 8080.
if [[ -n "${PILOT_PULL_TOKEN:-}" ]]; then
  PILOT_OUT="${PILOT_OUT:-/workspace/pilot-out}" PILOT_PULL_PORT="${PILOT_PULL_PORT:-8080}" \
    python3 "$dest/runpod/pull_server.py" >>/workspace/pilot-out/logs/pull_server.log 2>&1 &
  echo $! > /tmp/pull-server.pid
  log "pull server on ${PILOT_PULL_PORT:-8080}"
else
  log "PILOT_PULL_TOKEN unset; pull server not started"
fi
# Same packages as the training pod (transformers, torch, …). Reuses install_pod.
# shellcheck source=pod_env.sh
source "$dest/runpod/pod_env.sh"
PY=python3
if ! install_pod > /workspace/pilot-out/logs/install.log 2>&1; then
  {
    echo "install_pod failed"
    tail -n 120 /workspace/pilot-out/logs/install.log 2>/dev/null || true
  } > /workspace/pilot-out/FAILED
  log "install_pod failed"
  exit 1
fi
log "install_pod done ($PY)"
# Prove Mimi can load before any audio download. A failure is pullable.
export PYTHONPATH="$dest/runpod${PYTHONPATH:+:$PYTHONPATH}"
if ! "$PY" -c 'import traceback, sys
try:
    import transformers, torch
    from tokenize_store import _load_mimi
    _load_mimi("cuda" if torch.cuda.is_available() else "cpu")
except Exception:
    traceback.print_exc()
    sys.exit(1)
' > /workspace/pilot-out/logs/preflight.log 2>&1; then
  {
    echo "preflight failed: import transformers, torch; load Mimi"
    cat /workspace/pilot-out/logs/preflight.log 2>/dev/null || true
  } > /workspace/pilot-out/FAILED
  log "mimi preflight failed"
  exit 1
fi
log "mimi preflight ok"
cd "$dest"
exec "$PY" runpod/lean_prep.py --pod --out /workspace/pilot-out --work /workspace/work/lean \
  --hourly-cents "${PILOT_HOURLY_CENTS:-59}" --cap-cents "${PILOT_MAX_USD_CENTS:-300}"
