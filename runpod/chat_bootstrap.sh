#!/bin/bash
# Inside the chat pod. /start.sh is PID 1; this script is the job.
# It clones the public repo, starts the bash cost watchdog, then the chat server.
set -euo pipefail
log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
mkdir -p /workspace/chat-logs /workspace/chat-model
exec >>/workspace/chat-logs/bootstrap.log 2>&1
log "chat bootstrap"

repo="${CARAMEL_REPO_URL:-https://github.com/uritell/caramel-italiano.git}"
ref="${CARAMEL_REPO_REF:-main}"
dest=/workspace/caramel-italiano
git config --global --add safe.directory '*' || true
if ! git clone --recurse-submodules --branch "$ref" "$repo" "$dest"; then
  log "clone failed"
  exit 1
fi
log "cloned $ref"

# Do not export PILOT_WATCH_JOB. That flag is the training pull-grace
# terminate from the pilot watchdog. A chat pod that inherited it would
# delete itself with no terminate line in the launcher log.
unset PILOT_WATCH_JOB || true
if command -v setsid >/dev/null 2>&1; then
  setsid bash "$dest/runpod/watchdog.sh" </dev/null >>/workspace/chat-logs/watchdog.log 2>&1 &
else
  nohup bash "$dest/runpod/watchdog.sh" </dev/null >>/workspace/chat-logs/watchdog.log 2>&1 &
fi
log "watchdog started with bash (chat: no DONE/FAILED self-terminate)"

if ! python - <<'PY'
import inspect, sys
try:
    from transformers import MimiModel
    ok = "use_streaming" in inspect.signature(MimiModel.encode).parameters
except Exception:
    ok = False
sys.exit(0 if ok else 1)
PY
then
  python -m pip install --disable-pip-version-check 'transformers>=4.51'
fi
python -m pip install --disable-pip-version-check -q safetensors huggingface_hub

log "starting chat server"
exec python "$dest/chat/server.py" --wait-upload --model-dir /workspace/chat-model --port 8080
