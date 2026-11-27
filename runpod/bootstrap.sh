#!/bin/bash
# Fetched from the public repo and started under nohup. /start.sh stays PID 1
# (sshd, the HTTP proxy, sleep infinity). This process clones the repo and
# runs pod_run.sh. The watchdog body below the exit is copied out and run
# as its own process. Keep it identical to runpod/watchdog.sh.
set -u
mkdir -p /workspace/pilot-out/logs /workspace/work
echo "bootstrap $(date -u +%FT%TZ)" | tee -a /workspace/pilot-out/logs/bootstrap.log

if [[ -n "${PUBLIC_KEY:-}" ]]; then
  mkdir -p /root/.ssh
  printf '%s\n' "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
  chmod 700 /root/.ssh
  chmod 600 /root/.ssh/authorized_keys
  if [[ ! -f /etc/ssh/ssh_host_ed25519_key ]]; then
    ssh-keygen -t ed25519 -f /etc/ssh/ssh_host_ed25519_key -q -N '' || true
  fi
  service ssh start || /usr/sbin/sshd || true
  echo "sshd started" | tee -a /workspace/pilot-out/logs/bootstrap.log
else
  echo "PUBLIC_KEY unset; scp pull needs the HTTP port instead" | tee -a /workspace/pilot-out/logs/bootstrap.log
fi

sed -n '/^# BEGIN WATCHDOG$/,/^# END WATCHDOG$/p' "$0" | sed '1d;$d' > /tmp/pilot-watchdog.sh
chmod +x /tmp/pilot-watchdog.sh
if command -v setsid >/dev/null 2>&1; then
  setsid /tmp/pilot-watchdog.sh </dev/null >>/workspace/pilot-out/logs/watchdog.log 2>&1 &
else
  nohup /tmp/pilot-watchdog.sh </dev/null >>/workspace/pilot-out/logs/watchdog.log 2>&1 &
fi
echo $! > /tmp/pilot-watchdog.pid

repo="${CARAMEL_REPO_URL:-https://github.com/uritell/caramel-italiano.git}"
ref="${CARAMEL_REPO_REF:-main}"
dest=/workspace/caramel-italiano
git config --global --add safe.directory '*' || true
if [[ -n "${GITHUB_TOKEN:-}" ]]; then
  git config --global http.https://github.com/.extraheader "AUTHORIZATION: bearer ${GITHUB_TOKEN}"
fi
if ! git clone --recurse-submodules --branch "$ref" "$repo" "$dest"; then
  echo "clone-failed" | tee /workspace/pilot-out/status.txt | tee -a /workspace/pilot-out/logs/bootstrap.log
  git config --global --unset-all http.https://github.com/.extraheader || true
  sleep 120
  bash /tmp/pilot-watchdog.sh --once || true
  exit 1
fi
git config --global --unset-all http.https://github.com/.extraheader || true
echo "cloned $ref" | tee -a /workspace/pilot-out/logs/bootstrap.log
exec bash "$dest/runpod/pod_run.sh" --full

# BEGIN WATCHDOG
set -u

graphql_post() {
  local query="$1"
  local vars="$2"
  local qjson
  qjson=$(printf '%s' "$query" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')
  curl -fsS --max-time 45 https://api.runpod.io/graphql \
    -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
    -H "Content-Type: application/json" \
    --data "{\"query\":${qjson},\"variables\":${vars}}"
}

_pod_id_ok() {
  case "${1:-}" in
    ""|*[!A-Za-z0-9_-]*) return 1 ;;
  esac
  return 0
}

terminate_pod() {
  local id="${RUNPOD_POD_ID:-}"
  local key="${RUNPOD_API_KEY:-}"
  if [[ -z "$key" ]] || ! _pod_id_ok "$id"; then
    echo "terminate skipped: missing RUNPOD_POD_ID or RUNPOD_API_KEY"
    return 1
  fi
  local resp
  resp=$(graphql_post 'mutation($input:PodTerminateInput!){podTerminate(input:$input)}' "{\"input\":{\"podId\":\"${id}\"}}" || true)
  if [[ -n "$resp" && "$resp" != *'"errors"'* ]]; then
    echo "terminated via graphql podTerminate"
    return 0
  fi
  echo "graphql terminate failed: ${resp:-curl error}"
  if command -v runpodctl >/dev/null 2>&1; then
    if timeout 60 env RUNPOD_API_KEY="$key" runpodctl pod delete "$id"; then
      echo "terminated via runpodctl pod delete"
      return 0
    fi
  fi
  if curl -fsS --max-time 45 -X POST "https://api.runpod.io/v2/pods/${id}/action" \
      -H "Authorization: Bearer ${key}" \
      -H "Content-Type: application/json" \
      -d '{"action":"terminate"}'; then
    echo "terminated via REST fallback"
    return 0
  fi
  if curl -fsS --max-time 45 -X DELETE "https://api.runpod.io/v2/pods/${id}" \
      -H "Authorization: Bearer ${key}"; then
    echo "terminated via REST fallback DELETE"
    return 0
  fi
  return 1
}

# podStop exists so a manual call can pause the pod. The cap uses terminate_pod
# because a stopped pod can keep a disk charge.
stop_pod() {
  local id="${RUNPOD_POD_ID:-}"
  local key="${RUNPOD_API_KEY:-}"
  if [[ -z "$key" ]] || ! _pod_id_ok "$id"; then
    echo "stop skipped: missing RUNPOD_POD_ID or RUNPOD_API_KEY"
    return 1
  fi
  local resp
  resp=$(graphql_post 'mutation($input:PodStopInput!){podStop(input:$input){id desiredStatus}}' "{\"input\":{\"podId\":\"${id}\"}}" || true)
  if [[ -n "$resp" && "$resp" != *'"errors"'* ]]; then
    echo "stopped via graphql podStop"
    return 0
  fi
  echo "graphql stop failed: ${resp:-curl error}"
  curl -fsS --max-time 45 -X POST "https://api.runpod.io/v2/pods/${id}/action" \
    -H "Authorization: Bearer ${key}" \
    -H "Content-Type: application/json" \
    -d '{"action":"stop"}'
}

if [[ "${1:-}" == "--once" ]]; then
  terminate_pod
  exit $?
fi
if [[ "${1:-}" == "--stop" ]]; then
  stop_pod
  exit $?
fi

start="${PILOT_BILLING_EPOCH:-$(date +%s)}"
max_s="${PILOT_MAX_SECONDS:-9000}"
max_cents="${PILOT_MAX_USD_CENTS:-2200}"
hourly="${PILOT_HOURLY_CENTS:-878}"
echo "watchdog armed start=${start} wall=${max_s}s cap_cents=${max_cents} hourly_cents=${hourly}"
while true; do
  now=$(date +%s)
  elapsed=$((now - start))
  if (( elapsed < 0 )); then elapsed=0; fi
  # Same rounding as runpod/cost.py: (elapsed * hourly + 3599) / 3600
  cost=$(( (elapsed * hourly + 3599) / 3600 ))
  printf 'watchdog elapsed=%ss cost_cents=%s cap_cents=%s wall_s=%s\n' \
    "$elapsed" "$cost" "$max_cents" "$max_s"
  if (( elapsed >= max_s || cost >= max_cents )); then
    echo "watchdog cap reached"
    terminate_pod && exit 0
    sleep 15
    continue
  fi
  sleep 20
done
# END WATCHDOG
