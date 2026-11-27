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

{
  echo '#!/bin/bash'
  sed -n '/^# BEGIN WATCHDOG$/,/^# END WATCHDOG$/p' "$0" | sed '1d;$d'
} > /tmp/pilot-watchdog.sh
chmod +x /tmp/pilot-watchdog.sh
# Chat pods must not set this. It is what lets a finished training job
# terminate itself after the pull grace if the launcher has died.
export PILOT_WATCH_JOB=1
if command -v setsid >/dev/null 2>&1; then
  setsid bash /tmp/pilot-watchdog.sh </dev/null >>/workspace/pilot-out/logs/watchdog.log 2>&1 &
else
  nohup bash /tmp/pilot-watchdog.sh </dev/null >>/workspace/pilot-out/logs/watchdog.log 2>&1 &
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

cap_due() {
  # $1 elapsed seconds, $2 hourly cents, $3 wall seconds, $4 max cents
  local elapsed="$1" hourly="$2" max_s="$3" max_cents="$4"
  local cost=$(( (elapsed * hourly + 3599) / 3600 ))
  if (( elapsed >= max_s || cost >= max_cents )); then
    return 0
  fi
  return 1
}

self_test() {
  # Fake clocks. 2.5 h at $4.39 must trip. One minute must not.
  if ! cap_due 9000 439 9000 1200; then
    echo "watchdog self-test failed: 2.5h should trip the cap" >&2
    exit 1
  fi
  if cap_due 60 439 9000 1200; then
    echo "watchdog self-test failed: 60s should not trip the cap" >&2
    exit 1
  fi
  # Option A wall: 1.9 h at the quoted rate must trip. One minute must not.
  if ! cap_due 6840 439 6840 900; then
    echo "watchdog self-test failed: 1.9h should trip the option A wall" >&2
    exit 1
  fi
  # A finished job must die after the pull grace even if the box launcher is gone.
  if ! job_grace_elapsed 600 600; then
    echo "watchdog self-test failed: a finished job at the grace must terminate" >&2
    exit 1
  fi
  if job_grace_elapsed 599 600; then
    echo "watchdog self-test failed: a finished job inside the grace must stay up" >&2
    exit 1
  fi
  # $1.50 at $12/h trips at 448 s (about 7.5 min). That is what a chat
  # pod does if it is billed like the pilot H100 cap instead of an A40.
  if ! cap_due 448 1200 3600 150; then
    echo "watchdog self-test failed: \$1.50 at \$12/h should trip at 448s" >&2
    exit 1
  fi
  if cap_due 447 1200 3600 150; then
    echo "watchdog self-test failed: \$1.50 at \$12/h tripped a second early" >&2
    exit 1
  fi
  if cap_due 450 59 3600 150; then
    echo "watchdog self-test failed: an A40 at \$0.59/h must not hit a \$1.50 cap in 7.5 min" >&2
    exit 1
  fi
  saved_job="${PILOT_WATCH_JOB:-}"
  unset PILOT_WATCH_JOB
  if should_end_job 600 600; then
    echo "watchdog self-test failed: a chat pod must not self-terminate on DONE" >&2
    exit 1
  fi
  PILOT_WATCH_JOB=1
  if ! should_end_job 600 600; then
    echo "watchdog self-test failed: a training pod must self-terminate after the grace" >&2
    exit 1
  fi
  if should_end_job 599 600; then
    echo "watchdog self-test failed: a training pod inside the grace must stay up" >&2
    exit 1
  fi
  if [[ -n "$saved_job" ]]; then
    PILOT_WATCH_JOB="$saved_job"
  else
    unset PILOT_WATCH_JOB
  fi
  echo "watchdog ok (bash)"
}

job_grace_elapsed() {
  # $1 seconds since DONE or FAILED, $2 pull grace seconds.
  local age="$1" grace="$2"
  (( age >= grace ))
}

should_end_job() {
  # Training pods set PILOT_WATCH_JOB=1. A chat pod must not inherit this:
  # a stray DONE file, or the pull grace, would delete the GPU with no
  # terminate line in the launcher log.
  [[ "${PILOT_WATCH_JOB:-}" == 1 ]] && job_grace_elapsed "$1" "$2"
}

flush_artifacts() {
  local out="${PILOT_OUT:-/workspace/pilot-out}"
  local work="${PILOT_WORK:-/workspace/work}"
  mkdir -p "$out/checkpoints" "$out/stores"
  local f rel
  if [[ -d "$work/runs" ]]; then
    while IFS= read -r f; do
      rel="${f#"$work/runs/"}"
      mkdir -p "$out/checkpoints/$(dirname "$rel")"
      cp -f "$f" "$out/checkpoints/$rel" || true
    done < <(find "$work/runs" -type f \( -name best.pt -o -name ckpt.pt \) 2>/dev/null)
  fi
  if [[ -d "$work/stores" ]]; then
    cp -a "$work/stores/." "$out/stores/" 2>/dev/null || true
  fi
}

wait_for_pull() {
  local out="${PILOT_OUT:-/workspace/pilot-out}"
  local grace="${PILOT_CAP_PULL_S:-90}"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$out/CAP"
  local i=0
  while [[ "$i" -lt "$grace" ]]; do
    if [[ -f "$out/PULLED" && "$out/PULLED" -nt "$out/CAP" ]]; then
      echo "artifacts pulled"
      return 0
    fi
    sleep 5
    i=$((i + 5))
  done
  echo "pull grace ${grace}s elapsed"
}

if [[ "${1:-}" == "--once" ]]; then
  terminate_pod
  exit $?
fi
if [[ "${1:-}" == "--stop" ]]; then
  stop_pod
  exit $?
fi
if [[ "${1:-}" == "--self-test" ]]; then
  self_test
  exit 0
fi
self_test

start="${PILOT_BILLING_EPOCH:-$(date +%s)}"
max_s="${PILOT_MAX_SECONDS:-9000}"
max_cents="${PILOT_MAX_USD_CENTS:-1200}"
hourly="${PILOT_HOURLY_CENTS:-439}"
echo "watchdog armed start=${start} wall=${max_s}s cap_cents=${max_cents} hourly_cents=${hourly}"
while true; do
  now=$(date +%s)
  elapsed=$((now - start))
  if (( elapsed < 0 )); then elapsed=0; fi
  # Same rounding as runpod/cost.py: (elapsed * hourly + 3599) / 3600
  cost=$(( (elapsed * hourly + 3599) / 3600 ))
  printf 'watchdog elapsed=%ss cost_cents=%s cap_cents=%s wall_s=%s\n' \
    "$elapsed" "$cost" "$max_cents" "$max_s"
  if cap_due "$elapsed" "$hourly" "$max_s" "$max_cents"; then
    echo "watchdog cap reached"
    flush_artifacts || true
    wait_for_pull || true
    terminate_pod && exit 0
    sleep 15
    continue
  fi
  # The box process can be killed while it is copying files. DONE/FAILED plus
  # the pull grace is enough; this loop does not need the launcher to be alive.
  out="${PILOT_OUT:-/workspace/pilot-out}"
  grace="${PILOT_PULL_GRACE_S:-600}"
  mark=""
  if [[ -f "$out/DONE" ]]; then
    mark="$out/DONE"
  elif [[ -f "$out/FAILED" ]]; then
    mark="$out/FAILED"
  fi
  if [[ -n "$mark" ]]; then
    mtime=$(stat -c %Y "$mark" 2>/dev/null || echo "$now")
    age=$((now - mtime))
    if (( age < 0 )); then age=0; fi
    if should_end_job "$age" "$grace"; then
      echo "watchdog job finished ${age}s ago (grace ${grace}s); terminating"
      terminate_pod && exit 0
      sleep 15
      continue
    fi
  fi
  sleep 20
done
# END WATCHDOG
