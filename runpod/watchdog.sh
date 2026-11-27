#!/bin/bash
# In-pod cap. Started by bootstrap.sh before the clone, and again by
# pod_run.sh if that first copy is gone. Terminate, not stop.
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
  echo "watchdog ok (bash)"
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
    terminate_pod && exit 0
    sleep 15
    continue
  fi
  sleep 20
done
# END WATCHDOG
