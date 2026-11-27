#!/bin/bash
# Restart the reattach watcher if it dies. Exit 0 and exit 2 are finished
# states (job done, or the pod is really gone) and are not restarted.
set -u
cd "$(dirname "$0")/.."
while true; do
  python3 runpod/watch.py "$@"
  code=$?
  printf '%s watcher exited %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$code"
  if [[ "$code" -eq 0 || "$code" -eq 2 ]]; then
    exit "$code"
  fi
  echo "restarting watcher in 5s"
  sleep 5
done
