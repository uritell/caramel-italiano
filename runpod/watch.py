#!/usr/bin/env python3
"""Reattach to a pod the launcher already created. Does not create a pod.

Reads runpod/retrieved/<pod-id>/attach.json (pod id, pull token, clock).
The API key stays in the environment of this machine. It is not read from
the pod and it is not written into attach.json.

Network errors are retried inside the GraphQL client. This process exits 0
when the job finishes, 2 when the pod is missing three times in a row.
runpod/supervise.sh starts it again if it dies any other way.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import launch  # noqa: E402


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pod", default="", help="pod id; default is the newest attach.json")
    p.add_argument("--ssh-key", default=os.environ.get("PILOT_SSH_KEY", ""))
    args = p.parse_args(argv)
    api_key = os.environ.get("RUNPOD_API_KEY", "")
    if not api_key:
        print("RUNPOD_API_KEY is not set. Nothing was terminated.")
        return 1
    attach_path = _find_attach(args.pod)
    if attach_path is None:
        print("no attach.json found. Pass --pod after a launch has written one.")
        return 1
    meta = json.loads(attach_path.read_text(encoding="utf-8"))
    print(f"{launch._stamp()} reattach {meta['pod_id']} from {attach_path}", flush=True)
    return launch.follow(
        meta["pod_id"], api_key, meta["token"], int(meta["epoch"]),
        int(meta["hourly_cents"]), Path(meta["out"]), args.ssh_key,
    )


def _find_attach(pod: str) -> Path | None:
    root = HERE / "retrieved"
    if pod:
        path = root / pod / "attach.json"
        return path if path.is_file() else None
    found = sorted(root.glob("*/attach.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return found[0] if found else None


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
