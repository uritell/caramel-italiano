#!/usr/bin/env python3
"""Lean data pod: A40, VoxPopuli 2015 plus the rest of MLS Italian.

    python3 runpod/lean_launch.py --dry-run
    python3 runpod/lean_launch.py --launch

The default is the dry run. It does not send an API key and it does not
create a pod. --launch is what spends money. The pod encodes five minutes
first. If that realtime factor puts the full run over the cap ($3.00 unless
--cap says otherwise), the launcher terminates and the full encode does not start.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import cost
import graphql_api
import lean_data
from chat_launch import CANDIDATES, EU_DATACENTERS, choose_gpu
from chat_upload import notify_ready
from launch import IMAGE, docker_args_json, proxy_base, pull_outputs, pull_server_answered, bootstrap_raw_url

HERE = Path(__file__).resolve().parent


def lean_bootstrap_url(repo_url: str, ref: str) -> str:
    raw = bootstrap_raw_url(repo_url, ref)
    if not raw.endswith("/runpod/bootstrap.sh"):
        raise ValueError(f"unexpected bootstrap url {raw}")
    return raw[: -len("bootstrap.sh")] + "lean_bootstrap.sh"


def container_script(raw_url: str) -> str:
    if "'" in raw_url or any(ch in raw_url for ch in "\n\r"):
        raise ValueError("bootstrap URL must be a single shell word")
    return (
        "mkdir -p /workspace/pilot-out/logs /workspace/work\n"
        f"if curl -fsSL --retry 5 --retry-delay 2 -o /tmp/lean-bootstrap.sh '{raw_url}'; then\n"
        "  nohup bash /tmp/lean-bootstrap.sh >>/workspace/pilot-out/logs/job.log 2>&1 &\n"
        "else\n"
        "  echo bootstrap-download-failed >>/workspace/pilot-out/logs/job.log\n"
        "fi\n"
        "exec /start.sh\n"
    )


def build_env(
    hourly_cents: int, billing_epoch: int, token: str, repo_url: str, repo_ref: str,
    cap_cents: int = lean_data.MAX_USD_CENTS,
) -> list[dict]:
    env = {
        "PILOT_BILLING_EPOCH": str(billing_epoch),
        "PILOT_MAX_SECONDS": str(lean_data.MAX_WALL_S),
        "PILOT_MAX_USD_CENTS": str(cap_cents),
        "PILOT_HOURLY_CENTS": str(hourly_cents),
        "PILOT_WATCH_JOB": "1",
        "PILOT_PULL_TOKEN": token,
        "PILOT_OUT": "/workspace/pilot-out",
        "PILOT_PULL_PORT": "8080",
        "PILOT_LEAN_DATA": "1",
        "PILOT_ENCODE_BUDGET_S": str(lean_data.encode_budget_s()),
        "CARAMEL_REPO_URL": repo_url,
        "CARAMEL_REPO_REF": repo_ref,
        "HF_HOME": "/workspace/work/hf",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "PYTHONUNBUFFERED": "1",
    }
    return [{"key": k, "value": v} for k, v in env.items()]


def build_create_input(
    gpu_id: str, hourly_cents: int, billing_epoch: int, token: str, repo_url: str, repo_ref: str,
    cap_cents: int = lean_data.MAX_USD_CENTS,
) -> dict:
    deadline = datetime.fromtimestamp(billing_epoch + lean_data.MAX_WALL_S, tz=timezone.utc)
    return {
        "name": "caramel-it-lean-data",
        "cloudType": "SECURE",
        "gpuCount": 1,
        "gpuTypeId": gpu_id,
        "imageName": IMAGE,
        "containerDiskInGb": lean_data.DISK_GB,
        "volumeInGb": 0,
        "minVcpuCount": 4,
        "minMemoryInGb": 16,
        "minCudaVersion": "12.8",
        "ports": "8080/http",
        "startSsh": False,
        "startJupyter": False,
        "supportPublicIp": True,
        "dockerArgs": docker_args_json(container_script(lean_bootstrap_url(repo_url, repo_ref))),
        "terminateAfter": deadline.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "env": build_env(hourly_cents, billing_epoch, token, repo_url, repo_ref, cap_cents),
        "dataCenterIds": list(EU_DATACENTERS),
    }


def validate(body: dict, cap_cents: int = lean_data.MAX_USD_CENTS) -> list[str]:
    problems = []
    if body.get("cloudType") != "SECURE":
        problems.append("cloudType must be SECURE")
    if body.get("gpuCount") != 1:
        problems.append("gpuCount must be 1")
    if body.get("gpuTypeId") != "NVIDIA A40":
        problems.append("lean data uses the A40")
    if body.get("containerDiskInGb") != lean_data.DISK_GB:
        problems.append(f"disk must be {lean_data.DISK_GB} GB")
    if not body.get("terminateAfter"):
        problems.append("terminateAfter must be set")
    keys = {item["key"]: item["value"] for item in body.get("env") or []}
    if "RUNPOD_API_KEY" in keys:
        problems.append("the user's RUNPOD_API_KEY must not be copied into the pod")
    if keys.get("PILOT_MAX_USD_CENTS") != str(cap_cents):
        problems.append(f"cap is not ${cap_cents / 100:.2f}")
    if keys.get("PILOT_MAX_SECONDS") != str(lean_data.MAX_WALL_S):
        problems.append("wall is not 6 h")
    if keys.get("PILOT_WATCH_JOB") != "1":
        problems.append("the pod watchdog must terminate after the job and the pull grace")
    if "lean_bootstrap.sh" not in (body.get("dockerArgs") or ""):
        problems.append("start script must fetch lean_bootstrap.sh")
    ref = lean_data.reference_projection(int(keys.get("PILOT_HOURLY_CENTS") or lean_data.A40_HOURLY_CENTS))
    if lean_data.over_cap(ref["cents"], cap_cents):
        problems.append(
            f"reference projection ${ref['cents'] / 100:.2f} is over the ${cap_cents / 100:.2f} cap"
        )
    return problems


def dry_run(repo_url: str, repo_ref: str, cap_cents: int = lean_data.MAX_USD_CENTS) -> int:
    body = build_create_input(
        "NVIDIA A40", lean_data.A40_HOURLY_CENTS, int(time.time()), "dry-run-token", repo_url, repo_ref,
        cap_cents,
    )
    problems = validate(body, cap_cents)
    print(lean_data.explain_plan(cap_cents))
    print()
    print("pull: tokens.bin, index.npy, meta.json, ATTRIBUTION.txt. Streamed. No raw audio.")
    print("pull server: lean_bootstrap.sh starts pull_server.py on 8080 before any download.")
    print("if 8080 is not reachable within 10 min of RUNNING, the launcher terminates.")
    print("Dry run finished. No pod was created. No RunPod API key was sent.")
    if problems:
        print("dry run failed:")
        for item in problems:
            print(f"  - {item}")
        return 1
    return 0


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


HEARTBEAT_STALE_S = 10 * 60
PULL_TIMEOUT_S = 60
PULL_ATTEMPTS = 3
PULL_DEADLINE_S = 120


def heartbeat_stale(doc: dict, now: float, limit_s: float = HEARTBEAT_STALE_S) -> bool:
    """True when status.json is older than the limit. Missing fields are not stale."""
    raw = doc.get("ts") or doc.get("main_loop_ts")
    if not raw:
        return False
    try:
        stamp = datetime.strptime(str(raw), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return False
    return now - stamp.timestamp() >= limit_s


def read_status(out_dir: Path) -> dict | None:
    path = out_dir / "status.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def pull_with_deadline(
    pod: dict, out_dir: Path, token: str, *,
    timeout: int = PULL_TIMEOUT_S, attempts: int = PULL_ATTEMPTS, deadline: float = PULL_DEADLINE_S,
    pull=None,
) -> None:
    """One pull at a time. A slow pull logs and the loop keeps going."""
    if pull is None:
        pull = pull_outputs
    holder = getattr(pull_with_deadline, "thread", None)
    if holder is not None and holder.is_alive():
        print(f"{_stamp()} pull stuck, continuing", flush=True)
        return

    def work() -> None:
        try:
            pull(pod, out_dir, token, "", timeout=timeout, attempts=attempts)
        except Exception as exc:  # noqa: BLE001
            print(f"{_stamp()} pull retry: {exc}", flush=True)

    thread = threading.Thread(target=work, name="lean-pull", daemon=True)
    pull_with_deadline.thread = thread
    thread.start()
    thread.join(deadline)
    if thread.is_alive():
        print(f"{_stamp()} pull stuck, continuing", flush=True)


def follow(
    pod_id: str, api_key: str, token: str, epoch: int, hourly: int, out_dir: Path,
    cap_cents: int = lean_data.MAX_USD_CENTS, sleep=None,
) -> int:
    """Watch the pod. A speed-test projection over the cap terminates before the full encode."""
    out_dir.mkdir(parents=True, exist_ok=True)
    decided = False
    running_at = None
    proxy_up = False
    proxy_since = None
    last_doc: dict | None = None
    pull_with_deadline.thread = None
    pause = time.sleep if sleep is None else sleep

    def terminate(reason: str) -> None:
        print(f"{_stamp()} terminate ({reason})", flush=True)
        try:
            pull_outputs({"id": pod_id}, out_dir, token, "", timeout=PULL_TIMEOUT_S, attempts=PULL_ATTEMPTS)
        except Exception as exc:  # noqa: BLE001
            print(f"pull before terminate failed: {exc}", flush=True)
        try:
            graphql_api.terminate_pod(pod_id, api_key)
        except Exception as exc:  # noqa: BLE001
            print(f"TERMINATE FAILED: {exc}", flush=True)

    while True:
        print(f"{_stamp()} tick", flush=True)
        pause(20)
        try:
            got = graphql_api.pod_status(pod_id, api_key)
        except (graphql_api.ApiError, TimeoutError, OSError) as exc:
            print(f"{_stamp()} status retryable error: {exc}", flush=True)
            continue
        current = got.get("pod") or {}
        if not current:
            print(f"{_stamp()} pod {pod_id} is gone", flush=True)
            print(
                f"{_stamp()} last status.json {json.dumps(last_doc) if last_doc else 'none'}",
                flush=True,
            )
            print(
                f"{_stamp()} FAILED={(out_dir / 'FAILED').is_file()} DONE={(out_dir / 'DONE').is_file()}",
                flush=True,
            )
            return 2
        elapsed = int(time.time()) - epoch
        status = str(current.get("desiredStatus") or current.get("status") or "")
        if status == "RUNNING":
            if running_at is None:
                running_at = time.monotonic()
            if not proxy_up:
                proxy_up = pull_server_answered(pod_id, token)
            if proxy_up and proxy_since is None:
                proxy_since = time.monotonic()
            if lean_data.proxy_missed(time.monotonic() - running_at, proxy_up):
                terminate("pull server on 8080 did not answer within 10 min of RUNNING")
                return 1
        rate = hourly
        raw = current.get("costPerHr")
        if isinstance(raw, (int, float)) and raw > 0:
            rate = cost.dollars_to_cents_up(float(raw))
        spent = cost.cost_cents(elapsed, rate)
        reason = cost.stop_reason(elapsed, rate, lean_data.MAX_WALL_S, cap_cents)
        print(f"{_stamp()} [cost] {elapsed}s {cost.format_usd(spent)} status {status}", flush=True)
        pull_with_deadline(current, out_dir, token)
        doc = read_status(out_dir)
        if doc:
            last_doc = doc
            print(
                f"{_stamp()} status.json phase={doc.get('phase')} member={doc.get('current_member')} "
                f"audio_h_done={doc.get('audio_h_done')} frames={doc.get('frames')} "
                f"end_to_end_rtf={doc.get('end_to_end_rtf')} main_loop_ts={doc.get('main_loop_ts')}",
                flush=True,
            )
        if status == "RUNNING" and doc is not None and heartbeat_stale(doc, time.time()):
            terminate("heartbeat stale")
            return 1
        if (out_dir / "FAILED").is_file():
            terminate("FAILED")
            return 1
        if proxy_since is not None and lean_data.speed_report_missed(
            time.monotonic() - proxy_since, (out_dir / "speed_test.json").is_file(),
        ):
            terminate("no speed_test.json within 15 min of 8080 answering")
            return 1
        report_path = out_dir / "speed_test.json"
        if report_path.is_file() and not decided:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if lean_data.over_cap(int(report.get("cents") or 0), cap_cents):
                terminate(
                    f"speed test projects ${int(report['cents']) / 100:.2f}, over the cap"
                )
                return 1
            try:
                notify_ready(proxy_base(pod_id), token)
            except Exception as exc:  # noqa: BLE001
                print(f"{_stamp()} approve retry: {exc}", flush=True)
                continue
            decided = True
            print(
                f"speed test under the cap (encode {report.get('rtf')}x, "
                f"end to end {report.get('end_to_end_rtf')}x); bulk download may start",
                flush=True,
            )
        if reason:
            terminate(reason)
            return 0
        if (out_dir / "DONE").is_file() or (out_dir / "FAILED").is_file():
            terminate("lean data finished")
            return 0 if (out_dir / "DONE").is_file() else 1


def launch(repo_url: str, repo_ref: str, out: Path, cap_cents: int = lean_data.MAX_USD_CENTS) -> int:
    import secrets

    api_key = os.environ.get("RUNPOD_API_KEY", "")
    if not api_key:
        print("RUNPOD_API_KEY is not set. Nothing was created.")
        return 1
    quotes = []
    for gpu_id, _hint in CANDIDATES:
        if gpu_id != "NVIDIA A40":
            continue
        try:
            quotes.append((gpu_id, graphql_api.quote_gpu(gpu_id, api_key)))
        except graphql_api.ApiError as exc:
            print(f"quote {gpu_id} failed: {exc}", flush=True)
    picked = choose_gpu(quotes)
    if picked is None:
        print("A40 is not available under $1.20/h. Nothing was created.")
        return 1
    gpu_id, hourly = picked
    token = secrets.token_urlsafe(24)
    epoch = int(time.time())
    body = build_create_input(gpu_id, hourly, epoch, token, repo_url, repo_ref, cap_cents)
    problems = validate(body, cap_cents)
    if problems:
        print("refusing to create:")
        for item in problems:
            print(f"  - {item}")
        return 1
    print("creating the pod", flush=True)
    created = graphql_api.create_pod(body, api_key)
    pod_id = str((created.get("pod") or {}).get("id") or "")
    if not pod_id:
        print(f"create response had no id: {created}")
        return 1
    print(f"pod {pod_id}", flush=True)
    return follow(pod_id, api_key, token, epoch, hourly, out / pod_id, cap_cents)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--launch", action="store_true", help="rent the A40; this spends money")
    p.add_argument("--dry-run", action="store_true", help="print the plan only (this is the default)")
    p.add_argument("--repo-url", default=os.environ.get("CARAMEL_REPO_URL", "https://github.com/uritell/caramel-italiano.git"))
    p.add_argument("--repo-ref", default=os.environ.get("CARAMEL_REPO_REF", "main"))
    p.add_argument("--out", type=Path, default=HERE / "retrieved")
    p.add_argument("--cap", type=float, default=lean_data.MAX_USD_CENTS / 100,
                   help="dollar cap for this data pod (default 3.00). The speed test is checked against it.")
    args = p.parse_args(argv)
    if args.cap <= 0:
        print("--cap must be a positive number of dollars")
        return 2
    cap_cents = cost.dollars_to_cents_up(args.cap)
    if args.launch and args.dry_run:
        print("Pass either --launch or --dry-run.")
        return 2
    if not args.launch:
        return dry_run(args.repo_url, args.repo_ref, cap_cents)
    return launch(args.repo_url, args.repo_ref, args.out, cap_cents)


if __name__ == "__main__":
    sys.exit(main())
