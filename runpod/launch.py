#!/usr/bin/env python3
"""One-command pilot on a 2× H100 SXM RunPod. Default is a dry run.

    python runpod/launch.py --dry-run     # checks only, creates nothing
    python runpod/launch.py --launch      # rents the pod (needs RUNPOD_API_KEY)

Create, status, stop, and terminate use GraphQL at
https://api.runpod.io/graphql. REST on https://api.runpod.io/v2 is tried
only when that call fails. The cap terminates the pod (it does not stop it).
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import secrets
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cost
import graphql_api
import stages

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
IMAGE = "runpod/pytorch:2.2.0-cu1281-torch291-ubuntu2404"
GPU_ID = "NVIDIA H100 80GB HBM3"
REPO_URL_DEFAULT = "https://github.com/uritell/caramel-italiano.git"


def _read_bootstrap() -> str:
    return (HERE / "bootstrap.sh").read_text(encoding="utf-8")


def build_env(hourly_cents: int, billing_epoch: int, token: str, repo_url: str, repo_ref: str) -> list[dict]:
    bootstrap = base64.b64encode(_read_bootstrap().encode()).decode()
    env = {
        "PILOT_BOOTSTRAP_B64": bootstrap,
        "PILOT_BILLING_EPOCH": str(billing_epoch),
        "PILOT_MAX_SECONDS": str(cost.MAX_WALL_S),
        "PILOT_MAX_USD_CENTS": str(cost.MAX_USD_CENTS),
        "PILOT_HOURLY_CENTS": str(hourly_cents),
        "PILOT_GPU_COUNT": str(cost.GPU_COUNT),
        "PILOT_PULL_TOKEN": token,
        "PILOT_OUT": "/workspace/pilot-out",
        "PILOT_PULL_PORT": "8080",
        "CARAMEL_REPO_URL": repo_url,
        "CARAMEL_REPO_REF": repo_ref,
        "HF_HOME": "/workspace/work/hf",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "PYTHONUNBUFFERED": "1",
    }
    key = os.environ.get("RUNPOD_API_KEY", "")
    if key:
        env["RUNPOD_API_KEY"] = key
    gh = os.environ.get("GITHUB_TOKEN", "")
    if gh:
        env["GITHUB_TOKEN"] = gh
    return [{"key": k, "value": v} for k, v in env.items()]


def docker_command() -> str:
    # One shell command. The script itself is in PILOT_BOOTSTRAP_B64 so the
    # command line stays short. This replaces the image CMD (/start.sh).
    return 'base64 -d <<<"$PILOT_BOOTSTRAP_B64" | bash'


def build_create_input(hourly_cents: int, billing_epoch: int, token: str, repo_url: str, repo_ref: str) -> dict:
    deadline = datetime.fromtimestamp(billing_epoch + cost.MAX_WALL_S, tz=timezone.utc)
    return {
        "name": "caramel-it-pilot",
        "cloudType": "SECURE",
        "gpuCount": cost.GPU_COUNT,
        "gpuTypeId": GPU_ID,
        "imageName": IMAGE,
        "containerDiskInGb": cost.CONTAINER_DISK_GB,
        "volumeInGb": 0,
        "minVcpuCount": 8,
        "minMemoryInGb": 16,
        "minCudaVersion": "12.8",
        "ports": "22/tcp,8080/http",
        "startSsh": True,
        "startJupyter": False,
        "supportPublicIp": True,
        "dockerArgs": docker_command(),
        "terminateAfter": deadline.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "env": build_env(hourly_cents, billing_epoch, token, repo_url, repo_ref),
    }


def redact(body: dict) -> dict:
    hidden = {"RUNPOD_API_KEY", "GITHUB_TOKEN", "PILOT_BOOTSTRAP_B64", "PILOT_PULL_TOKEN"}
    out = json.loads(json.dumps(body))
    for item in out.get("env") or []:
        if item.get("key") in hidden:
            item["value"] = "***"
    return out


def validate_input(body: dict) -> list[str]:
    problems = []
    if body.get("cloudType") != "SECURE":
        problems.append("cloudType must be SECURE")
    if body.get("gpuCount") != 2:
        problems.append("gpuCount must be 2")
    if "H100" not in body.get("gpuTypeId", "") or "HBM3" not in body.get("gpuTypeId", ""):
        problems.append("gpuTypeId must be the H100 SXM (HBM3) id")
    if body.get("volumeInGb") not in (0, None):
        problems.append("volumeInGb must be 0 so no volume disk is created")
    if body.get("networkVolumeId"):
        problems.append("networkVolumeId must be unset")
    if "cu128" not in body.get("imageName", "") or "torch291" not in body.get("imageName", ""):
        problems.append(f"unexpected image {body.get('imageName')}")
    if body.get("minCudaVersion") != "12.8":
        problems.append("minCudaVersion must be 12.8")
    if "22/tcp" not in body.get("ports", "") or "8080/http" not in body.get("ports", ""):
        problems.append("ports must expose 22/tcp and 8080/http")
    if not body.get("startSsh"):
        problems.append("startSsh must be true")
    if not body.get("terminateAfter"):
        problems.append("terminateAfter must be set")
    if body.get("stopAfter"):
        problems.append("stopAfter must be unset; the cap terminates")
    keys = {item["key"] for item in body.get("env") or []}
    for required in ("PILOT_MAX_SECONDS", "PILOT_MAX_USD_CENTS", "PILOT_HOURLY_CENTS", "PILOT_BILLING_EPOCH"):
        if required not in keys:
            problems.append(f"env missing {required}")
    blob = json.dumps(body)
    if "rest.runpod.io" in blob:
        problems.append("create input mentions the unreachable REST host")
    if len(blob) > 90000:
        problems.append(f"create input is {len(blob)} bytes, close to the request limit")
    return problems


def _per_gpu_cents(quote: dict) -> int | None:
    """securePrice is USD per GPU-hour. Fall back to lowestPrice if needed."""
    secure = quote.get("securePrice")
    if isinstance(secure, (int, float)) and secure > 0:
        return cost.dollars_to_cents_up(float(secure))
    lowest = (quote.get("lowestPrice") or {}).get("uninterruptablePrice")
    if isinstance(lowest, (int, float)) and lowest > 0:
        # A figure above $12 is the 2-GPU total, not the per-GPU rate.
        value = float(lowest)
        if value > 12:
            value = value / cost.GPU_COUNT
        return cost.dollars_to_cents_up(value)
    return None


def check_quote(api_key: str) -> tuple[int, dict]:
    quote = graphql_api.quote_gpu(GPU_ID, api_key)
    if not quote:
        raise SystemExit(f"GraphQL gpuTypes has no entry for {GPU_ID}")
    if quote.get("secureCloud") is False:
        raise SystemExit(f"{GPU_ID} is not listed on Secure Cloud")
    max_count = quote.get("maxGpuCountSecureCloud")
    if isinstance(max_count, int) and max_count < cost.GPU_COUNT:
        raise SystemExit(f"Secure Cloud max GPU count for {GPU_ID} is {max_count}")
    cents = _per_gpu_cents(quote)
    if cents is None:
        raise SystemExit(f"no Secure Cloud price on the quote: {json.dumps(quote)[:500]}")
    reason = cost.price_ok(cents)
    if reason:
        raise SystemExit(reason)
    return cents, quote


def print_plan(body: dict, hourly_cents: int) -> None:
    stop_at, why = cost.first_stop(hourly_cents)
    wall_cost = cost.cost_cents(cost.MAX_WALL_S, hourly_cents)
    print(f"image: {IMAGE}")
    print(f"gpu:   {cost.GPU_COUNT}x {GPU_ID}  Secure Cloud")
    print(f"disk:  {cost.CONTAINER_DISK_GB} GB container disk, volumeInGb 0, no network volume")
    print(f"rate:  {cost.format_usd(hourly_cents)} per pod-hour")
    print(f"cap:   {cost.MAX_WALL_S / 3600:.1f} h or {cost.format_usd(cost.MAX_USD_CENTS)}, whichever first")
    print(f"       at this rate the first stop is {stop_at}s ({why})")
    print(f"       bill at 2.5 h would be {cost.format_usd(wall_cost)}")
    print(f"api:   {graphql_api.GRAPHQL_URL}")
    print("ops:   podFindAndDeployOnDemand, pod, podStop, podTerminate")
    print()
    print(stages.format_commands(stages.full_commands(
        "python", Path("tokenize/out/mono"), Path("tokenize/out/stereo"),
        Path("tokenize/out/dev"), Path("runs"))))


def dry_run(repo_url: str, repo_ref: str) -> int:
    epoch = int(time.time())
    body = build_create_input(cost.QUOTED_HOURLY_CENTS, epoch, "dry-run-token", repo_url, repo_ref)
    problems = validate_input(body)
    script_problems = _local_scripts()
    problems.extend(script_problems)
    print_plan(body, cost.QUOTED_HOURLY_CENTS)
    print()
    print("create input (secrets removed):")
    print(json.dumps(redact(body), indent=2)[:4000])
    print()
    head = subprocess.run(
        [sys.executable, str(HERE / "download_open.py"), "--head"],
        cwd=str(ROOT),
    )
    if head.returncode != 0:
        problems.append("a public data URL returned 404 or 410")
    check = subprocess.run([sys.executable, str(HERE / "self_check.py")], cwd=str(ROOT))
    if check.returncode != 0:
        problems.append("self_check failed")
    if problems:
        print("dry run failed:")
        for item in problems:
            print(f"  - {item}")
        return 1
    print("Dry run finished. No pod was created. No RunPod API key was sent.")
    return 0


def _local_scripts() -> list[str]:
    problems = []
    for name in ("pod_run.sh", "bootstrap.sh", "watchdog.sh"):
        path = HERE / name
        if not path.exists():
            problems.append(f"missing {name}")
            continue
        proc = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
        if proc.returncode != 0:
            problems.append(f"bash -n {name}: {proc.stderr.strip()}")
    return problems


def _public_port(pod: dict, private: int) -> tuple[str, int] | None:
    runtime = pod.get("runtime") or {}
    for port in runtime.get("ports") or []:
        if port.get("privatePort") == private and port.get("ip") and port.get("publicPort"):
            return str(port["ip"]), int(port["publicPort"])
    return None


def _ssh_target(pod: dict) -> tuple[str, int, str] | None:
    direct = _public_port(pod, 22)
    if direct:
        return direct[0], direct[1], "root"
    host_id = ((pod.get("machine") or {}).get("podHostId")) or ""
    if host_id:
        return "ssh.runpod.io", 22, host_id
    return None


def _pull_ssh(pod: dict, dest: Path, key_path: str) -> bool:
    target = _ssh_target(pod)
    if target is None or shutil.which("scp") is None:
        return False
    host, port, user = target
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [
        "scp", "-r", "-P", str(port),
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=20",
    ]
    if key_path:
        cmd += ["-i", key_path]
    cmd += [f"{user}@{host}:/workspace/pilot-out/.", str(dest)]
    print("pull via scp", " ".join(cmd[:6]), "...", flush=True)
    proc = subprocess.run(cmd)
    return proc.returncode == 0 and any(dest.iterdir())


def _http_get(url: str, token: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"X-Pilot-Token": token, "User-Agent": "caramel-italiano-pilot"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _pull_http(pod: dict, dest: Path, token: str) -> bool:
    found = _public_port(pod, 8080)
    if not found:
        return False
    ip, port = found
    base = f"http://{ip}:{port}"
    try:
        listing = json.loads(_http_get(base + "/", token))
    except Exception as exc:  # noqa: BLE001
        print(f"HTTP list failed: {exc}", flush=True)
        return False
    files = listing.get("files") or []
    if not files:
        return False
    for rel in files:
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        data = _http_get(base + "/" + rel, token, timeout=300)
        target.write_bytes(data)
        print(f"pulled {rel} ({len(data)} bytes)", flush=True)
    return True


def pull_outputs(pod: dict, dest: Path, token: str, key_path: str) -> str:
    if _pull_ssh(pod, dest, key_path):
        return "ssh"
    if _pull_http(pod, dest, token):
        return "http"
    return ""


def _hourly_from_pod(pod: dict, fallback: int) -> int:
    raw = pod.get("costPerHr")
    if isinstance(raw, (int, float)) and raw > 0:
        return cost.dollars_to_cents_up(float(raw))
    per = ((pod.get("machine") or {}).get("currentPricePerGpu"))
    if isinstance(per, (int, float)) and per > 0:
        return cost.dollars_to_cents_up(float(per) * cost.GPU_COUNT)
    return fallback


def _status_name(pod: dict) -> str:
    return str(pod.get("desiredStatus") or pod.get("status") or "")


def launch(repo_url: str, repo_ref: str, out_dir: Path, ssh_key: str) -> int:
    api_key = os.environ.get("RUNPOD_API_KEY", "")
    if not api_key:
        print("RUNPOD_API_KEY is not set. Nothing was created.")
        return 1
    per_gpu, quote = check_quote(api_key)
    hourly = cost.hourly_cents(per_gpu)
    print(
        f"catalog {quote.get('id')} securePrice={quote.get('securePrice')} "
        f"-> {cost.format_usd(hourly)}/pod-hour",
        flush=True,
    )
    token = secrets.token_urlsafe(24)
    epoch = int(time.time())
    body = build_create_input(hourly, epoch, token, repo_url, repo_ref)
    problems = validate_input(body)
    if problems:
        print("refusing to create:")
        for item in problems:
            print(f"  - {item}")
        return 1
    print_plan(body, hourly)
    print("creating the pod via GraphQL podFindAndDeployOnDemand", flush=True)
    created = graphql_api.create_pod(body, api_key)
    pod = created["pod"]
    pod_id = str(pod.get("id") or "")
    if not pod_id:
        print(f"create response had no id: {created}")
        return 1
    print(f"pod {pod_id} via {created['via']}", flush=True)
    out_dir = out_dir / pod_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "create.json").write_text(json.dumps(redact({"pod": pod, "via": created["via"]}), indent=2) + "\n")
    print(f"pull token (for a manual HTTP fetch): {token}")
    print(f"outputs will land in {out_dir}")

    def terminate(reason: str) -> None:
        print(f"terminate ({reason})", flush=True)
        try:
            latest = graphql_api.pod_status(pod_id, api_key).get("pod") or pod
            how = pull_outputs(latest, out_dir, token, ssh_key)
            print(f"pull before terminate: {how or 'nothing retrieved'}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"pull before terminate failed: {exc}", flush=True)
        try:
            done = graphql_api.terminate_pod(pod_id, api_key)
            print(f"terminated via {done.get('via')}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"TERMINATE FAILED: {exc}", flush=True)
            print(f"Delete pod {pod_id} by hand. It may still be billing.")

    try:
        while True:
            time.sleep(20)
            got = graphql_api.pod_status(pod_id, api_key)
            current = got.get("pod") or {}
            elapsed = int(time.time()) - epoch
            rate = _hourly_from_pod(current, hourly)
            spent = cost.cost_cents(elapsed, rate)
            reason = cost.stop_reason(elapsed, rate)
            status = _status_name(current)
            print(
                f"[cost] {elapsed}s  {cost.format_usd(spent)}  "
                f"rate {cost.format_usd(rate)}/h  status {status or 'unknown'}  via {got.get('via')}",
                flush=True,
            )
            if reason:
                terminate(reason)
                return 0
            if status in {"TERMINATED", "EXITED", "ERROR"} or current.get("desiredStatus") in {"TERMINATED"}:
                terminate(f"pod status {status}")
                return 0 if status == "EXITED" else 1
            # A finished job writes DONE. Pull it, then terminate.
            if status == "RUNNING":
                marker = out_dir / "DONE"
                if not marker.exists():
                    pull_outputs(current, out_dir, token, ssh_key)
                if (out_dir / "DONE").exists():
                    terminate("pilot finished")
                    return 0
    except KeyboardInterrupt:
        terminate("interrupted")
        return 1


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--launch", action="store_true", help="create the pod and watch it")
    p.add_argument("--dry-run", action="store_true", help="validate only (this is also the default)")
    p.add_argument("--repo-url", default=os.environ.get("CARAMEL_REPO_URL", REPO_URL_DEFAULT))
    p.add_argument("--repo-ref", default=os.environ.get("CARAMEL_REPO_REF", "main"))
    p.add_argument("--out", type=Path, default=HERE / "retrieved")
    p.add_argument("--ssh-key", default=os.environ.get("PILOT_SSH_KEY", ""))
    args = p.parse_args(argv)
    if args.launch and args.dry_run:
        print("Pass either --launch or --dry-run.")
        return 2
    if not args.launch:
        return dry_run(args.repo_url, args.repo_ref)
    return launch(args.repo_url, args.repo_ref, args.out, args.ssh_key)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
