#!/usr/bin/env python3
"""Rent one small GPU and open a browser chat with the Italian model.

    python runpod/chat_launch.py --dry-run
    python runpod/chat_launch.py --launch --model-dir runpod/retrieved/605n4gr7j7dz7z/caramel-it

The default is the dry run. It does not send an API key and it does not
create a pod. --launch is what spends money.

The weights are not in the git repo. --model-dir is a folder with
config.json and model.safetensors (the same files model.py load() reads).
The launcher uploads that folder through the pod's proxy, the same POST
/upload path the pilot uses for a resume.

GPU order, Secure Cloud only: A40, then RTX 4090, then L40S. A quote above
$1.20 an hour is refused. EU datacenters are preferred. The user's
RUNPOD_API_KEY is not copied into the pod.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import cost
import graphql_api
from launch import IMAGE, docker_args_json, proxy_base, bootstrap_raw_url

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# RunPod gpu type ids. --launch confirms the live securePrice and skips one
# that is missing or priced above the ceiling.
CANDIDATES = (
    ("NVIDIA A40", 59),                 # $0.59/h, operator check 27 Nov 2026
    ("NVIDIA GeForce RTX 4090", 89),    # $0.89/h, same day
    ("NVIDIA L40S", None),              # third choice; price comes from the quote
)
MAX_PRICE_CENTS = 120
DEFAULT_MINUTES = 60
MAX_MINUTES = 120
DEFAULT_CAP_CENTS = 150
DISK_GB = 40
EU_DATACENTERS = ("EU-RO-1", "EU-CZ-1", "EU-NL-1", "EU-FR-1", "EU-SE-1")
STARTUP_LIMIT_S = 12 * 60


def chat_bootstrap_url(repo_url: str, ref: str) -> str:
    raw = bootstrap_raw_url(repo_url, ref)
    if not raw.endswith("/runpod/bootstrap.sh"):
        raise ValueError(f"unexpected bootstrap url {raw}")
    return raw[: -len("bootstrap.sh")] + "chat_bootstrap.sh"


def container_script(raw_url: str) -> str:
    if "'" in raw_url or any(ch in raw_url for ch in "\n\r"):
        raise ValueError("bootstrap URL must be a single shell word")
    return (
        "mkdir -p /workspace/chat-logs\n"
        f"if curl -fsSL --retry 5 --retry-delay 2 -o /tmp/chat-bootstrap.sh '{raw_url}'; then\n"
        "  nohup bash /tmp/chat-bootstrap.sh >>/workspace/chat-logs/job.log 2>&1 &\n"
        "else\n"
        "  echo bootstrap-download-failed >>/workspace/chat-logs/job.log\n"
        "fi\n"
        "exec /start.sh\n"
    )


def build_env(hourly_cents: int, billing_epoch: int, token: str, repo_url: str, repo_ref: str, wall_s: int, cap_cents: int) -> list[dict]:
    env = {
        "PILOT_BILLING_EPOCH": str(billing_epoch),
        "PILOT_MAX_SECONDS": str(wall_s),
        "PILOT_MAX_USD_CENTS": str(cap_cents),
        "PILOT_HOURLY_CENTS": str(hourly_cents),
        "CHAT_TOKEN": token,
        "PILOT_PULL_TOKEN": token,
        "CHAT_MODEL_DIR": "/workspace/chat-model",
        "CHAT_PORT": "8080",
        "CARAMEL_REPO_URL": repo_url,
        "CARAMEL_REPO_REF": repo_ref,
        "HF_HOME": "/workspace/chat-hf",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "PYTHONUNBUFFERED": "1",
    }
    # The pod-scoped RUNPOD_API_KEY is injected by RunPod. Do not copy the user's key.
    return [{"key": k, "value": v} for k, v in env.items()]


def build_create_input(
    gpu_id: str,
    hourly_cents: int,
    billing_epoch: int,
    token: str,
    repo_url: str,
    repo_ref: str,
    wall_s: int,
    cap_cents: int,
    datacenters: tuple[str, ...] = EU_DATACENTERS,
) -> dict:
    deadline = datetime.fromtimestamp(billing_epoch + wall_s, tz=timezone.utc)
    body = {
        "name": "caramel-it-chat",
        "cloudType": "SECURE",
        "gpuCount": 1,
        "gpuTypeId": gpu_id,
        "imageName": IMAGE,
        "containerDiskInGb": DISK_GB,
        "volumeInGb": 0,
        "minVcpuCount": 4,
        "minMemoryInGb": 16,
        "minCudaVersion": "12.8",
        "ports": "8080/http",
        "startSsh": False,
        "startJupyter": False,
        "supportPublicIp": True,
        "dockerArgs": docker_args_json(container_script(chat_bootstrap_url(repo_url, repo_ref))),
        "terminateAfter": deadline.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "env": build_env(hourly_cents, billing_epoch, token, repo_url, repo_ref, wall_s, cap_cents),
    }
    if datacenters:
        body["dataCenterIds"] = list(datacenters)
    return body


def validate(body: dict, wall_s: int = DEFAULT_MINUTES * 60, cap_cents: int = DEFAULT_CAP_CENTS) -> list[str]:
    problems = []
    if body.get("cloudType") != "SECURE":
        problems.append("cloudType must be SECURE")
    if body.get("gpuCount") != 1:
        problems.append("gpuCount must be 1")
    if body.get("containerDiskInGb") != DISK_GB:
        problems.append(f"container disk must be {DISK_GB} GB")
    if body.get("volumeInGb") not in (0, None):
        problems.append("volumeInGb must be 0")
    if "8080/http" not in body.get("ports", ""):
        problems.append("ports must include 8080/http")
    if not body.get("terminateAfter"):
        problems.append("terminateAfter must be set")
    keys = {item["key"]: item["value"] for item in body.get("env") or []}
    if "RUNPOD_API_KEY" in keys:
        problems.append("the user's RUNPOD_API_KEY must not be copied into the pod")
    if keys.get("PILOT_MAX_USD_CENTS") != str(cap_cents):
        problems.append("pod cost cap env does not match")
    if keys.get("PILOT_MAX_SECONDS") != str(wall_s):
        problems.append("pod time cap env does not match")
    if not keys.get("CHAT_TOKEN"):
        problems.append("chat token missing")
    try:
        spec = json.loads(body.get("dockerArgs") or "")
    except json.JSONDecodeError:
        problems.append("dockerArgs is not JSON")
        spec = {}
    cmd = (spec.get("cmd") or [""])[0]
    if spec.get("entrypoint") != ["/bin/bash", "-c"] or not cmd.rstrip().endswith("exec /start.sh"):
        problems.append("start script must be bash -c and must exec /start.sh")
    if "chat_bootstrap.sh" not in cmd:
        problems.append("start script must fetch chat_bootstrap.sh")
    return problems


def stock_ok(status) -> bool:
    if status is None:
        return True
    text = str(status).strip().lower()
    return text not in {"", "none", "null", "unavailable", "out_of_stock", "outofstock"}


def price_cents(quote: dict) -> int | None:
    secure = quote.get("securePrice")
    if isinstance(secure, (int, float)) and not isinstance(secure, bool) and secure > 0:
        return cost.dollars_to_cents_up(float(secure))
    lowest = (quote.get("lowestPrice") or {}).get("uninterruptablePrice")
    if isinstance(lowest, (int, float)) and not isinstance(lowest, bool) and lowest > 0:
        return cost.dollars_to_cents_up(float(lowest))
    return None


def choose_gpu(quotes: list[tuple[str, dict]]) -> tuple[str, int] | None:
    """First candidate at or under $1.20/h with stock. quotes follow CANDIDATES order."""
    for gpu_id, quote in quotes:
        cents = price_cents(quote)
        if cents is None or cents > MAX_PRICE_CENTS:
            continue
        if not stock_ok((quote.get("lowestPrice") or {}).get("stockStatus")):
            continue
        return gpu_id, cents
    return None


def quote_candidates(api_key: str) -> list[tuple[str, dict]]:
    found = []
    for gpu_id, _hint in CANDIDATES:
        try:
            found.append((gpu_id, graphql_api.quote_gpu(gpu_id, api_key)))
        except graphql_api.ApiError as exc:
            print(f"quote {gpu_id} failed: {exc}", flush=True)
            found.append((gpu_id, {}))
    return found


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def upload_model(pod_id: str, token: str, model_dir: Path) -> None:
    import http.client
    from urllib.parse import urlparse

    base = proxy_base(pod_id)
    parsed = urlparse(base)
    files = ("config.json", "model.safetensors")
    extra = model_dir / "SHA256SUMS"
    names = list(files) + (["SHA256SUMS"] if extra.is_file() else [])
    for name in names:
        path = model_dir / name
        size = path.stat().st_size
        rel = f"model/{name}"
        conn = http.client.HTTPSConnection(parsed.hostname, timeout=600)
        with path.open("rb") as fh:
            conn.request("POST", "/upload", body=fh, headers={
                "X-Chat-Token": token,
                "X-Pilot-Token": token,
                "X-Pilot-Path": rel,
                "Content-Length": str(size),
                "Content-Type": "application/octet-stream",
            })
        resp = conn.getresponse()
        resp.read()
        conn.close()
        if resp.status != 200:
            raise RuntimeError(f"upload {rel} returned {resp.status}")
        print(f"uploaded {rel} ({size} bytes)", flush=True)
    conn = http.client.HTTPSConnection(parsed.hostname, timeout=120)
    conn.request("POST", "/ready", body=b"", headers={
        "X-Chat-Token": token,
        "Content-Length": "0",
    })
    resp = conn.getresponse()
    resp.read()
    conn.close()
    if resp.status != 200:
        raise RuntimeError(f"ready returned {resp.status}")
    print("model ready", flush=True)


def _health(pod_id: str, token: str) -> bool:
    url = proxy_base(pod_id) + "/health?token=" + token
    req = urllib.request.Request(url, headers={"X-Chat-Token": token, "User-Agent": "caramel-italiano-chat"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status == 200
    except urllib.error.HTTPError as exc:
        return exc.code in (200, 401)
    except Exception:
        return False


def follow(pod_id: str, api_key: str, token: str, epoch: int, hourly: int, wall_s: int, cap_cents: int, model_dir: Path) -> int:
    missing = 0
    uploaded = False
    healthy = False

    def terminate(reason: str) -> None:
        print(f"{_stamp()} terminate ({reason})", flush=True)
        try:
            graphql_api.terminate_pod(pod_id, api_key)
        except Exception as exc:  # noqa: BLE001
            print(f"TERMINATE FAILED: {exc}", flush=True)

    try:
        while True:
            time.sleep(20)
            try:
                got = graphql_api.pod_status(pod_id, api_key)
            except (graphql_api.ApiError, TimeoutError, OSError) as exc:
                print(f"{_stamp()} status retryable error: {exc}", flush=True)
                continue
            current = got.get("pod") or {}
            if not current:
                try:
                    ids = graphql_api.list_pod_ids(api_key)
                except (graphql_api.ApiError, TimeoutError, OSError) as exc:
                    print(f"{_stamp()} pod list failed: {exc}", flush=True)
                    continue
                if pod_id in ids:
                    missing = 0
                    print(f"{_stamp()} status empty but myself.pods still has {pod_id}", flush=True)
                    continue
                missing += 1
                print(f"{_stamp()} pod missing {missing}/3", flush=True)
                if missing >= 3:
                    print(f"{_stamp()} pod {pod_id} is gone; not calling terminate")
                    return 2
                continue
            missing = 0
            elapsed = int(time.time()) - epoch
            rate = hourly
            raw = current.get("costPerHr")
            if isinstance(raw, (int, float)) and raw > 0:
                rate = cost.dollars_to_cents_up(float(raw))
            spent = cost.cost_cents(elapsed, rate)
            reason = cost.stop_reason(elapsed, rate, wall_s, cap_cents)
            print(
                f"{_stamp()} [cost] {elapsed}s  {cost.format_usd(spent)}  "
                f"rate {cost.format_usd(rate)}/h  cap {cost.format_usd(cap_cents)}",
                flush=True,
            )
            if reason:
                terminate(reason)
                return 0
            if not healthy:
                healthy = _health(pod_id, token)
            if elapsed >= STARTUP_LIMIT_S and not healthy:
                terminate("chat server never answered on the proxy")
                return 1
            if healthy and not uploaded:
                try:
                    upload_model(pod_id, token, model_dir)
                except (TimeoutError, OSError, RuntimeError) as exc:
                    print(f"{_stamp()} upload will retry: {exc}", flush=True)
                    continue
                uploaded = True
                print(f"open this page: {proxy_base(pod_id)}/?token={token}", flush=True)
    except KeyboardInterrupt:
        terminate("interrupted")
        return 1


def _check_model_dir(path: Path | None, required: bool) -> str | None:
    if path is None:
        return "pass --model-dir (config.json and model.safetensors)" if required else None
    if not (path / "config.json").is_file() or not (path / "model.safetensors").is_file():
        return f"{path} needs config.json and model.safetensors"
    return None


def dry_run(minutes: int, cap_cents: int, repo_url: str, repo_ref: str, model_dir: Path | None) -> int:
    wall_s = minutes * 60
    epoch = int(time.time())
    body = build_create_input(CANDIDATES[0][0], CANDIDATES[0][1], epoch, "dry-run-token", repo_url, repo_ref, wall_s, cap_cents)
    problems = validate(body, wall_s, cap_cents)
    model_problem = _check_model_dir(model_dir, required=False)
    half_hour = cost.cost_cents(30 * 60, CANDIDATES[0][1])
    print(f"image: {IMAGE}")
    print("gpu order (Secure Cloud), first one at or under $1.20/h:")
    for gpu_id, hint in CANDIDATES:
        shown = cost.format_usd(hint) + "/h checked 27 Nov 2026" if hint else "live quote only"
        print(f"  - {gpu_id}  {shown}")
    print(f"disk: {DISK_GB} GB container, volumeInGb 0")
    print(f"prefer datacenters: {', '.join(EU_DATACENTERS)}")
    print(f"stop: {minutes} min or {cost.format_usd(cap_cents)}, whichever first, plus RunPod terminateAfter")
    print(f"30 min on the A40 check price is {cost.format_usd(half_hour)}")
    print("the user's RUNPOD_API_KEY is not put in the pod env")
    if model_problem and model_dir is not None:
        problems.append(model_problem)
    if problems:
        print("dry run failed:")
        for item in problems:
            print(f"  - {item}")
        return 1
    hidden = json.loads(json.dumps(body))
    for item in hidden["env"]:
        if item["key"] in {"CHAT_TOKEN", "PILOT_PULL_TOKEN"}:
            item["value"] = "***"
    print(json.dumps(hidden, indent=2))
    print("Dry run finished. No pod was created. No RunPod API key was sent.")
    return 0


def launch(minutes: int, cap_cents: int, repo_url: str, repo_ref: str, model_dir: Path) -> int:
    import secrets

    problem = _check_model_dir(model_dir, required=True)
    if problem:
        print(problem)
        return 1
    api_key = os.environ.get("RUNPOD_API_KEY", "")
    if not api_key:
        print("RUNPOD_API_KEY is not set. Nothing was created.")
        return 1
    if minutes > MAX_MINUTES or minutes < 1:
        print(f"--minutes must be 1..{MAX_MINUTES}")
        return 1
    picked = choose_gpu(quote_candidates(api_key))
    if picked is None:
        print(f"no candidate GPU at or under {cost.format_usd(MAX_PRICE_CENTS)}/h. Nothing was created.")
        return 1
    gpu_id, hourly = picked
    print(f"using {gpu_id} at {cost.format_usd(hourly)}/h", flush=True)
    token = secrets.token_urlsafe(24)
    epoch = int(time.time())
    wall_s = minutes * 60
    body = build_create_input(gpu_id, hourly, epoch, token, repo_url, repo_ref, wall_s, cap_cents)
    problems = validate(body, wall_s, cap_cents)
    if problems:
        print("refusing to create:")
        for item in problems:
            print(f"  - {item}")
        return 1
    print("creating the pod", flush=True)
    try:
        created = graphql_api.create_pod(body, api_key)
    except graphql_api.ApiError as exc:
        print(f"create with EU datacenters failed ({exc}); retrying without a datacenter list", flush=True)
        body.pop("dataCenterIds", None)
        created = graphql_api.create_pod(body, api_key)
    pod_id = str((created.get("pod") or {}).get("id") or "")
    if not pod_id:
        print(f"create response had no id: {created}")
        return 1
    print(f"pod {pod_id}", flush=True)
    print(f"chat URL (after the model upload finishes): {proxy_base(pod_id)}/?token={token}", flush=True)
    return follow(pod_id, api_key, token, epoch, hourly, wall_s, cap_cents, model_dir)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--launch", action="store_true", help="rent the pod; this spends money")
    p.add_argument("--dry-run", action="store_true", help="check the plan only (this is the default)")
    p.add_argument("--model-dir", type=Path, default=None)
    p.add_argument("--minutes", type=int, default=DEFAULT_MINUTES)
    p.add_argument("--max-usd", type=float, default=DEFAULT_CAP_CENTS / 100)
    p.add_argument("--repo-url", default=os.environ.get("CARAMEL_REPO_URL", "https://github.com/uritell/caramel-italiano.git"))
    p.add_argument("--repo-ref", default=os.environ.get("CARAMEL_REPO_REF", "main"))
    args = p.parse_args(argv)
    if args.minutes > MAX_MINUTES or args.minutes < 1:
        print(f"--minutes must be 1..{MAX_MINUTES}")
        return 1
    cap_cents = cost.dollars_to_cents_up(args.max_usd)
    if args.launch:
        if args.model_dir is None:
            print("--launch needs --model-dir")
            return 1
        return launch(args.minutes, cap_cents, args.repo_url, args.repo_ref, args.model_dir)
    return dry_run(args.minutes, cap_cents, args.repo_url, args.repo_ref, args.model_dir)


if __name__ == "__main__":
    sys.exit(main())
