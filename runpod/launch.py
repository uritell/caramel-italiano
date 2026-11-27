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
import json
import os
import re
import secrets
import shlex
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import cost
import graphql_api
import stages

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
IMAGE = "runpod/pytorch:2.2.0-cu1281-torch291-ubuntu2404"
GPU_ID = "NVIDIA H100 80GB HBM3"
REPO_URL_DEFAULT = "https://github.com/uritell/caramel-italiano.git"
# Image pull plus a crash loop must not burn the whole cap. Measured failure
# on pod umpal4do5pop5k: uptime stayed <= 0 for 10 minutes.
STARTUP_LIMIT_S = 12 * 60
# One env value. RunPod rejects oversized create bodies (about 100 KB).
# The old base64 bootstrap was 7608 bytes; it is no longer sent.
ENV_VALUE_LIMIT = 4096


def github_slug(repo_url: str) -> str:
    url = repo_url.strip().rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]
    marker = "github.com/"
    if marker not in url:
        raise ValueError(f"bootstrap is fetched from GitHub, not from {repo_url}")
    return url.split(marker, 1)[1]


def bootstrap_raw_url(repo_url: str, ref: str) -> str:
    """Public raw URL. A ref with a slash uses refs/heads so the path is unambiguous."""
    slug = github_slug(repo_url)
    if not re.fullmatch(r"[A-Za-z0-9._/-]+", ref):
        raise ValueError(f"ref {ref!r} is not safe to put in a URL")
    if "/" in ref:
        return f"https://raw.githubusercontent.com/{slug}/refs/heads/{ref}/runpod/bootstrap.sh"
    return f"https://raw.githubusercontent.com/{slug}/{ref}/runpod/bootstrap.sh"


def container_script(raw_url: str) -> str:
    """One bash -c script. /start.sh stays PID 1; the job is a child.

    RunPod shlex-splits a bare dockerArgs string and execs it. A pipe or
    here-string never reaches a shell. This script is a single argv element.
    """
    if "'" in raw_url or any(ch in raw_url for ch in "\n\r"):
        raise ValueError("bootstrap URL must be a single shell word")
    return (
        "mkdir -p /workspace/pilot-out/logs\n"
        f"if curl -fsSL --retry 5 --retry-delay 2 -o /tmp/pilot-bootstrap.sh '{raw_url}'; then\n"
        "  nohup bash /tmp/pilot-bootstrap.sh >>/workspace/pilot-out/logs/job.log 2>&1 &\n"
        "else\n"
        "  echo bootstrap-download-failed >>/workspace/pilot-out/logs/job.log\n"
        "fi\n"
        "exec /start.sh\n"
    )


def docker_args_json(script: str) -> str:
    """Shape RunPod keeps as arrays. A bare string would be shlex-split."""
    return json.dumps(
        {"entrypoint": ["/bin/bash", "-c"], "cmd": [script]},
        separators=(",", ":"),
    )


def decode_docker_args(raw: str) -> tuple[list[str], list[str]]:
    """Match RunPod: JSON entrypoint/cmd is used as-is; anything else is shlex-split."""
    text = raw.strip()
    if text.startswith("{"):
        obj = json.loads(text)
        entry = list(obj.get("entrypoint") or [])
        cmd = list(obj.get("cmd") or [])
        return entry, cmd
    return [], shlex.split(text)


def build_env(hourly_cents: int, billing_epoch: int, token: str, repo_url: str, repo_ref: str) -> list[dict]:
    env = {
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
    # Do not copy the user's key into the pod. RunPod injects a pod-scoped
    # RUNPOD_API_KEY, and that is what watchdog.sh uses to terminate.
    if os.environ.get("PILOT_RESUME_FROM"):
        env["PILOT_RESUME"] = "1"
    return [{"key": k, "value": v} for k, v in env.items()]


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
        "dockerArgs": docker_args_json(container_script(bootstrap_raw_url(repo_url, repo_ref))),
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
    if body.get("gpuCount") != 1:
        problems.append("gpuCount must be 1")
    if any(item.get("key") == "RUNPOD_API_KEY" for item in body.get("env") or []):
        problems.append("the user's RUNPOD_API_KEY must not be copied into the pod")
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
    keys = {item["key"]: item["value"] for item in body.get("env") or []}
    for required in ("PILOT_MAX_SECONDS", "PILOT_MAX_USD_CENTS", "PILOT_HOURLY_CENTS", "PILOT_BILLING_EPOCH"):
        if required not in keys:
            problems.append(f"env missing {required}")
    if "PILOT_BOOTSTRAP_B64" in keys:
        problems.append("PILOT_BOOTSTRAP_B64 must not be sent; the script is fetched from the public repo")
    for key, value in keys.items():
        if len(value) > ENV_VALUE_LIMIT:
            problems.append(f"env {key} is {len(value)} bytes, over {ENV_VALUE_LIMIT}")
    try:
        entry, cmd = decode_docker_args(body.get("dockerArgs") or "")
    except json.JSONDecodeError as exc:
        problems.append(f"dockerArgs is not JSON: {exc}")
        entry, cmd = [], []
    if entry != ["/bin/bash", "-c"] or len(cmd) != 1:
        problems.append("dockerArgs must be bash -c with one script argument")
    elif "<<<" in cmd[0] or not cmd[0].rstrip().endswith("exec /start.sh"):
        problems.append("start script must exec /start.sh and must not use a here-string")
    elif "nohup bash /tmp/pilot-bootstrap.sh" not in cmd[0]:
        problems.append("start script must launch bootstrap.sh under nohup")
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


def proxy_base(pod_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", pod_id or ""):
        raise ValueError(f"pod id {pod_id!r} is not safe in a proxy hostname")
    return f"https://{pod_id}-8080.proxy.runpod.net"


def _http_get(url: str, token: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"X-Pilot-Token": token, "User-Agent": "caramel-italiano-pilot"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def pull_server_answered(pod_id: str, token: str, timeout: int = 15) -> bool:
    """True when our server answered. Cloudflare's 404 means nothing is listening."""
    try:
        _http_get(proxy_base(pod_id) + "/", token, timeout=timeout)
        return True
    except urllib.error.HTTPError as exc:
        # 401 is our server. 404 is the proxy with no listener.
        return exc.code == 401
    except Exception:
        return False


def _ack(base: str, token: str) -> None:
    for method in ("POST", "GET"):
        req = urllib.request.Request(
            base.rstrip("/") + "/ack",
            data=b"" if method == "POST" else None,
            method=method,
            headers={"X-Pilot-Token": token, "User-Agent": "caramel-italiano-pilot"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                if resp.status == 200:
                    print(f"acked pull via {method}", flush=True)
                    return
        except Exception as exc:  # noqa: BLE001
            print(f"ack {method} failed: {exc}", flush=True)


def _pull_http_base(base: str, dest: Path, token: str) -> bool:
    try:
        listing = json.loads(_http_get(base.rstrip("/") + "/", token))
    except Exception as exc:  # noqa: BLE001
        print(f"HTTP list {base} failed: {exc}", flush=True)
        return False
    files = listing.get("files") or []
    if not files:
        return False
    dest.mkdir(parents=True, exist_ok=True)
    for rel in files:
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        size_hdr = None
        try:
            probe = urllib.request.Request(
                base.rstrip("/") + "/" + rel,
                method="HEAD",
                headers={"X-Pilot-Token": token, "User-Agent": "caramel-italiano-pilot"},
            )
            with urllib.request.urlopen(probe, timeout=30) as resp:
                size_hdr = resp.headers.get("Content-Length")
        except Exception:
            size_hdr = None
        if target.is_file() and size_hdr and target.stat().st_size == int(size_hdr):
            continue
        data = _http_get(base.rstrip("/") + "/" + rel, token, timeout=300)
        target.write_bytes(data)
        print(f"pulled {rel} ({len(data)} bytes)", flush=True)
    _ack(base, token)
    return True


def _pull_http(pod: dict, dest: Path, token: str) -> bool:
    found = _public_port(pod, 8080)
    if not found:
        return False
    ip, port = found
    return _pull_http_base(f"http://{ip}:{port}", dest, token)


def upload_restore(pod_id: str, token: str, src: Path) -> None:
    """Send stores/ and checkpoints/ so the new pod can skip download and tokenize."""
    import http.client
    from urllib.parse import urlparse

    base = proxy_base(pod_id)
    parsed = urlparse(base)
    sent = 0
    for folder in ("stores", "checkpoints"):
        root = src / folder
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            rel = str(path.relative_to(src))
            size = path.stat().st_size
            conn = http.client.HTTPSConnection(parsed.hostname, timeout=300)
            with path.open("rb") as fh:
                conn.request("POST", "/upload", body=fh, headers={
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
            sent += 1
            print(f"uploaded {rel} ({size} bytes)", flush=True)
    conn = http.client.HTTPSConnection(parsed.hostname, timeout=60)
    conn.request("POST", "/ready", body=b"", headers={"X-Pilot-Token": token, "Content-Length": "0"})
    resp = conn.getresponse()
    resp.read()
    conn.close()
    if resp.status != 200:
        raise RuntimeError(f"ready returned {resp.status}")
    print(f"uploaded {sent} files and marked READY", flush=True)


def pull_outputs(pod: dict, dest: Path, token: str, key_path: str) -> str:
    """Proxy first. Direct ip:port is blocked from the launcher. SSH is last."""
    pod_id = str(pod.get("id") or "")
    if pod_id:
        try:
            if _pull_http_base(proxy_base(pod_id), dest, token):
                return "proxy"
        except ValueError as exc:
            print(f"proxy url skipped: {exc}", flush=True)
    if _pull_http(pod, dest, token):
        return "http"
    if _pull_ssh(pod, dest, key_path):
        if pod_id:
            try:
                _ack(proxy_base(pod_id), token)
            except ValueError:
                pass
        return "ssh"
    return ""


def _hourly_from_pod(pod: dict, fallback: int) -> int:
    raw = pod.get("costPerHr")
    if isinstance(raw, (int, float)) and raw > 0:
        return cost.dollars_to_cents_up(float(raw))
    per = ((pod.get("machine") or {}).get("currentPricePerGpu"))
    if isinstance(per, (int, float)) and per > 0:
        return cost.dollars_to_cents_up(float(per) * cost.GPU_COUNT)
    return fallback


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _status_name(pod: dict) -> str:
    return str(pod.get("desiredStatus") or pod.get("status") or "")


def _uptime_seconds(pod: dict) -> int | None:
    runtime = pod.get("runtime") or {}
    raw = runtime.get("uptimeInSeconds")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    return int(raw)


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
    print(f"proxy: {proxy_base(pod_id)}/")
    print(f"outputs will land in {out_dir}")

    return follow(pod_id, api_key, token, epoch, hourly, out_dir, ssh_key, pod)


def follow(pod_id, api_key, token, epoch, hourly, out_dir, ssh_key, pod=None) -> int:
    if pod is None:
        pod = {"id": pod_id}
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

    best_uptime = None
    proxy_up = False
    missing = 0
    uploaded = False
    resume = os.environ.get("PILOT_RESUME_FROM", "")
    attach = {
        "pod_id": pod_id,
        "token": token,
        "epoch": epoch,
        "hourly_cents": hourly,
        "out": str(out_dir),
    }
    (out_dir / "attach.json").write_text(json.dumps(attach, indent=2) + "\n")
    try:
        while True:
            time.sleep(20)
            try:
                got = graphql_api.pod_status(pod_id, api_key)
            except (graphql_api.ApiError, TimeoutError, OSError) as exc:
                print(f"{_stamp()} status retryable error: {exc}", flush=True)
                continue
            current = got.get("pod") or {}
            gone = not current
            if gone:
                try:
                    ids = graphql_api.list_pod_ids(api_key)
                except (graphql_api.ApiError, TimeoutError, OSError) as exc:
                    print(f"{_stamp()} pod list failed: {exc}", flush=True)
                    continue
                if pod_id in ids:
                    missing = 0
                    print(f"{_stamp()} status was empty but myself.pods still has {pod_id}", flush=True)
                    continue
                missing += 1
                print(f"{_stamp()} pod missing {missing}/3", flush=True)
                if missing >= 3:
                    print(f"{_stamp()} pod {pod_id} is gone from myself.pods; not calling terminate")
                    return 2
                continue
            missing = 0
            current.setdefault("id", pod_id)
            elapsed = int(time.time()) - epoch
            rate = _hourly_from_pod(current, hourly)
            spent = cost.cost_cents(elapsed, rate)
            reason = cost.stop_reason(elapsed, rate)
            status = _status_name(current)
            uptime = _uptime_seconds(current)
            if uptime is not None:
                best_uptime = uptime if best_uptime is None else max(best_uptime, uptime)
            if not proxy_up:
                proxy_up = pull_server_answered(pod_id, token)
            if proxy_up and resume and not uploaded:
                upload_restore(pod_id, token, Path(resume))
                uploaded = True
            print(
                f"{_stamp()} [cost] {elapsed}s  {cost.format_usd(spent)}  "
                f"rate {cost.format_usd(rate)}/h  status {status or 'unknown'}  "
                f"uptime {uptime if uptime is not None else 'none'}  "
                f"proxy {'up' if proxy_up else 'down'}  via {got.get('via')}",
                flush=True,
            )
            if reason:
                terminate(reason)
                return 0
            if elapsed >= STARTUP_LIMIT_S and ((best_uptime is None or best_uptime <= 0) or not proxy_up):
                why = []
                if best_uptime is None or best_uptime <= 0:
                    why.append(f"uptimeInSeconds still {best_uptime}")
                if not proxy_up:
                    why.append("pull server never answered on the proxy")
                terminate("startup failed: " + "; ".join(why))
                return 1
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


def rehearse_container_start() -> None:
    """Exec the argv RunPod builds. /workspace is a private mount, not this repo.

    Docker is not required. The pytorch image is not started. This checks the
    exec semantics that crashed pod umpal4do5pop5k: a pipe is not a shell.
    """
    import http.server
    import socketserver
    import tempfile
    import threading

    old = 'base64 -d <<<"$PILOT_BOOTSTRAP_B64" | bash'
    old_argv = shlex.split(old)
    if "|" not in old_argv:
        raise RuntimeError("expected the old dockerArgs pipe to be a bare argv")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        marker = root / "job"
        boot = root / "bootstrap.sh"
        boot.write_text(f"#!/bin/bash\necho job-ok > {shlex.quote(str(marker))}\n", encoding="utf-8")
        boot.chmod(0o755)

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                data = boot.read_bytes()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, fmt: str, *args) -> None:
                return

        httpd = socketserver.TCPServer(("127.0.0.1", 0), Handler)
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            script = container_script(f"http://127.0.0.1:{port}/bootstrap.sh")
            entry, cmd = decode_docker_args(docker_args_json(script))
            if entry != ["/bin/bash", "-c"] or cmd != [script]:
                raise RuntimeError("dockerArgs JSON did not decode to one bash -c script")
            wrapper = 'mkdir -p /tmp/ws && mount --bind /tmp/ws /workspace && exec "$@"'
            proc = subprocess.run(
                ["unshare", "--user", "--map-root-user", "--mount", "bash", "-c", wrapper, "runpod", *entry, *cmd],
                capture_output=True,
                text=True,
                timeout=30,
            )
            time.sleep(0.5)
        finally:
            httpd.shutdown()
            httpd.server_close()
        if not marker.exists() or "job-ok" not in marker.read_text(encoding="utf-8"):
            raise RuntimeError(f"bootstrap did not run under bash -c: {proc.stderr[-400:]}")
        if "/start.sh" not in (proc.stderr or ""):
            raise RuntimeError(f"exec /start.sh did not run: rc={proc.returncode} stderr={proc.stderr[-400:]}")


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--launch", action="store_true", help="create the pod and watch it")
    p.add_argument("--dry-run", action="store_true", help="validate only (this is also the default)")
    p.add_argument("--repo-url", default=os.environ.get("CARAMEL_REPO_URL", REPO_URL_DEFAULT))
    p.add_argument("--repo-ref", default=os.environ.get("CARAMEL_REPO_REF", "main"))
    p.add_argument("--out", type=Path, default=HERE / "retrieved")
    p.add_argument("--ssh-key", default=os.environ.get("PILOT_SSH_KEY", ""))
    p.add_argument("--resume-from", type=Path, default=None,
                   help="upload stores/ and checkpoints/ from this directory once the new pod answers")
    args = p.parse_args(argv)
    if args.resume_from:
        os.environ["PILOT_RESUME_FROM"] = str(args.resume_from)
    if args.launch and args.dry_run:
        print("Pass either --launch or --dry-run.")
        return 2
    if not args.launch:
        return dry_run(args.repo_url, args.repo_ref)
    return launch(args.repo_url, args.repo_ref, args.out, args.ssh_key)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
