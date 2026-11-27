#!/usr/bin/env python3
"""Offline checks for the pilot launcher. No network, no pod."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cost  # noqa: E402
import graphql_api  # noqa: E402
import launch  # noqa: E402
import stages  # noqa: E402
from download_open import LimitedReader  # noqa: E402


def _region(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    start = text.split("# BEGIN WATCHDOG\n", 1)
    if len(start) != 2:
        raise SystemExit(f"{path.name} has no watchdog region")
    body = start[1].split("# END WATCHDOG", 1)[0]
    return body.strip() + "\n"


def check_cost() -> None:
    if cost.GPU_COUNT != 1 or cost.MAX_USD_CENTS != 1200 or cost.CONTAINER_DISK_GB != 160:
        raise SystemExit("retry config is not 1 GPU, $12, 160 GB")
    if cost.QUOTED_HOURLY_CENTS != 439:
        raise SystemExit(f"quoted hourly cents {cost.QUOTED_HOURLY_CENTS}")
    stop_s, why = cost.first_stop(cost.QUOTED_HOURLY_CENTS)
    if (stop_s, why) != (cost.MAX_WALL_S, "wall"):
        raise SystemExit(f"quoted rate should hit the wall at {cost.MAX_WALL_S}s, got {stop_s} {why}")
    wall_bill = cost.cost_cents(cost.MAX_WALL_S, cost.QUOTED_HOURLY_CENTS)
    if wall_bill > cost.MAX_USD_CENTS or wall_bill > cost.USER_BUDGET_CENTS:
        raise SystemExit(f"2.5 h quote bills {wall_bill} cents")
    if wall_bill != 1098:
        raise SystemExit(f"expected $10.98 at the quote, got {wall_bill} cents")
    early, early_why = cost.first_stop(2000)  # $20/h pod
    if early_why != "dollars" or early >= cost.MAX_WALL_S:
        raise SystemExit(f"high price should hit dollars first, got {early} {early_why}")
    if cost.stop_reason(0, 439) is not None:
        raise SystemExit("t=0 must not stop")
    if cost.price_ok(491) is None:
        raise SystemExit("price above the ceiling was accepted")
    if cost.price_ok(439) is not None:
        raise SystemExit(f"quoted price was rejected: {cost.price_ok(439)}")
    print(
        f"cost ok: wall at {stop_s}s bills {cost.format_usd(wall_bill)}; "
        f"$20/h would stop at {early}s on dollars"
    )


def _check_pull_ack() -> None:
    import tempfile
    import urllib.request

    import socket

    root = Path(tempfile.mkdtemp())
    (root / "hello.txt").write_text("hi", encoding="utf-8")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = os.environ.copy()
    env.update(PILOT_OUT=str(root), PILOT_PULL_TOKEN="tok", PILOT_PULL_PORT=str(port))
    proc = subprocess.Popen(
        [sys.executable, str(HERE / "pull_server.py")],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        import urllib.error

        last = None
        for _ in range(50):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=0.2)
                break
            except urllib.error.HTTPError as exc:
                if exc.code in (401, 404):
                    break
                last = exc
                time.sleep(0.05)
            except Exception as exc:  # noqa: BLE001
                last = exc
                time.sleep(0.05)
        else:
            raise SystemExit(f"pull server did not start: {last}")
        early = urllib.request.Request(
            f"http://127.0.0.1:{port}/ack", data=b"", method="POST",
            headers={"X-Pilot-Token": "tok"},
        )
        with urllib.request.urlopen(early, timeout=2) as resp:
            early_body = json.loads(resp.read().decode())
        if early_body.get("ok") or (root / "PULLED").exists():
            raise SystemExit("ack before DONE wrote PULLED")
        (root / "DONE").write_text("done\n", encoding="utf-8")
        time.sleep(0.05)
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/ack", data=b"", method="POST",
            headers={"X-Pilot-Token": "tok"},
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            if resp.status != 200:
                raise SystemExit(f"ack status {resp.status}")
        if not (root / "PULLED").is_file() or (root / "PULLED").stat().st_mtime <= (root / "DONE").stat().st_mtime:
            raise SystemExit("ack after DONE did not write a newer PULLED")
    finally:
        proc.terminate()
        proc.wait(timeout=5)
    print("pull ack ok")


def check_graphql_shape() -> None:
    blob = (HERE / "graphql_api.py").read_text(encoding="utf-8")
    for needle in (
        "podFindAndDeployOnDemand",
        "podTerminate",
        "podStop",
        "https://api.runpod.io/graphql",
        "https://api.runpod.io/v2",
    ):
        if needle not in blob:
            raise SystemExit(f"graphql client missing {needle}")
    if "rest.runpod.io" in blob:
        raise SystemExit("graphql client names the unreachable REST host")
    if "podTerminate(input: $input) {" in graphql_api.TERMINATE.replace(" ", ""):
        raise SystemExit("podTerminate must not have a selection set")
    body = launch.build_create_input(
        878, 1_700_000_000, "tok", "https://github.com/uritell/caramel-italiano.git", "main",
    )
    problems = launch.validate_input(body)
    if problems:
        raise SystemExit("create input rejected: " + "; ".join(problems))
    if body["volumeInGb"] != 0 or "networkVolumeId" in body:
        raise SystemExit("create input still asks for storage that outlives the pod")
    if body.get("gpuCount") != 1 or body.get("containerDiskInGb") != 160:
        raise SystemExit(f"create input is not 1 GPU / 160 GB: {body.get('gpuCount')} {body.get('containerDiskInGb')}")
    env_keys = {item["key"] for item in body.get("env") or []}
    if "RUNPOD_API_KEY" in env_keys:
        raise SystemExit("create input copies RUNPOD_API_KEY into the pod")
    redacted = json.dumps(launch.redact(body))
    if "tok" in redacted and "PILOT_PULL_TOKEN" in redacted:
        # The dry-run token is replaced. A real token must not survive redact.
        pass
    if launch.redact(body)["env"][0]["value"] == body["env"][0]["value"] and body["env"][0]["key"] in {
        "RUNPOD_API_KEY", "PILOT_BOOTSTRAP_B64", "PILOT_PULL_TOKEN", "GITHUB_TOKEN"
    }:
        raise SystemExit("redact left a secret in place")
    hidden = {item["key"]: item["value"] for item in launch.redact(body)["env"]}
    if hidden.get("PILOT_PULL_TOKEN") != "***":
        raise SystemExit("pull token was not redacted")
    queries = "\n".join((
        graphql_api.GPU_QUOTE, graphql_api.POD_CORE, graphql_api.POD_STATUS,
        graphql_api.CREATE, graphql_api.STATUS, graphql_api.STOP, graphql_api.TERMINATE,
    ))
    for banned in graphql_api._HOST_ONLY_FIELDS:
        if banned in queries:
            raise SystemExit(f"query still selects host-only field {banned}")
    if "availableGpuCount" in queries:
        raise SystemExit("availableGpuCount is still selected")
    if "securePrice" not in graphql_api.GPU_QUOTE:
        raise SystemExit("GPU quote no longer reads securePrice")
    kept = graphql_api.quote_from_payload({
        "data": {"gpuTypes": [{"id": "NVIDIA H100 80GB HBM3", "securePrice": 4.39}]},
        "errors": [{
            "message": "Not authorized. Missing required scope(s): CSR_READ, HOST_DAEMON",
            "path": ["gpuTypes", 0, "lowestPrice", "availableGpuCount"],
        }],
    })
    if kept[0].get("securePrice") != 4.39 or not kept[1]:
        raise SystemExit("optional field error dropped a quote that had securePrice")
    try:
        graphql_api.quote_from_payload({
            "data": {"gpuTypes": [{"id": "NVIDIA H100 80GB HBM3"}]},
            "errors": [{
                "message": "Not authorized",
                "path": ["gpuTypes", 0, "securePrice"],
            }],
        })
    except graphql_api.ApiError:
        pass
    else:
        raise SystemExit("an error on securePrice was treated as optional")
    try:
        graphql_api.quote_from_payload({
            "errors": [{"message": "Not authorized", "path": ["gpuTypes", 0, "lowestPrice", "availableGpuCount"]}],
        })
    except graphql_api.ApiError:
        pass
    else:
        raise SystemExit("a field error with no securePrice was ignored")
    launch.rehearse_container_start()
    _check_pull_ack()
    print("graphql shape ok")


def check_stages() -> None:
    cmds = stages.full_commands("python", Path("mono"), Path("stereo"), Path("dev"), Path("runs"))
    stages.assert_full_numbers(cmds)
    joined = " ".join(" ".join(argv) for _, argv, _ in cmds)
    from pilot_config import plan
    expected = plan()
    for needle in (str(expected["s1_stop"]), str(expected["s2_end"]), str(expected["s3_end"])):
        if needle not in joined:
            raise SystemExit(f"stage commands lost step {needle}")
    if "torch.distributed" in joined or "--nproc_per_node" in joined:
        raise SystemExit("stages still launch a second GPU")
    print("stages ok")


def check_watchdog_copies() -> None:
    a = _region(HERE / "watchdog.sh")
    b = _region(HERE / "bootstrap.sh")
    if a != b:
        raise SystemExit("bootstrap.sh watchdog copy drifted from watchdog.sh")
    for needle in ("podTerminate", "podStop", "api.runpod.io/graphql", "(elapsed * hourly + 3599) / 3600"):
        if needle not in a:
            raise SystemExit(f"watchdog missing {needle}")
    if "rest.runpod.io" in a:
        raise SystemExit("watchdog names the unreachable REST host")
    print("watchdog copies match")


def check_reader() -> None:
    class Raw:
        def __init__(self):
            self.left = b"abcdefghij"

        def read(self, n):
            chunk = self.left[:n]
            self.left = self.left[n:]
            return chunk

    reader = LimitedReader(Raw(), 4)
    got = reader.read(3) + reader.read(10)
    if got != b"abcd" or not reader.hit_cap:
        raise SystemExit(f"byte cap returned {got!r}")
    print("byte cap ok")


def check_shell() -> None:
    for name in ("pod_run.sh", "bootstrap.sh", "watchdog.sh", "supervise.sh"):
        proc = subprocess.run(["bash", "-n", str(HERE / name)], capture_output=True, text=True)
        if proc.returncode != 0:
            raise SystemExit(proc.stderr)
    bash = subprocess.run(["bash", str(HERE / "watchdog.sh"), "--self-test"], capture_output=True, text=True)
    if bash.returncode != 0:
        raise SystemExit(bash.stderr or bash.stdout)
    dash = subprocess.run(["dash", str(HERE / "watchdog.sh"), "--self-test"], capture_output=True, text=True)
    if dash.returncode == 0:
        raise SystemExit("dash ran the watchdog self-test; the cap checks would be skipped")
    print("bash -n ok; watchdog self-test passes under bash and fails under dash")


def main() -> int:
    check_cost()
    check_graphql_shape()
    check_stages()
    check_watchdog_copies()
    check_reader()
    check_shell()
    print("self-check ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
