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


def check_warmstart() -> None:
    import os
    import warmstart
    est = warmstart.estimate(439)
    if est["over_abort"] or est["gpu_cents"] > 250:
        raise SystemExit(f"warmstart estimate is above $2.50: {est['gpu_cents']} cents")
    if warmstart.MAX_USD_CENTS != 300 or warmstart.MAX_WALL_S != 2280:
        raise SystemExit("warmstart cap is not $3 or 38 min")
    os.environ["PILOT_CONFIG"] = "warmstart"
    try:
        cmds = stages.full_commands("python", Path("m"), Path("s"), Path("d"), Path("runs"))
        names = [name for name, _, _ in cmds]
        if names != ["s2", "s3"]:
            raise SystemExit(f"warmstart stages: {names}")
        stages.assert_warmstart_numbers(cmds)
        if "--resume" in " ".join(cmds[0][1]):
            raise SystemExit("stage 2 must load safetensors, not resume")
        try:
            stages.assert_full_numbers(cmds)
        except SystemExit:
            pass
        else:
            raise SystemExit("the pilot assert accepted warmstart commands")
    finally:
        os.environ.pop("PILOT_CONFIG", None)
    print(f"warmstart ok: {est['seconds'] / 60:.1f} min, ${est['gpu_cents'] / 100:.2f} GPU")


def check_option_a() -> None:
    import os
    import option_a
    est = option_a.estimate(439)
    if est["over_abort"] or est["gpu_cents"] > 800:
        raise SystemExit(f"option A estimate is above $8: {est['gpu_cents']} cents")
    if option_a.MAX_USD_CENTS != 900 or option_a.MAX_WALL_S != 6840:
        raise SystemExit("option A cap is not $9 or 1.9 h")
    if option_a.DISK_GB < 200:
        raise SystemExit("option A disk is smaller than the scaled 182 h measurement")
    os.environ["PILOT_CONFIG"] = "optionA"
    try:
        spec = option_a.training_plan()
        if spec["s3_end"] <= 1254:
            raise SystemExit("option A did not add steps for the larger pools")
        cmds = stages.full_commands("python", Path("m"), Path("s"), Path("d"), Path("runs"))
        blob = " ".join(" ".join(argv) for _, argv, _ in cmds)
        if str(spec["s1_stop"]) not in blob or "--save_best" not in blob:
            raise SystemExit("option A commands lost the step count or save_best")
    finally:
        os.environ.pop("PILOT_CONFIG", None)
    # The module-level pilot numbers must still describe the retry.
    if stages.S1_STOP != 156:
        raise SystemExit("import-time pilot steps changed")
    print(
        f"option A ok: {est['seconds'] / 60:.1f} min, "
        f"${est['gpu_cents'] / 100:.2f} GPU, disk ${est['disk_cents'] / 100:.2f}"
    )


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


def check_launch_dispatch() -> None:
    """--launch must call launch(); the dry-run return must not swallow it."""
    seen = {}
    orig_launch, orig_dry = launch.launch, launch.dry_run

    def fake_launch(*args, **kwargs):
        seen["launch"] = args
        return 0

    def fake_dry(*args, **kwargs):
        seen["dry"] = args
        return 0

    launch.launch = fake_launch
    launch.dry_run = fake_dry
    try:
        rc = launch.main([
            "--launch", "--config", "warmstart",
            "--stores", "runpod/retrieved/mviebvzw3ht7hn/stores",
        ])
        if rc != 0 or "launch" not in seen or "dry" in seen:
            raise SystemExit(f"--launch did not call launch(): rc={rc} keys={list(seen)}")
        if seen["launch"][4] != "warmstart":
            raise SystemExit(f"launch was not warmstart: {seen['launch'][4]}")
        if not str(seen["launch"][5]).endswith("mviebvzw3ht7hn/stores"):
            raise SystemExit(f"stores were not passed to launch(): {seen['launch'][5]}")
        rc = launch.main(["--dry-run", "--config", "pilot"])
        if rc != 0 or "dry" not in seen:
            raise SystemExit("dry-run did not call dry_run()")
    finally:
        launch.launch = orig_launch
        launch.dry_run = orig_dry
        os.environ.pop("PILOT_CONFIG", None)
    print("launch dispatch ok")


def check_proxy_retry_and_pull() -> None:
    """A proxy 502 is retried. A later stop still pulls before terminate."""
    import io
    import tempfile
    import urllib.error

    hits = {"n": 0}
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class OneShot(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            hits["n"] += 1
            if hits["n"] < 3:
                body = b"bad"
                self.send_response(502)
            else:
                body = b"pulled"
                self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            return

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), OneShot)
    port = httpd.server_address[1]
    thread = __import__("threading").Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        body = launch._http_get(f"http://127.0.0.1:{port}/log", "tok", timeout=5, attempts=5, pause=lambda _s: None)
    finally:
        httpd.shutdown()
    if body != b"pulled" or hits["n"] != 3:
        raise SystemExit(f"502 was not retried: hits={hits['n']} body={body!r}")

    calls = {"status": 0, "pull": 0, "term": 0}

    def status(pod_id, key):
        calls["status"] += 1
        if calls["status"] == 1:
            raise urllib.error.HTTPError("https://proxy", 502, "bad gateway", None, io.BytesIO(b"bad"))
        return {"pod": {"id": pod_id, "desiredStatus": "RUNNING", "costPerHr": 4.39,
                        "runtime": {"uptimeInSeconds": 5}}, "via": "test"}

    def pull(*_a, **_k):
        calls["pull"] += 1
        return "proxy"

    def term(*_a, **_k):
        calls["term"] += 1
        return {"via": "test"}

    saved = (graphql_api.pod_status, graphql_api.terminate_pod, launch.pull_outputs,
             launch.pull_server_answered, time.sleep)
    graphql_api.pod_status = status
    graphql_api.terminate_pod = term
    launch.pull_outputs = pull
    launch.pull_server_answered = lambda *_a, **_k: True
    time.sleep = lambda _s: None
    try:
        with tempfile.TemporaryDirectory() as tmp:
            rc = launch.follow("pod123", "k", "tok", int(time.time()), 439, Path(tmp), "",
                               wall_s=0, cap_cents=300)
    finally:
        (graphql_api.pod_status, graphql_api.terminate_pod, launch.pull_outputs,
         launch.pull_server_answered, time.sleep) = saved
    if rc != 0 or calls["status"] < 2 or calls["pull"] < 1 or calls["term"] < 1:
        raise SystemExit(f"follow did not retry then pull before terminate: rc={rc} {calls}")
    print("proxy retry ok")


def main() -> int:
    check_cost()
    check_option_a()
    check_warmstart()
    check_graphql_shape()
    check_stages()
    check_watchdog_copies()
    check_reader()
    check_shell()
    check_launch_dispatch()
    check_proxy_retry_and_pull()
    print("self-check ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
