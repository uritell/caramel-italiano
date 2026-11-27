#!/usr/bin/env python3
"""Offline checks for the pilot launcher. No network, no pod."""
from __future__ import annotations

import json
import subprocess
import sys
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
    if cost.QUOTED_HOURLY_CENTS != 878:
        raise SystemExit(f"quoted hourly cents {cost.QUOTED_HOURLY_CENTS}")
    stop_s, why = cost.first_stop(cost.QUOTED_HOURLY_CENTS)
    if (stop_s, why) != (cost.MAX_WALL_S, "wall"):
        raise SystemExit(f"quoted rate should hit the wall at {cost.MAX_WALL_S}s, got {stop_s} {why}")
    wall_bill = cost.cost_cents(cost.MAX_WALL_S, cost.QUOTED_HOURLY_CENTS)
    if wall_bill > cost.MAX_USD_CENTS or wall_bill > cost.USER_BUDGET_CENTS:
        raise SystemExit(f"2.5 h quote bills {wall_bill} cents")
    if wall_bill != 2195:
        raise SystemExit(f"expected $21.95 at the quote, got {wall_bill} cents")
    early, early_why = cost.first_stop(2000)  # $20/h pod
    if early_why != "dollars" or early >= cost.MAX_WALL_S:
        raise SystemExit(f"high price should hit dollars first, got {early} {early_why}")
    if cost.stop_reason(0, 878) is not None:
        raise SystemExit("t=0 must not stop")
    if cost.price_ok(491) is None:
        raise SystemExit("price above the ceiling was accepted")
    if cost.price_ok(439) is not None:
        raise SystemExit(f"quoted price was rejected: {cost.price_ok(439)}")
    print(
        f"cost ok: wall at {stop_s}s bills {cost.format_usd(wall_bill)}; "
        f"$20/h would stop at {early}s on dollars"
    )


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
    body = launch.build_create_input(878, 1_700_000_000, "tok", "https://example.invalid/repo.git", "main")
    problems = launch.validate_input(body)
    if problems:
        raise SystemExit("create input rejected: " + "; ".join(problems))
    if body["volumeInGb"] != 0 or "networkVolumeId" in body:
        raise SystemExit("create input still asks for storage that outlives the pod")
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
    print("graphql shape ok")


def check_stages() -> None:
    cmds = stages.full_commands("python", Path("mono"), Path("stereo"), Path("dev"), Path("runs"))
    stages.assert_full_numbers(cmds)
    joined = " ".join(" ".join(argv) for _, argv, _ in cmds)
    if "401" not in joined or "1072" not in joined or "1981" not in joined:
        raise SystemExit("stage commands lost the pilot step counts")
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
    for name in ("pod_run.sh", "bootstrap.sh", "watchdog.sh"):
        proc = subprocess.run(["bash", "-n", str(HERE / name)], capture_output=True, text=True)
        if proc.returncode != 0:
            raise SystemExit(proc.stderr)
    print("bash -n ok")


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
