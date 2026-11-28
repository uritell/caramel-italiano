#!/usr/bin/env python3
"""Fixture checks for the lean data plan. No download and no pod."""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import lean_data  # noqa: E402
import lean_launch  # noqa: E402
import lean_prep  # noqa: E402
import stages  # noqa: E402
import warmstart  # noqa: E402


def _tar_names(members: list[tuple[str, bytes]]) -> list[str]:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, payload in members:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    buf.seek(0)
    with tarfile.open(fileobj=buf, mode="r") as tar:
        return tar.getnames()


def check_year_and_segments() -> None:
    good = _tar_names([
        ("2015/20150312-0900-PLENARY-16_it.ogg", b"ogg"),
        ("2015/20150601-1700-PLENARY_it.ogg", b"ogg"),
    ])
    audio = lean_data.verify_2015_members(good)
    if len(audio) != 2:
        raise SystemExit(f"2015 tar was not accepted: {audio}")
    bad = _tar_names([
        ("2015/20150312-0900-PLENARY-16_it.ogg", b"ogg"),
        ("2016/20160110-0900-PLENARY-16_it.ogg", b"ogg"),
    ])
    try:
        lean_data.verify_2015_members(bad)
    except SystemExit as exc:
        if "2016" not in str(exc):
            raise
    else:
        raise SystemExit("a 2016 event was accepted as 2015")
    cuts = lean_data.segment_duration(30.0)
    if cuts != [(0.0, 12.0), (12.0, 12.0), (24.0, 6.0)]:
        raise SystemExit(f"segments are not the 12 s mono cuts: {cuts}")
    if lean_data.segment_duration(0.1):
        raise SystemExit("a clip under 0.2 s was kept")
    text = (HERE / "prepare_audio.py").read_text(encoding="utf-8")
    if "TURN_CAP_S = 12.0" not in text or lean_data.SEGMENT_S != 12.0:
        raise SystemExit("lean segments drifted from the mono 12 s turn cap")
    print("year and segments ok")


def check_hours() -> None:
    vox = lean_data.take_hours([3600.0] * 1200, lean_data.VOX_CAP_H)
    if lean_data.VOX_CAP_H != 700 or len(vox) != 700:
        raise SystemExit(f"vox cap kept {len(vox)} hours at {lean_data.VOX_CAP_H}, not 700")
    mls = [3600.0] * 300
    kept = lean_data.mls_remainder(mls)
    if len(kept) != int(lean_data.MLS_KEEP_H):
        raise SystemExit(f"mls remainder kept {len(kept)} h, expected {lean_data.MLS_KEEP_H:.0f}")
    text = lean_data.attribution_text()
    if "CC0" not in text or "CC BY 4.0" not in text or "it_2015.tar" not in text:
        raise SystemExit("attribution is missing a licence line")
    print("hours and attribution ok")


def check_speed_gate() -> None:
    slow = lean_data.speed_report(300.0, 300.0, lean_data.A40_HOURLY_CENTS)
    # 1x realtime on 1158 h is hundreds of dollars.
    if not slow["over_cap"]:
        raise SystemExit(f"a 1x encode was not refused: {slow}")
    fast = lean_data.speed_report(300.0, 1.0, lean_data.A40_HOURLY_CENTS)
    if fast["over_cap"] or fast["rtf"] < 200:
        raise SystemExit(f"a fast encode was refused: {fast}")
    if lean_data.MAX_USD_CENTS != 300 or lean_data.MAX_WALL_S != 6 * 3600:
        raise SystemExit("lean data cap is not $3.00 or 6 h")
    # About $4.85 at the A40 rate: over the $3 prep cap, under the old $5.50.
    mid = lean_data.speed_report(300.0, 2.0, lean_data.A40_HOURLY_CENTS)
    if not mid["over_cap"] or mid["cents"] > 550:
        raise SystemExit(f"a projection between $3 and $5.50 was not refused: {mid}")
    ref = lean_data.reference_projection()
    if lean_data.over_cap(ref["cents"]):
        raise SystemExit(f"reference projection exceeds the cap: {ref}")
    try:
        lean_prep.gate(slow, lean_data.MAX_USD_CENTS)
    except SystemExit:
        pass
    else:
        raise SystemExit("gate() allowed an over-cap projection")
    print("speed gate ok")


def check_warmstart_mix() -> None:
    plain = warmstart.lean_mix(False)
    if abs(sum(w for _, w in plain) - 1) > 1e-9:
        raise SystemExit(f"weights do not sum to 1: {plain}")
    if plain[0] != ("stereo", 0.55):
        raise SystemExit(f"stereo is not the heaviest store: {plain}")
    names = [name for name, _ in plain]
    if names != ["stereo", "mono_voxpopuli", "mono_mls2"]:
        raise SystemExit(f"lean stores: {names}")
    with_synth = warmstart.lean_mix(True)
    if "synth_stereo" not in [name for name, _ in with_synth]:
        raise SystemExit("optional synth_stereo was dropped")
    if with_synth[0][0] != "stereo" or with_synth[0][1] < with_synth[1][1]:
        raise SystemExit(f"synth mix does not keep stereo heaviest: {with_synth}")
    os.environ["PILOT_CONFIG"] = "warmstart"
    os.environ["PILOT_LEAN_DATA"] = "1"
    try:
        cmds = stages.full_commands("python", Path("stores/mono"), Path("stores/stereo"), Path("stores/dev"), Path("runs"))
        stages.assert_warmstart_numbers(cmds)
        blob = " ".join(" ".join(argv) for _, argv, _ in cmds)
        for needle in ("mono_voxpopuli", "mono_mls2", "0.55", "0.30", "0.15", "--init-weights", "1e-4"):
            if needle not in blob:
                raise SystemExit(f"lean warmstart command missing {needle}")
    finally:
        os.environ.pop("PILOT_LEAN_DATA", None)
        os.environ.pop("PILOT_CONFIG", None)
    print("warmstart mix ok")


def check_fixture_and_dry_run() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        clips = root / "clips.jsonl"
        clips.write_text(json.dumps({"path": "a.ogg", "duration_s": 300, "start_s": 0}) + "\n", encoding="utf-8")
        out = root / "out"
        rc = lean_prep.main(["--speed-test", "--fixture", str(clips), "--out", str(out), "--hourly-cents", "59", "--cap-cents", "300"])
        if rc != 0:
            raise SystemExit(f"fixture speed test failed: {rc}")
        report = json.loads((out / "speed_test.json").read_text(encoding="utf-8"))
        if report["speed_audio_s"] != 300 or report["over_cap"]:
            raise SystemExit(f"fixture report: {report}")
        store = root / "stores" / "mono_voxpopuli"
        store.mkdir(parents=True)
        (store / "tokens.bin").write_bytes(b"\x00\x01")
        (store / "index.npy").write_bytes(b"idx")
        (store / "meta.json").write_text("{}\n", encoding="utf-8")
        (store / "wave.wav").write_bytes(b"nope")
        published = lean_prep.publish_tokens([store], root / "pull")
        if any(name.endswith(".wav") for name in published):
            raise SystemExit(f"pull included audio: {published}")
        if not (root / "pull" / "stores" / "mono_voxpopuli" / "tokens.bin").is_file():
            raise SystemExit("tokens were not published")
        if "CC0" not in (root / "pull" / "ATTRIBUTION.txt").read_text(encoding="utf-8"):
            raise SystemExit("pull is missing the attribution")
    proc = subprocess.run(
        [sys.executable, str(HERE / "lean_launch.py"), "--dry-run"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(proc.stdout + proc.stderr)
    text = proc.stdout
    for needle in ("it_2015.tar", "it_2015_2.tar", "No pod was created", "158", "700", "$3.00", "6 h", "pull_server.py", "10 min of RUNNING"):
        if needle not in text:
            raise SystemExit(f"dry run missing {needle}:\n{text}")
    boot = (HERE / "lean_bootstrap.sh").read_text(encoding="utf-8")
    pull_at = boot.find("pull_server.py")
    prep_at = boot.find("lean_prep.py")
    if pull_at < 0 or prep_at < 0 or pull_at > prep_at:
        raise SystemExit("lean_bootstrap.sh must start pull_server.py before lean_prep.py")
    if not lean_data.proxy_missed(10 * 60, False):
        raise SystemExit("10 min without 8080 must fail")
    if lean_data.proxy_missed(10 * 60 - 1, False) or lean_data.proxy_missed(10 * 60, True):
        raise SystemExit("proxy deadline fired too early or while the server was up")
    if not lean_data.speed_report_missed(15 * 60, False):
        raise SystemExit("15 min without speed_test.json must fail")
    if lean_data.speed_report_missed(15 * 60 - 1, False) or lean_data.speed_report_missed(15 * 60, True):
        raise SystemExit("speed-test deadline fired too early or after the report existed")
    for earlier, later in (
        ("pull_server.py", "install_pod"),
        ("install_pod", "import transformers, torch"),
        ("_load_mimi", "lean_prep.py"),
    ):
        if boot.find(earlier) < 0 or boot.find(later) < 0 or boot.find(earlier) > boot.find(later):
            raise SystemExit(f"lean_bootstrap.sh must run {earlier} before {later}")
    if "creating the pod" in text.lower():
        raise SystemExit("dry run tried to create a pod")
    print("fixture and dry run ok")


def _install_pod_packages(text: str) -> list[str]:
    import re

    match = re.search(r"^install_pod\(\) \{\n(.*)\n\}", text, re.M | re.S)
    if not match:
        raise SystemExit("install_pod() not found")
    pkgs: list[str] = []
    buf = ""
    for line in match.group(1).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.endswith("\\"):
            buf += stripped[:-1] + " "
            continue
        buf += " " + stripped
        if "pip install" not in buf:
            buf = ""
            continue
        tokens = buf.split()
        i = tokens.index("install") + 1
        while i < len(tokens):
            tok = tokens[i]
            if tok in ("||", "true"):
                break
            if tok.startswith("-"):
                i += 2 if tok in ("--python", "--index-url") else 1
                continue
            if tok.startswith("$") or tok.startswith("http"):
                i += 1
                continue
            pkgs.append(tok.strip("'\""))
            i += 1
        buf = ""
    return pkgs


def _packages_via_install_pod(script: str, env: str) -> list[str]:
    if "pod_env.sh" not in script or "install_pod" not in script:
        raise SystemExit("script does not reuse install_pod from pod_env.sh")
    own = [
        line.strip() for line in script.splitlines()
        if "pip install" in line and not line.strip().startswith("#")
    ]
    if own:
        raise SystemExit("script has its own pip install:\n" + "\n".join(own))
    return _install_pod_packages(env)


def check_same_packages() -> None:
    env = (HERE / "pod_env.sh").read_text(encoding="utf-8")
    pod_script = (HERE / "pod_run.sh").read_text(encoding="utf-8")
    if "pod_env.sh" not in pod_script or "install_pod" not in pod_script:
        raise SystemExit("pod_run.sh does not call the shared install_pod")
    pod = _install_pod_packages(env)
    lean = _packages_via_install_pod((HERE / "lean_bootstrap.sh").read_text(encoding="utf-8"), env)
    if pod != lean:
        raise SystemExit(f"package lists differ\n  pod: {pod}\n  lean: {lean}")
    expected = [
        "torch==2.9.1", "triton", "numpy", "transformers==4.46.3",
        "safetensors", "soundfile", "huggingface_hub", "soxr",
    ]
    if pod != expected:
        raise SystemExit(f"install_pod packages: {pod}")
    print("install_pod packages match")


def check_crash_writes_failed() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out"
        original = lean_prep._run_pod

        def boom(_args):
            raise ModuleNotFoundError("No module named 'transformers'")

        lean_prep._run_pod = boom
        try:
            rc = lean_prep.main(["--pod", "--out", str(out), "--work", str(Path(tmp) / "work")])
        finally:
            lean_prep._run_pod = original
        if rc == 0:
            raise SystemExit("a crash returned success")
        text = (out / "FAILED").read_text(encoding="utf-8")
        if "Traceback" not in text or "ModuleNotFoundError" not in text or "transformers" not in text:
            raise SystemExit(f"FAILED is missing the traceback:\n{text}")
    print("crash writes FAILED")


def _writer():
    class Writer:
        def add(self, frames, audio_s, encode_s):
            return None

        def flush(self):
            return None

    return Writer()


def check_decode_once() -> None:
    calls = []

    def read_mono(path):
        calls.append(str(path))
        return [0] * (24000 * 48)

    clips = []
    for path, count in (("a.ogg", 3), ("b.ogg", 1)):
        for i in range(count):
            clips.append({"path": path, "name": path, "start_s": i * 12, "duration_s": 12})
    audio, _enc, _whole, rest = lean_prep._encode_until(
        None, "cpu", _writer(), clips, 0, read_mono=read_mono, encode_fn=lambda _piece: [1],
    )
    if calls != ["a.ogg", "b.ogg"]:
        raise SystemExit(f"read_audio ran {calls}, expected once per file")
    if audio != 48 or rest:
        raise SystemExit(f"encoded {audio}s rest={rest}")
    print("decode once ok")


def check_speed_counts_load() -> None:
    import time as time_mod

    def read_mono(_path):
        time_mod.sleep(0.15)
        return [0] * (24000 * 12)

    clips = [{"path": "a.ogg", "name": "a.ogg", "start_s": 0, "duration_s": 12}]
    audio, enc, whole, _rest = lean_prep._encode_until(
        None, "cpu", _writer(), clips, 0, read_mono=read_mono, encode_fn=lambda _piece: [1],
    )
    report = lean_data.speed_report(audio, enc if enc > 0 else 1e-6, 59, end_to_end_s=whole)
    if whole < 0.15 or report["end_to_end_rtf"] >= report["rtf"]:
        raise SystemExit(f"load time did not lower the rate: enc={enc} whole={whole} {report}")
    print("speed test counts load ok")


def check_walk_retry() -> None:
    blob = io.BytesIO()
    with tarfile.open(fileobj=blob, mode="w") as tar:
        info = tarfile.TarInfo("2015/a.ogg")
        info.size = 100
        tar.addfile(info, io.BytesIO(b"A" * 100))
        info = tarfile.TarInfo("2015/b.ogg")
        info.size = 5000
        tar.addfile(info, io.BytesIO(b"B" * 5000))
    raw = blob.getvalue()
    opens: list[int] = []

    class Flaky:
        def __init__(self, data: bytes, fail_after_reads: int | None):
            self.data = data
            self.pos = 0
            self.reads = 0
            self.fail_after_reads = fail_after_reads

        def read(self, n: int = -1) -> bytes:
            self.reads += 1
            if self.fail_after_reads is not None and self.reads > self.fail_after_reads:
                raise ConnectionResetError("reset")
            if n is None or n < 0:
                n = len(self.data) - self.pos
            chunk = self.data[self.pos:self.pos + n]
            self.pos += len(chunk)
            return chunk

        def close(self) -> None:
            return None

    def opener(_url, start):
        start = int(start or 0)
        opens.append(start)
        fail = 1 if len(opens) == 2 else None
        return Flaky(raw[start:], fail)

    with tempfile.TemporaryDirectory() as tmp:
        kept, _got = lean_prep.walk_tar(
            "https://example.invalid/it_2015.tar", Path(tmp), mode="r|", check_year=True,
            skip_names=set(), max_new_s=0, opener=opener, pause=lambda _s: None,
            duration_of=lambda _path: 12.0,
        )
    if len(opens) < 3 or not (opens[1] > 0 and opens[2] == opens[1]):
        raise SystemExit(f"did not reopen at the member offset: {opens}")
    if sorted(row["name"] for row in kept) != ["a.ogg", "b.ogg"]:
        raise SystemExit(f"members: {kept}")
    print("walk_tar retry ok")


def check_budget_stop() -> None:
    if not lean_prep.over_encode_budget(600, 600) or lean_prep.over_encode_budget(599, 600):
        raise SystemExit("budget comparison is wrong")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        lean_prep.finish_at_budget(out, 2 * 3600)
        text = (out / "DONE").read_text(encoding="utf-8")
        if "stopped at budget" not in text or "2.00 h" not in text:
            raise SystemExit(text)
    print("budget stop ok")


def check_heartbeat_file() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        hb = lean_prep.Heartbeat(out)
        hb.start()
        try:
            hb.update(phase="bulk-vox", current_member="a.ogg", audio_h_done=1.5, frames=10, end_to_end_rtf=40.0)
            doc = json.loads((out / "status.json").read_text(encoding="utf-8"))
        finally:
            hb.stop()
            if hb._thread is not None:
                hb._thread.join(2)
    for key in (
        "ts", "phase", "current_member", "audio_h_done", "frames", "end_to_end_rtf",
        "main_loop_ts", "disk_free_gb", "mem_available_mb", "gpu_mem_used_mb",
    ):
        if key not in doc:
            raise SystemExit(f"status.json missing {key}: {doc}")
    if doc["phase"] != "bulk-vox" or doc["current_member"] != "a.ogg":
        raise SystemExit(doc)
    print("heartbeat ok")


def check_stale_heartbeat() -> None:
    import contextlib
    from datetime import datetime, timedelta, timezone

    fresh = {"ts": "2026-11-28T10:00:00Z"}
    now = datetime(2026, 11, 28, 10, 10, tzinfo=timezone.utc).timestamp()
    if not lean_launch.heartbeat_stale(fresh, now):
        raise SystemExit("10 min old heartbeat was not stale")
    if lean_launch.heartbeat_stale(fresh, now - 120):
        raise SystemExit("a fresh heartbeat was treated as stale")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        old = (datetime.now(timezone.utc) - timedelta(minutes=11)).strftime("%Y-%m-%dT%H:%M:%SZ")
        (out / "status.json").write_text(json.dumps({
            "ts": old, "phase": "bulk-vox", "current_member": "a.ogg",
            "audio_h_done": 1.25, "frames": 8, "end_to_end_rtf": 20, "main_loop_ts": old,
        }), encoding="utf-8")
        terminated = []
        saved = (
            lean_launch.graphql_api.pod_status,
            lean_launch.graphql_api.terminate_pod,
            lean_launch.pull_outputs,
            lean_launch.pull_server_answered,
        )

        def pod_status(pod_id, _api_key):
            return {"pod": {"id": pod_id, "desiredStatus": "RUNNING", "costPerHr": 0.59}}

        def terminate_pod(pod_id, _api_key):
            terminated.append(pod_id)

        lean_launch.graphql_api.pod_status = pod_status
        lean_launch.graphql_api.terminate_pod = terminate_pod
        lean_launch.pull_outputs = lambda *_a, **_k: "proxy"
        lean_launch.pull_server_answered = lambda *_a, **_k: True
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                rc = lean_launch.follow(
                    "pod1", "key", "tok", int(datetime.now(timezone.utc).timestamp()), 59, out,
                    sleep=lambda _s: None,
                )
        finally:
            (
                lean_launch.graphql_api.pod_status,
                lean_launch.graphql_api.terminate_pod,
                lean_launch.pull_outputs,
                lean_launch.pull_server_answered,
            ) = saved
            lean_launch.pull_with_deadline.thread = None
        text = buf.getvalue()
        if rc != 1 or "heartbeat stale" not in text or terminated != ["pod1"]:
            raise SystemExit(f"rc={rc} terminated={terminated}\n{text}")
        if "status.json" not in text or "bulk-vox" not in text:
            raise SystemExit(f"tick did not print status.json:\n{text}")
    print("stale heartbeat ok")


def main() -> int:
    check_year_and_segments()
    check_hours()
    check_speed_gate()
    check_warmstart_mix()
    check_fixture_and_dry_run()
    check_same_packages()
    check_crash_writes_failed()
    check_decode_once()
    check_speed_counts_load()
    check_walk_retry()
    check_budget_stop()
    check_heartbeat_file()
    check_stale_heartbeat()
    print("lean test ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
