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
    if len(vox) != 1000:
        raise SystemExit(f"vox cap kept {len(vox)} hours, not 1000")
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
    for needle in ("it_2015.tar", "it_2015_2.tar", "No pod was created", "158", "1000", "$3.00", "6 h"):
        if needle not in text:
            raise SystemExit(f"dry run missing {needle}:\n{text}")
    if "creating the pod" in text.lower():
        raise SystemExit("dry run tried to create a pod")
    print("fixture and dry run ok")


def main() -> int:
    check_year_and_segments()
    check_hours()
    check_speed_gate()
    check_warmstart_mix()
    check_fixture_and_dry_run()
    print("lean test ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
