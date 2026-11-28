#!/usr/bin/env python3
"""On the pod: download the lean audio, encode it, publish tokens only.

    python3 runpod/lean_prep.py --speed-test --fixture runpod/testdata/lean
    python3 runpod/lean_prep.py --pod          # full run; speed test first

``--pod`` is what lean_bootstrap.sh runs. It encodes five minutes, writes
speed_test.json, and exits before the full encode when that projection is
over the cap. Raw audio is not copied into the pull directory.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lean_data


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def publish_tokens(store_dirs: list[Path], out: Path) -> list[str]:
    """Copy store files and attribution. Leave the wavs behind."""
    import shutil

    published = []
    out.mkdir(parents=True, exist_ok=True)
    _write(out / "ATTRIBUTION.txt", lean_data.attribution_text())
    published.append("ATTRIBUTION.txt")
    for store in store_dirs:
        if not store.is_dir():
            continue
        dest = out / "stores" / store.name
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("tokens.bin", "index.npy", "meta.json"):
            src = store / name
            if src.is_file():
                shutil.copy2(src, dest / name)
                published.append(f"stores/{store.name}/{name}")
    return published


def gate(report: dict, cap_cents: int) -> None:
    if report.get("over_cap") or lean_data.over_cap(int(report["cents"]), cap_cents):
        raise SystemExit(
            f"speed test projects ${int(report['cents']) / 100:.2f}, "
            f"over the ${cap_cents / 100:.2f} cap; not starting the full encode"
        )


def encode_speed(clips: list[dict], encode, seconds: float = lean_data.SPEED_TEST_S) -> tuple[float, float]:
    """Encode a deterministic prefix until ``seconds`` of audio. encode(clip) -> (frames, dt)."""
    audio_s = 0.0
    encode_s = 0.0
    for clip in clips:
        if audio_s >= seconds:
            break
        _frames, dt = encode(clip)
        audio_s += float(clip["duration_s"])
        encode_s += float(dt)
    return audio_s, encode_s


def stream_vox(url: str, dest: Path, hours_left: float) -> tuple[list[dict], float]:
    """Stream one official tar. Stop at the hour cap. Abort if a member is not 2015."""
    import tarfile
    import urllib.request

    dest.mkdir(parents=True, exist_ok=True)
    records = []
    got_s = 0.0
    cap_s = hours_left * 3600.0
    req = urllib.request.Request(url, headers={"User-Agent": "caramel-italiano-lean/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        tar = tarfile.open(fileobj=resp, mode="r|")
        for member in tar:
            if got_s >= cap_s:
                break
            if not member.isfile():
                continue
            if Path(member.name).suffix.lower() not in lean_data.AUDIO_EXT:
                extracted = tar.extractfile(member)
                if extracted is not None:
                    while extracted.read(1 << 20):
                        pass
                continue
            if lean_data.event_year(member.name) != lean_data.VOX_YEAR:
                raise SystemExit(f"{url} member {member.name} is not a {lean_data.VOX_YEAR} event")
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            path = dest / Path(member.name).name
            path.write_bytes(extracted.read())
            from download_open import _duration

            dur = _duration(path)
            if dur < lean_data.MIN_SEGMENT_S:
                path.unlink(missing_ok=True)
                continue
            if got_s + dur > cap_s and got_s > 0:
                path.unlink(missing_ok=True)
                break
            got_s += dur
            records.append({
                "source": "voxpopuli_it_2015",
                "path": str(path),
                "duration_s": round(dur, 3),
                "year": lean_data.VOX_YEAR,
            })
    return records, got_s / 3600.0


def segments_for(records: list[dict]) -> list[dict]:
    """One jsonl row per 12 s slice, archive order, deterministic."""
    rows = []
    for rec in records:
        for start, dur in lean_data.segment_duration(float(rec["duration_s"])):
            rows.append({
                **rec,
                "start_s": start,
                "duration_s": dur,
            })
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pod", action="store_true", help="full pod run: speed test, then encode if under the cap")
    p.add_argument("--speed-test", action="store_true", help="encode about 5 min and write the projection")
    p.add_argument("--fixture", type=Path, default=None, help="local clips jsonl; no download")
    p.add_argument("--out", type=Path, default=Path("/workspace/pilot-out"))
    p.add_argument("--work", type=Path, default=Path("/workspace/work/lean"))
    p.add_argument("--hourly-cents", type=int, default=lean_data.A40_HOURLY_CENTS)
    p.add_argument("--cap-cents", type=int, default=lean_data.MAX_USD_CENTS)
    args = p.parse_args(argv)
    if args.fixture is None and not args.pod:
        print("lean_prep.py: pass --fixture for a local check, or --pod on the machine", file=sys.stderr)
        return 2
    if args.pod and args.fixture is None:
        return _run_pod(args)
    clips = []
    if args.fixture is not None:
        for line in args.fixture.read_text(encoding="utf-8").splitlines():
            if line.strip():
                clips.append(json.loads(line))
    if not clips and args.fixture is not None:
        print("fixture has no clips", file=sys.stderr)
        return 1

    def fake_encode(clip: dict):
        # Fixture stand-in, not a GPU measurement. Fast enough that 5 min
        # projects under the cap; the slow case is checked in gate().
        return [], max(0.001, float(clip["duration_s"]) / 600.0)

    audio_s, encode_s = encode_speed(clips, fake_encode)
    if audio_s <= 0:
        print("speed test saw no audio", file=sys.stderr)
        return 1
    report = lean_data.speed_report(audio_s, encode_s, args.hourly_cents, args.cap_cents)
    _write(args.out / "speed_test.json", json.dumps(report, indent=2) + "\n")
    print(
        f"speed test {audio_s:.1f}s audio in {encode_s:.2f}s, "
        f"{report['rtf']:.1f}x, full run ${report['cents'] / 100:.2f}",
        flush=True,
    )
    if args.speed_test and not args.pod:
        try:
            gate(report, args.cap_cents)
        except SystemExit as exc:
            print(exc, file=sys.stderr)
            return 1
        return 0
    try:
        gate(report, args.cap_cents)
    except SystemExit as exc:
        _write(args.out / "FAILED", str(exc) + "\n")
        print(exc, file=sys.stderr)
        return 1
    if not args.pod:
        return 0
    print("full encode is the pod path; this process stops after the gate in the fixture", flush=True)
    _write(args.out / "DONE", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
    return 0


def _encode_store(clips: list[dict], out: Path, seconds: float = 0.0) -> tuple[float, float]:
    """Encode 12 s slices with the same Mimi settings as the mono store."""
    from audioio import read_audio, resample, to_mono
    from tokenize_store import SR, _BatchWriter, _encode_channel, _load_mimi

    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
    except Exception:
        device = "cpu"
    mimi = _load_mimi(device)
    writer = _BatchWriter(out, lean_data.WIDTH, 2000)
    audio_s = 0.0
    encode_s = 0.0
    for clip in clips:
        if seconds and audio_s >= seconds:
            break
        wave, sr = read_audio(Path(clip["path"]))
        mono = to_mono(resample(wave, sr, SR)).reshape(-1)
        start = int(float(clip.get("start_s") or 0) * SR)
        n = max(1, int(float(clip["duration_s"]) * SR))
        piece = mono[start:start + n]
        t0 = time.perf_counter()
        frames = _encode_channel(mimi, piece, device, 0)
        dt = time.perf_counter() - t0
        writer.add(frames, float(clip["duration_s"]), dt)
        audio_s += float(clip["duration_s"])
        encode_s += dt
    writer.flush()
    return audio_s, encode_s


def _run_pod(args) -> int:
    work = args.work
    work.mkdir(parents=True, exist_ok=True)
    vox_records = []
    left = lean_data.VOX_CAP_H
    for url in lean_data.vox_urls():
        if left <= 0:
            break
        got, hours = stream_vox(url, work / "vox", left)
        vox_records.extend(got)
        left -= hours
    vox_clips = segments_for(vox_records)
    from download_open import stream_audio, _mls_speaker

    mls_raw = stream_audio(
        lean_data.MLS_URL, work / "mls",
        source="mls_it", split="italian",
        max_audio_s=0, max_bytes=0, speaker_from_name=_mls_speaker,
    )
    durations = [float(row["duration_s"]) for row in mls_raw if float(row["duration_s"]) > 0]
    n_skip = len(lean_data.take_hours(durations, lean_data.MLS_ALREADY_H))
    n_keep = len(lean_data.mls_remainder(durations))
    positive = [row for row in mls_raw if float(row["duration_s"]) > 0]
    mls_kept = positive[n_skip:n_skip + n_keep]
    mls_clips = segments_for(mls_kept)
    _write(work / "ATTRIBUTION.txt", lean_data.attribution_text())
    speed_clips = vox_clips or mls_clips
    audio_s, encode_s = _encode_store(speed_clips, work / "speed", lean_data.SPEED_TEST_S)
    report = lean_data.speed_report(audio_s, encode_s, args.hourly_cents, args.cap_cents)
    _write(args.out / "speed_test.json", json.dumps(report, indent=2) + "\n")
    print(
        f"speed test {audio_s:.1f}s in {encode_s:.1f}s, {report['rtf']:.1f}x, "
        f"projected ${report['cents'] / 100:.2f}",
        flush=True,
    )
    try:
        gate(report, args.cap_cents)
    except SystemExit as exc:
        _write(args.out / "FAILED", str(exc) + "\n")
        print(exc, file=sys.stderr)
        return 1
    vox_store = work / "stores" / lean_data.STORE_VOX
    mls_store = work / "stores" / lean_data.STORE_MLS
    _encode_store(vox_clips, vox_store)
    _encode_store(mls_clips, mls_store)
    publish_tokens([vox_store, mls_store], args.out)
    _write(args.out / "ATTRIBUTION.txt", lean_data.attribution_text())
    _write(args.out / "DONE", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
