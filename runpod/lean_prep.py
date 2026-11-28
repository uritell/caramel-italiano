#!/usr/bin/env python3
"""On the pod: download the lean audio, encode it, publish tokens only.

    python3 runpod/lean_prep.py --speed-test --fixture runpod/testdata/lean
    python3 runpod/lean_prep.py --pod          # full run; speed test first

``--pod`` downloads about five minutes from the first tar, encodes it, and
writes speed_test.json for the launcher. Bulk download starts only after
the launcher POSTs /ready. Later files are encoded and deleted one at a
time, and the token store in the pull directory grows as it goes.
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


class StopTar(Exception):
    """The caller has all the audio it wants from this stream."""


def walk_tar(
    url: str,
    dest: Path,
    *,
    mode: str,
    check_year: bool,
    skip_names: set[str],
    max_new_s: float,
    on_file=None,
) -> tuple[list[dict], float]:
    """Stream one tar. ``mode`` is ``r|`` or ``r|gz``.

    Members in ``skip_names`` are discarded. New audio stops at ``max_new_s``
    (0 means no limit). If ``on_file`` is set it is called with the file still
    on disk, and the file is deleted afterwards. Otherwise the files stay.
    """
    import tarfile
    import urllib.request

    from download_open import _duration

    dest.mkdir(parents=True, exist_ok=True)
    kept: list[dict] = []
    got_s = 0.0
    req = urllib.request.Request(url, headers={"User-Agent": "caramel-italiano-lean/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        tar = tarfile.open(fileobj=resp, mode=mode)
        for member in tar:
            if max_new_s and got_s >= max_new_s:
                break
            if not member.isfile():
                continue
            suffix = Path(member.name).suffix.lower()
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            name = Path(member.name).name
            if suffix not in lean_data.AUDIO_EXT or name in skip_names or member.name in skip_names:
                while extracted.read(1 << 20):
                    pass
                continue
            if check_year and lean_data.event_year(member.name) != lean_data.VOX_YEAR:
                raise SystemExit(f"{url} member {member.name} is not a {lean_data.VOX_YEAR} event")
            path = dest / name
            with path.open("wb") as fh:
                while True:
                    block = extracted.read(1 << 20)
                    if not block:
                        break
                    fh.write(block)
            dur = _duration(path)
            if dur < lean_data.MIN_SEGMENT_S:
                path.unlink(missing_ok=True)
                continue
            stop_after = bool(max_new_s and got_s + dur > max_new_s and got_s > 0)
            rec = {
                "source": "voxpopuli_it_2015" if check_year else "mls_it",
                "path": str(path),
                "member": member.name,
                "name": name,
                "duration_s": round(dur, 3),
            }
            if on_file is not None:
                try:
                    on_file(rec)
                except StopTar:
                    path.unlink(missing_ok=True)
                    return kept, got_s
                finally:
                    path.unlink(missing_ok=True)
            else:
                kept.append(rec)
            got_s += dur
            if stop_after:
                break
    return kept, got_s


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


def record_failure(out: Path, exc: BaseException) -> None:
    """Write a traceback the launcher can pull. One file means stop the pod."""
    import traceback

    text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    _write(out / "FAILED", text)
    print(text, file=sys.stderr)


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
        try:
            return _run_pod(args)
        except SystemExit as exc:
            if not (args.out / "FAILED").is_file():
                record_failure(args.out, exc)
            code = exc.code
            return code if isinstance(code, int) else 1
        except Exception as exc:
            record_failure(args.out, exc)
            return 1
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


def _open_mimi():
    from tokenize_store import _load_mimi

    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
    except Exception:
        device = "cpu"
    return _load_mimi(device), device


def _encode_until(mimi, device: str, writer, clips: list[dict], limit_s: float) -> tuple[float, float, list[dict]]:
    """Encode slices until limit_s (0 means all). Returns audio, encode time, and the rest."""
    from audioio import read_audio, resample, to_mono
    from tokenize_store import SR, _encode_channel

    audio_s = 0.0
    encode_s = 0.0
    used = 0
    for clip in clips:
        if limit_s and audio_s >= limit_s:
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
        used += 1
    writer.flush()
    return audio_s, encode_s, clips[used:]


def wait_for_go(out: Path, timeout_s: int = 20 * 60) -> bool:
    """Block until the launcher POSTs /ready (restore/READY) after reading the speed test."""
    ready = out / "restore" / "READY"
    deadline = time.time() + timeout_s
    print("speed_test.json is pullable; waiting for the launcher before the bulk download", flush=True)
    while time.time() < deadline:
        if ready.is_file():
            print("launcher approved the speed test", flush=True)
            return True
        time.sleep(5)
    return False


def _encode_store(clips: list[dict], out: Path, seconds: float = 0.0) -> tuple[float, float]:
    """Encode slices with one Mimi load. Used by tests and the speed-test prefix."""
    from tokenize_store import _BatchWriter

    mimi, device = _open_mimi()
    writer = _BatchWriter(out, lean_data.WIDTH, 2000)
    audio_s, encode_s, _rest = _encode_until(mimi, device, writer, clips, seconds)
    return audio_s, encode_s


def _run_pod(args) -> int:
    """Five minutes first, then the launcher decides, then one file at a time."""
    from tokenize_store import _BatchWriter

    work = args.work
    work.mkdir(parents=True, exist_ok=True)
    _write(args.out / "ATTRIBUTION.txt", lean_data.attribution_text())
    sample_dir = work / "sample"
    sample: list[dict] = []
    for url in lean_data.vox_urls():
        if sum(float(r["duration_s"]) for r in sample) >= lean_data.SPEED_TEST_S:
            break
        got, _secs = walk_tar(
            url, sample_dir, mode="r|", check_year=True, skip_names=set(),
            max_new_s=lean_data.SPEED_TEST_S - sum(float(r["duration_s"]) for r in sample),
            on_file=None,
        )
        sample.extend(got)
    clips = segments_for(sample)
    mimi, device = _open_mimi()
    vox_store = args.out / "stores" / lean_data.STORE_VOX
    vox_writer = _BatchWriter(vox_store, lean_data.WIDTH, 50)
    audio_s, encode_s, rest = _encode_until(mimi, device, vox_writer, clips, lean_data.SPEED_TEST_S)
    if audio_s <= 0:
        _write(args.out / "FAILED", "speed test saw no audio\n")
        print("speed test saw no audio", file=sys.stderr)
        return 1
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
        for rec in sample:
            Path(rec["path"]).unlink(missing_ok=True)
        _write(args.out / "FAILED", str(exc) + "\n")
        print(exc, file=sys.stderr)
        return 1
    if not wait_for_go(args.out):
        _write(args.out / "FAILED", "launcher did not approve the speed test\n")
        print("launcher did not approve the speed test", file=sys.stderr)
        return 1
    if rest:
        _encode_until(mimi, device, vox_writer, rest, 0)
    for rec in sample:
        Path(rec["path"]).unlink(missing_ok=True)
    done = {rec["name"] for rec in sample}
    vox_left = lean_data.VOX_CAP_H * 3600.0 - audio_s

    def encode_one(store_writer, rec: dict) -> None:
        segs = segments_for([rec])
        _encode_until(mimi, device, store_writer, segs, 0)

    for url in lean_data.vox_urls():
        if vox_left <= 0:
            break

        def on_vox(rec: dict, writer=vox_writer) -> None:
            nonlocal vox_left
            encode_one(writer, rec)
            vox_left -= float(rec["duration_s"])

        _added, _got = walk_tar(
            url, work / "vox", mode="r|", check_year=True, skip_names=done,
            max_new_s=max(0.0, vox_left), on_file=on_vox,
        )
    mls_store = args.out / "stores" / lean_data.STORE_MLS
    mls_writer = _BatchWriter(mls_store, lean_data.WIDTH, 50)
    skipped = 0.0
    kept = 0.0
    skip_cap = lean_data.MLS_ALREADY_H * 3600.0
    keep_cap = lean_data.MLS_KEEP_H * 3600.0

    def on_mls(rec: dict) -> None:
        nonlocal skipped, kept
        dur = float(rec["duration_s"])
        if skipped < skip_cap and not (skipped > 0 and skipped + dur > skip_cap):
            skipped += dur
            return
        if kept >= keep_cap or (kept > 0 and kept + dur > keep_cap):
            raise StopTar()
        encode_one(mls_writer, rec)
        kept += dur

    walk_tar(
        lean_data.MLS_URL, work / "mls", mode="r|gz", check_year=False, skip_names=set(),
        max_new_s=0, on_file=on_mls,
    )
    _write(args.out / "DONE", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
