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
import os
import sys
import tarfile
import threading
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


WALK_RETRIES = 5


def _http_open(url: str, start: int | None):
    """Open a tar URL. A positive start sends ``Range: bytes=<start>-``."""
    import urllib.request

    headers = {"User-Agent": "caramel-italiano-lean/1.0"}
    if start:
        headers["Range"] = f"bytes={int(start)}-"
    req = urllib.request.Request(url, headers=headers)
    return urllib.request.urlopen(req, timeout=120)


def _read_exact(fh, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = fh.read(n - len(buf))
        if not chunk:
            break
        buf += chunk
    return bytes(buf)


def _discard(fh, n: int) -> None:
    left = n
    while left > 0:
        chunk = fh.read(min(1 << 20, left))
        if not chunk:
            raise OSError("truncated tar member")
        left -= len(chunk)


def _copy_member(fh, size: int, dest: Path, on_progress=None) -> None:
    left = size
    with dest.open("wb") as out:
        while left > 0:
            chunk = fh.read(min(1 << 20, left))
            if not chunk:
                raise OSError("truncated tar member")
            out.write(chunk)
            left -= len(chunk)
            if on_progress is not None:
                on_progress()
    pad = (512 - (size % 512)) % 512
    if pad and len(_read_exact(fh, pad)) < pad:
        raise OSError("truncated tar padding")


def _header_at(fh, offset: int):
    """Read one ustar header at ``offset``. Returns (info, next_offset) or None at EOF.

    GNU long-name headers are folded into the following member. ``info.offset``
    is the byte where this member's header starts, which is the Range resume point.
    """
    long_name = None
    while True:
        block = _read_exact(fh, 512)
        if not block:
            return None
        if len(block) < 512:
            raise OSError(f"short tar header at {offset}")
        if block == b"\0" * 512:
            return None
        info = tarfile.TarInfo.frombuf(block, "utf-8", "surrogateescape")
        data_off = offset + 512
        size = int(info.size)
        end = data_off + ((size + 511) // 512) * 512
        if info.type == tarfile.GNUTYPE_LONGNAME:
            raw = _read_exact(fh, size)
            pad = ((size + 511) // 512) * 512 - size
            if pad and len(_read_exact(fh, pad)) < pad:
                raise OSError("truncated long name")
            long_name = raw.split(b"\0", 1)[0].decode("utf-8", "surrogateescape")
            offset = end
            continue
        if long_name:
            info.name = long_name
        info.offset = offset
        info.offset_data = data_off
        return info, end


def _accept_file(path: Path, member: str, check_year: bool, duration_of) -> dict | None:
    dur = float(duration_of(path))
    if dur < lean_data.MIN_SEGMENT_S:
        path.unlink(missing_ok=True)
        return None
    return {
        "source": "voxpopuli_it_2015" if check_year else "mls_it",
        "path": str(path),
        "member": member,
        "name": path.name,
        "duration_s": round(dur, 3),
        "bytes": path.stat().st_size,
    }


def _deliver(rec: dict, path: Path, on_file, kept: list[dict]) -> None:
    if on_file is None:
        kept.append(rec)
        return
    try:
        on_file(rec)
    finally:
        path.unlink(missing_ok=True)


def _retry_pause(fails: int, exc: BaseException, pos: int, pause) -> None:
    print(f"walk_tar retry {fails}/{WALK_RETRIES} at byte {pos}: {exc}", flush=True)
    delay = min(30.0, 2.0 ** (fails - 1))
    pause(delay)


def _walk_range(url, dest, check_year, skip_names, max_new_s, on_file, opener, pause, duration_of, on_progress) -> tuple[list[dict], float]:
    """One HTTP request per member. A dropped body reopens at that member's offset."""
    pos = 0
    fails = 0
    got_s = 0.0
    kept: list[dict] = []
    while True:
        if max_new_s and got_s >= max_new_s:
            return kept, got_s
        resp = None
        path: Path | None = None
        try:
            resp = opener(url, pos)
            parsed = _header_at(resp, pos)
            if parsed is None:
                return kept, got_s
            info, end = parsed
            name = Path(info.name).name
            suffix = Path(info.name).suffix.lower()
            audio = info.type in (tarfile.REGTYPE, tarfile.AREGTYPE) and suffix in lean_data.AUDIO_EXT
            if not audio or name in skip_names or info.name in skip_names:
                _discard(resp, int(info.size))
                pad = (512 - (int(info.size) % 512)) % 512
                if pad:
                    _discard(resp, pad)
                pos = end
                fails = 0
                continue
            if check_year and lean_data.event_year(info.name) != lean_data.VOX_YEAR:
                raise SystemExit(f"{url} member {info.name} is not a {lean_data.VOX_YEAR} event")
            path = dest / name
            # pos stays at info.offset until the body is complete, so a reset retries this member.
            _copy_member(resp, int(info.size), path, on_progress)
            pos = end
            fails = 0
        except (StopTar, SystemExit):
            raise
        except Exception as exc:
            if path is not None:
                path.unlink(missing_ok=True)
            fails += 1
            if fails >= WALK_RETRIES:
                raise
            _retry_pause(fails, exc, pos, pause)
            continue
        finally:
            if resp is not None:
                close = getattr(resp, "close", None)
                if close is not None:
                    close()
        # The socket is closed before the long encode, so the S3 stream is not idle.
        rec = _accept_file(path, info.name, check_year, duration_of)
        if rec is None:
            continue
        stop_after = bool(max_new_s and got_s + rec["duration_s"] > max_new_s and got_s > 0)
        try:
            _deliver(rec, path, on_file, kept)
        except StopTar:
            return kept, got_s
        got_s += rec["duration_s"]
        if stop_after:
            return kept, got_s


def _walk_gzip(url, dest, check_year, skip_names, max_new_s, on_file, opener, pause, duration_of, on_progress) -> tuple[list[dict], float]:
    """MLS is gzip. Restart from the beginning and skip members already saved."""
    import urllib.error

    finished = set(skip_names)
    fails = 0
    got_s = 0.0
    kept: list[dict] = []
    while True:
        if max_new_s and got_s >= max_new_s:
            return kept, got_s
        resp = None
        try:
            resp = opener(url, 0)
            tar = tarfile.open(fileobj=resp, mode="r|gz")
            for member in tar:
                if max_new_s and got_s >= max_new_s:
                    return kept, got_s
                if not member.isfile():
                    continue
                name = Path(member.name).name
                suffix = Path(member.name).suffix.lower()
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                if suffix not in lean_data.AUDIO_EXT or name in finished or member.name in finished:
                    while extracted.read(1 << 20):
                        if on_progress is not None:
                            on_progress()
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
                        if on_progress is not None:
                            on_progress()
                rec = _accept_file(path, member.name, check_year, duration_of)
                if rec is None:
                    finished.add(name)
                    continue
                stop_after = bool(max_new_s and got_s + rec["duration_s"] > max_new_s and got_s > 0)
                try:
                    _deliver(rec, path, on_file, kept)
                except StopTar:
                    return kept, got_s
                finished.add(name)
                got_s += rec["duration_s"]
                if stop_after:
                    return kept, got_s
            return kept, got_s
        except (StopTar, SystemExit):
            raise
        except (OSError, urllib.error.URLError, tarfile.TarError, EOFError) as exc:
            fails += 1
            if fails >= WALK_RETRIES:
                raise
            _retry_pause(fails, exc, 0, pause)
        finally:
            if resp is not None:
                close = getattr(resp, "close", None)
                if close is not None:
                    close()


def walk_tar(
    url: str,
    dest: Path,
    *,
    mode: str,
    check_year: bool,
    skip_names: set[str],
    max_new_s: float,
    on_file=None,
    opener=None,
    pause=time.sleep,
    duration_of=None,
    on_progress=None,
) -> tuple[list[dict], float]:
    """Save audio members. VoxPopuli uses Range; a gzip stream restarts and skips.

    Members in ``skip_names`` are discarded. New audio stops at ``max_new_s``
    (0 means no limit). If ``on_file`` is set it is called with the file still
    on disk, and the file is deleted afterwards. Otherwise the files stay.
    The HTTP body is not held open across ``on_file`` for an uncompressed tar.
    """
    if duration_of is None:
        from download_open import _duration
        duration_of = _duration
    if opener is None:
        opener = _http_open
    dest.mkdir(parents=True, exist_ok=True)
    if mode == "r|gz":
        return _walk_gzip(
            url, dest, check_year, skip_names, max_new_s, on_file, opener, pause, duration_of, on_progress,
        )
    return _walk_range(
        url, dest, check_year, skip_names, max_new_s, on_file, opener, pause, duration_of, on_progress,
    )


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
        job_log(args.out, "lean_prep start")
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


def _iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _disk_free_gb(path: Path) -> float:
    import shutil
    try:
        return round(shutil.disk_usage(path).free / (1024 ** 3), 3)
    except OSError:
        return 0.0


def _mem_available_mb() -> int:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except OSError:
        return 0
    return 0


def _gpu_mem_mb() -> int:
    try:
        import torch
        if torch.cuda.is_available():
            return int(torch.cuda.memory_allocated() / (1024 * 1024))
    except Exception:
        return 0
    return 0


class Heartbeat:
    """Rewrite status.json every 15 s. main_loop_ts moves only when the loop does."""

    def __init__(self, out: Path):
        self.out = Path(out)
        now = _iso()
        self.state = {
            "ts": now,
            "phase": "start",
            "current_member": "",
            "audio_h_done": 0.0,
            "frames": 0,
            "end_to_end_rtf": 0.0,
            "main_loop_ts": now,
            "disk_free_gb": 0.0,
            "mem_available_mb": 0,
            "gpu_mem_used_mb": 0,
        }
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.write()
        self._thread = threading.Thread(target=self._loop, name="lean-heartbeat", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def update(self, **fields) -> None:
        now = _iso()
        self.state["ts"] = now
        self.state["main_loop_ts"] = now
        self.state.update(fields)
        self.write()

    def write(self) -> None:
        self.state["disk_free_gb"] = _disk_free_gb(self.out)
        self.state["mem_available_mb"] = _mem_available_mb()
        self.state["gpu_mem_used_mb"] = _gpu_mem_mb()
        self.state["ts"] = _iso()
        path = self.out / "status.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.state) + "\n", encoding="utf-8")

    def _loop(self) -> None:
        while not self._stop.wait(15):
            try:
                self.write()
            except Exception:
                pass


def job_log(out: Path, line: str) -> None:
    path = out / "logs" / "job.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"{_iso()} {line}\n")
    print(line, flush=True)


def install_fault_handlers(out: Path) -> None:
    """Dump all threads on SIGUSR1. SIGTERM writes FAILED and exits."""
    import faulthandler
    import signal

    log_path = out / "logs" / "faulthandler.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(log_path, "a", buffering=1)
    faulthandler.enable(file=handle, all_threads=True)
    try:
        faulthandler.register(signal.SIGUSR1, file=handle, all_threads=True)
    except (RuntimeError, OSError, ValueError):
        pass

    def on_term(signum, frame) -> None:
        try:
            _write(out / "FAILED", "terminated\n")
        finally:
            os._exit(143)

    signal.signal(signal.SIGTERM, on_term)


def over_encode_budget(elapsed_s: float, budget_s: float) -> bool:
    return float(elapsed_s) >= float(budget_s)


def finish_at_budget(out: Path, audio_s: float) -> None:
    hours = float(audio_s) / 3600.0
    _write(out / "DONE", f"stopped at budget, {hours:.2f} h\n")
    print(f"stopped at budget, {hours:.2f} h", flush=True)


def budget_seconds() -> int:
    raw = os.environ.get("PILOT_ENCODE_BUDGET_S")
    if raw:
        return max(0, int(float(raw)))
    wall = int(os.environ.get("PILOT_MAX_SECONDS", lean_data.MAX_WALL_S))
    return lean_data.encode_budget_s(wall)


def _read_member_mono(path: str):
    from audioio import read_audio, resample, to_mono
    from tokenize_store import SR

    import numpy as np

    wave, sr = read_audio(Path(path))
    mono = to_mono(resample(wave, sr, SR)).reshape(-1)
    return np.ascontiguousarray(mono, dtype=np.float32)


def _encode_until(
    mimi, device: str, writer, clips: list[dict], limit_s: float,
    read_mono=None, encode_fn=None, on_slice=None,
) -> tuple[float, float, float, list[dict]]:
    """Encode slices until limit_s (0 means all).

    Each member file is decoded once. Returns audio seconds, mimi-only seconds,
    whole-slice seconds (load + resample + encode + writer), and the clips left over.
    """
    if read_mono is None:
        read_mono = _read_member_mono
    if encode_fn is None:
        from tokenize_store import _encode_channel

        def encode_fn(piece, _mimi=mimi, _device=device):
            return _encode_channel(_mimi, piece, _device, 0)

    from tokenize_store import SR

    cached_path = None
    cached = None
    audio_s = 0.0
    encode_s = 0.0
    whole_s = 0.0
    used = 0
    for clip in clips:
        if limit_s and audio_s >= limit_s:
            break
        t_all = time.perf_counter()
        path = str(clip["path"])
        if cached_path != path:
            cached = read_mono(path)
            cached_path = path
        start = int(float(clip.get("start_s") or 0) * SR)
        n = max(1, int(float(clip["duration_s"]) * SR))
        piece = cached[start:start + n]
        t_enc = time.perf_counter()
        frames = encode_fn(piece)
        dt = time.perf_counter() - t_enc
        encode_s += dt
        writer.add(frames, float(clip["duration_s"]), dt)
        audio_s += float(clip["duration_s"])
        whole_s += time.perf_counter() - t_all
        used += 1
        if on_slice is not None:
            on_slice(clip, audio_s)
    t_flush = time.perf_counter()
    writer.flush()
    whole_s += time.perf_counter() - t_flush
    return audio_s, encode_s, whole_s, clips[used:]


def wait_for_go(out: Path, timeout_s: int = 20 * 60, heartbeat: Heartbeat | None = None) -> bool:
    """Block until the launcher POSTs /ready (restore/READY) after reading the speed test."""
    ready = out / "restore" / "READY"
    deadline = time.time() + timeout_s
    print("speed_test.json is pullable; waiting for the launcher before the bulk download", flush=True)
    while time.time() < deadline:
        if heartbeat is not None:
            heartbeat.update(phase="wait-go")
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
    audio_s, encode_s, _whole, _rest = _encode_until(mimi, device, writer, clips, seconds)
    return audio_s, encode_s


def _run_pod(args) -> int:
    """Five minutes first, then the launcher decides, then one file at a time."""
    from tokenize_store import _BatchWriter

    install_fault_handlers(args.out)
    hb = Heartbeat(args.out)
    hb.start()
    try:
        return _run_pod_body(args, hb)
    finally:
        hb.stop()


def _frames_of(writer) -> int:
    extra = 0
    for item in getattr(writer, "pending", []):
        extra += len(item)
    return int(getattr(writer, "offset", 0)) + extra


def _run_pod_body(args, hb: Heartbeat) -> int:
    work = args.work
    work.mkdir(parents=True, exist_ok=True)
    _write(args.out / "ATTRIBUTION.txt", lean_data.attribution_text())
    hb.update(phase="download-sample")
    sample_dir = work / "sample"
    sample: list[dict] = []
    sample_bytes = 0
    t_dl = time.perf_counter()

    def touch_download() -> None:
        hb.update(phase="download-sample", current_member=hb.state.get("current_member") or "")

    for url in lean_data.vox_urls():
        if sum(float(r["duration_s"]) for r in sample) >= lean_data.SPEED_TEST_S:
            break
        got, _secs = walk_tar(
            url, sample_dir, mode="r|", check_year=True, skip_names=set(),
            max_new_s=lean_data.SPEED_TEST_S - sum(float(r["duration_s"]) for r in sample),
            on_file=None, on_progress=touch_download,
        )
        sample.extend(got)
        sample_bytes += sum(int(r.get("bytes") or 0) for r in got)
    dl_s = max(1e-6, time.perf_counter() - t_dl)
    clips = segments_for(sample)
    hb.update(phase="speed-test", current_member=sample[0]["name"] if sample else "")
    mimi, device = _open_mimi()
    vox_store = args.out / "stores" / lean_data.STORE_VOX
    vox_writer = _BatchWriter(vox_store, lean_data.WIDTH, 50)
    audio_done = 0.0
    whole_done = 0.0

    def on_slice(clip: dict, audio_in_call: float, phase: str, writer) -> None:
        done_s = audio_done + audio_in_call
        rtf = (done_s / whole_done) if whole_done > 0 else 0.0
        hb.update(
            phase=phase,
            current_member=str(clip.get("name") or Path(str(clip["path"])).name),
            audio_h_done=round(done_s / 3600.0, 4),
            frames=_frames_of(writer),
            end_to_end_rtf=round(rtf, 3),
        )

    audio_s, encode_s, whole_s, rest = _encode_until(
        mimi, device, vox_writer, clips, lean_data.SPEED_TEST_S,
        on_slice=lambda clip, n: on_slice(clip, n, "speed-test", vox_writer),
    )
    audio_done = audio_s
    whole_done = whole_s
    if audio_s <= 0 or encode_s <= 0:
        _write(args.out / "FAILED", "speed test saw no audio\n")
        print("speed test saw no audio", file=sys.stderr)
        return 1
    sample_audio = sum(float(r["duration_s"]) for r in sample)
    allowance, mbps = lean_data.projected_download_s(sample_bytes, dl_s, sample_audio)
    report = lean_data.speed_report(
        audio_s, encode_s, args.hourly_cents, args.cap_cents,
        end_to_end_s=whole_s, download_s=allowance, download_mbps=mbps,
    )
    _write(args.out / "speed_test.json", json.dumps(report, indent=2) + "\n")
    hb.update(
        phase="speed-test",
        audio_h_done=round(audio_done / 3600.0, 4),
        frames=_frames_of(vox_writer),
        end_to_end_rtf=round(report["end_to_end_rtf"], 3),
    )
    print(
        f"speed test {audio_s:.1f}s in {encode_s:.1f}s encode, "
        f"{report['end_to_end_rtf']:.1f}x end to end, "
        f"download {mbps:.2f} MB/s, projected ${report['cents'] / 100:.2f}",
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
    if not wait_for_go(args.out, heartbeat=hb):
        _write(args.out / "FAILED", "launcher did not approve the speed test\n")
        print("launcher did not approve the speed test", file=sys.stderr)
        return 1
    bulk_t0 = time.monotonic()
    budget = budget_seconds()
    if rest:
        more, _enc, more_whole, _left = _encode_until(
            mimi, device, vox_writer, rest, 0,
            on_slice=lambda clip, n: on_slice(clip, n, "bulk-vox", vox_writer),
        )
        audio_done += more
        whole_done += more_whole
    for rec in sample:
        Path(rec["path"]).unlink(missing_ok=True)
    done = {rec["name"] for rec in sample}
    vox_left = lean_data.VOX_CAP_H * 3600.0 - audio_done

    def budget_hit() -> bool:
        if (args.out / "DONE").is_file():
            return True
        if over_encode_budget(time.monotonic() - bulk_t0, budget):
            finish_at_budget(args.out, audio_done)
            hb.update(phase="budget", audio_h_done=round(audio_done / 3600.0, 4), frames=_frames_of(vox_writer))
            return True
        return False

    def encode_one(store_writer, rec: dict, phase: str) -> None:
        nonlocal audio_done, whole_done
        segs = segments_for([rec])
        more, _enc, more_whole, _left = _encode_until(
            mimi, device, store_writer, segs, 0,
            on_slice=lambda clip, n, writer=store_writer: on_slice(clip, n, phase, writer),
        )
        audio_done += more
        whole_done += more_whole
        hb.update(
            phase=phase,
            current_member=rec["name"],
            audio_h_done=round(audio_done / 3600.0, 4),
            frames=_frames_of(store_writer),
            end_to_end_rtf=round(audio_done / whole_done, 3) if whole_done else 0.0,
        )

    for url in lean_data.vox_urls():
        if vox_left <= 0 or budget_hit():
            break

        def on_vox(rec: dict, writer=vox_writer) -> None:
            nonlocal vox_left
            if budget_hit():
                raise StopTar()
            hb.update(phase="bulk-vox", current_member=rec["name"])
            encode_one(writer, rec, "bulk-vox")
            vox_left -= float(rec["duration_s"])

        walk_tar(
            url, work / "vox", mode="r|", check_year=True, skip_names=done,
            max_new_s=max(0.0, vox_left), on_file=on_vox,
            on_progress=lambda: hb.update(phase="bulk-vox"),
        )
    if (args.out / "DONE").is_file():
        return 0
    mls_store = args.out / "stores" / lean_data.STORE_MLS
    mls_writer = _BatchWriter(mls_store, lean_data.WIDTH, 50)
    skipped = 0.0
    kept = 0.0
    skip_cap = lean_data.MLS_ALREADY_H * 3600.0
    keep_cap = lean_data.MLS_KEEP_H * 3600.0

    def on_mls(rec: dict) -> None:
        nonlocal skipped, kept
        if budget_hit():
            raise StopTar()
        dur = float(rec["duration_s"])
        if skipped < skip_cap and not (skipped > 0 and skipped + dur > skip_cap):
            skipped += dur
            hb.update(phase="skip-mls", current_member=rec["name"])
            return
        if kept >= keep_cap or (kept > 0 and kept + dur > keep_cap):
            raise StopTar()
        hb.update(phase="bulk-mls", current_member=rec["name"])
        encode_one(mls_writer, rec, "bulk-mls")
        kept += dur

    if not budget_hit():
        walk_tar(
            lean_data.MLS_URL, work / "mls", mode="r|gz", check_year=False, skip_names=set(),
            max_new_s=0, on_file=on_mls,
            on_progress=lambda: hb.update(phase="bulk-mls"),
        )
    if not (args.out / "DONE").is_file():
        hb.update(phase="done", audio_h_done=round(audio_done / 3600.0, 4))
        _write(args.out / "DONE", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
