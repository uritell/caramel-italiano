#!/usr/bin/env python3
"""Encode clips with Kyutai Mimi into a caramel TokenStore.

The store is the layout train.py reads: tokens.bin uint16 [frames, W],
index.npy int64 [clips, 2] of (offset, length) in frames, meta.json
{"width", "frames"}. W is 8 for one speaker and 16 for stereo (left
codebooks, then right). The delay pattern is applied later by train.py.

Throughput is written to speed.json next to the store. On two GPUs the
clip list is split and the stores are merged.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SR = 24000
FPS = 12.5
CODEBOOKS = 8


def _load_clips(path: Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _codes_to_frames(codes) -> np.ndarray:
    arr = codes.detach().to("cpu")
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        raise RuntimeError(f"unexpected Mimi code shape {tuple(arr.shape)}")
    if arr.shape[0] == CODEBOOKS:
        arr = arr.transpose(0, 1)
    elif arr.shape[1] != CODEBOOKS:
        raise RuntimeError(f"neither axis is {CODEBOOKS} codebooks: {tuple(arr.shape)}")
    out = arr.contiguous().numpy().astype(np.int64)
    if out.min() < 0 or out.max() > 65535:
        raise RuntimeError(f"codebook values outside uint16: {out.min()}..{out.max()}")
    return out


def _encode_channel(mimi, wave: np.ndarray, device: str, chunk_s: float) -> np.ndarray:
    import torch

    wave = np.ascontiguousarray(wave.reshape(-1), dtype=np.float32)
    chunk = max(1, int(chunk_s * SR)) if chunk_s else len(wave)
    parts = []
    with torch.inference_mode():
        for start in range(0, len(wave), chunk):
            piece = wave[start:start + chunk]
            if len(piece) < int(0.05 * SR):
                continue
            wav = torch.from_numpy(piece).to(device).view(1, 1, -1)
            enc = mimi.encode(wav, num_quantizers=CODEBOOKS)
            parts.append(_codes_to_frames(enc.audio_codes))
    if not parts:
        return np.zeros((0, CODEBOOKS), dtype=np.int64)
    return np.concatenate(parts, axis=0)


def _load_mimi(device: str):
    import torch
    from transformers import MimiModel

    t0 = time.perf_counter()
    model = MimiModel.from_pretrained("kyutai/mimi").to(device).eval()
    print(f"mimi loaded on {device} in {time.perf_counter() - t0:.1f}s", flush=True)
    return model


def _load_wave(path: str, width: int) -> np.ndarray:
    """CPU load + resample. Top-level so a process pool can call it."""
    from audioio import read_audio, resample, to_mono, to_stereo

    audio, sr = read_audio(Path(path))
    audio = resample(audio, sr, SR)
    if width == 8:
        return np.ascontiguousarray(to_mono(audio))
    return np.ascontiguousarray(to_stereo(audio))


class _BatchWriter:
    """Append token batches so a crash keeps every finished batch."""

    def __init__(self, out: Path, width: int, batch: int):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.batch = max(1, batch)
        self.path = self.out / "tokens.bin"
        self.pending: list[np.ndarray] = []
        self.audio_s = 0.0
        self.encode_s = 0.0
        if (self.out / "index.npy").exists() and self.path.exists():
            saved = np.load(self.out / "index.npy")
            self.index = [tuple(map(int, row)) for row in saved]
            self.offset = int(self.index[-1][0] + self.index[-1][1]) if self.index else 0
        else:
            self.index = []
            self.offset = 0

    @property
    def clips_done(self) -> int:
        return len(self.index)

    def add(self, frames: np.ndarray, audio_s: float, encode_s: float) -> None:
        if len(frames) == 0:
            return
        self.pending.append(np.ascontiguousarray(frames, dtype=np.uint16))
        self.audio_s += audio_s
        self.encode_s += encode_s
        if len(self.pending) >= self.batch:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        blob = np.concatenate(self.pending, axis=0)
        with self.path.open("ab") as fh:
            fh.write(blob.tobytes())
        for frames in self.pending:
            self.index.append((self.offset, int(len(frames))))
            self.offset += int(len(frames))
        self.pending.clear()
        np.save(self.out / "index.npy", np.asarray(self.index, dtype=np.int64))
        (self.out / "meta.json").write_text(json.dumps({
            "width": self.width, "frames": self.offset, "clips": len(self.index),
        }) + "\n")


def encode_clips(
    clips: list[dict],
    out: Path,
    width: int,
    *,
    device: str,
    chunk_s: float,
    deadline_epoch: int,
    reserve_s: int,
) -> dict:
    import torch

    if width not in (8, 16):
        raise SystemExit("width must be 8 or 16")
    batch = int(os.environ.get("PILOT_TOKEN_BATCH", "2000"))
    writer = _BatchWriter(out, width, batch)
    if writer.clips_done:
        print(f"resuming {out} at clip {writer.clips_done}", flush=True)
        clips = clips[writer.clips_done:]
    if not clips and writer.clips_done:
        return _finish_writer(writer, device)
    mimi = _load_mimi(device)
    workers = int(os.environ.get("PILOT_LOAD_WORKERS", "4"))
    pool = None
    if workers > 0 and len(clips) > 1:
        from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

        pool = ProcessPoolExecutor(max_workers=workers)
    try:
        if pool is None:
            loaded = ((clip, _load_wave(clip["path"], width)) for clip in clips)
            _encode_loaded(mimi, writer, loaded, width, device, chunk_s, deadline_epoch, reserve_s)
        else:
            _encode_prefetched(pool, mimi, writer, clips, width, device, chunk_s, deadline_epoch, reserve_s, FIRST_COMPLETED, wait)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
        del mimi
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    writer.flush()
    if not writer.index:
        raise SystemExit(f"no tokens written for {out}")
    return _finish_writer(writer, device)


def _past_deadline(deadline_epoch: int, reserve_s: int) -> bool:
    if not deadline_epoch or not reserve_s:
        return False
    remaining = deadline_epoch + int(os.environ.get("PILOT_MAX_SECONDS", "9000")) - time.time()
    if remaining < reserve_s:
        print(f"stopping tokenize with {remaining:.0f}s left (reserve {reserve_s}s)", flush=True)
        return True
    return False


def _frames_for(mimi, wave: np.ndarray, width: int, device: str, chunk_s: float) -> tuple[np.ndarray, float]:
    audio_s = (len(wave) if wave.ndim == 1 else len(wave)) / SR
    t0 = time.perf_counter()
    if width == 8:
        frames = _encode_channel(mimi, wave, device, chunk_s)
    else:
        left = _encode_channel(mimi, wave[:, 0], device, chunk_s)
        right = _encode_channel(mimi, wave[:, 1], device, chunk_s)
        n = min(len(left), len(right))
        frames = np.concatenate([left[:n], right[:n]], axis=1) if n else np.zeros((0, 16), np.int64)
    return frames, time.perf_counter() - t0


def _encode_loaded(mimi, writer, loaded, width, device, chunk_s, deadline_epoch, reserve_s) -> None:
    for n, (clip, wave) in enumerate(loaded, start=writer.clips_done + 1):
        if _past_deadline(deadline_epoch, reserve_s):
            break
        frames, dt = _frames_for(mimi, wave, width, device, chunk_s)
        audio_s = len(wave) / SR
        writer.add(frames, audio_s, dt)
        if n % 10 == 0:
            rate = writer.audio_s / writer.encode_s if writer.encode_s else 0
            print(f"  clip {n}  audio {writer.audio_s:.1f}s  {rate:.2f}x realtime", flush=True)


def _encode_prefetched(pool, mimi, writer, clips, width, device, chunk_s, deadline_epoch, reserve_s, first_completed, wait) -> None:
    inflight = {}
    pending = iter(clips)
    window = max(2, int(os.environ.get("PILOT_LOAD_WORKERS", "4")))

    def fill() -> None:
        while len(inflight) < window:
            try:
                clip = next(pending)
            except StopIteration:
                return
            inflight[pool.submit(_load_wave, clip["path"], width)] = clip

    fill()
    n = writer.clips_done
    while inflight:
        if _past_deadline(deadline_epoch, reserve_s):
            break
        done, _ = wait(inflight, return_when=first_completed)
        for fut in done:
            clip = inflight.pop(fut)
            wave = fut.result()
            frames, dt = _frames_for(mimi, wave, width, device, chunk_s)
            writer.add(frames, len(wave) / SR, dt)
            n += 1
            if n % 10 == 0:
                rate = writer.audio_s / writer.encode_s if writer.encode_s else 0
                print(f"  clip {n}  audio {writer.audio_s:.1f}s  {rate:.2f}x realtime", flush=True)
        fill()


def _finish_writer(writer: _BatchWriter, device: str) -> dict:
    writer.flush()
    speed = {
        "audio_s": round(writer.audio_s, 3),
        "encode_s": round(writer.encode_s, 3),
        "frames": writer.offset,
        "clips": len(writer.index),
        "width": writer.width,
        "frames_per_s": round(writer.offset / writer.encode_s, 1) if writer.encode_s else 0,
        "realtime_factor": round(writer.audio_s / writer.encode_s, 3) if writer.encode_s else None,
        "device": device,
    }
    (writer.out / "speed.json").write_text(json.dumps(speed, indent=2) + "\n")
    (writer.out / "meta.json").write_text(json.dumps({
        "width": writer.width, "frames": writer.offset, "clips": len(writer.index),
    }) + "\n")
    print(json.dumps(speed), flush=True)
    _verify(writer.out)
    return speed


def self_test() -> None:
    """Force the two-worker merge into a directory that does not exist yet."""
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="tokenize-merge-"))
    parts = []
    for i in range(2):
        part = tmp / f"part{i}"
        frames = [np.arange(4 * 8, dtype=np.uint16).reshape(4, 8) + i]
        _write_store(part, frames, [(0, 4)], 8, 0.3, 0.1, "cpu")
        parts.append(part)
    dest = tmp / "stores" / "mono"
    if dest.exists():
        raise SystemExit("test dest should not exist before merge")
    os.environ["PILOT_TOKENIZE_FORCE_PARALLEL"] = "1"
    _merge(parts, dest, 8)
    if not (dest / "tokens.bin").is_file():
        raise SystemExit("merge did not create tokens.bin")
    meta = json.loads((dest / "meta.json").read_text())
    if meta["frames"] != 8 or meta["width"] != 8:
        raise SystemExit(f"merged meta wrong: {meta}")
    # A second batch appends without rewriting the first frames.
    writer = _BatchWriter(dest, 8, batch=1)
    if writer.clips_done != 2:
        raise SystemExit(f"resume saw {writer.clips_done} clips, want 2")
    writer.add(np.zeros((2, 8), np.uint16), 0.1, 0.01)
    writer.flush()
    if writer.clips_done != 3 or writer.offset != 10:
        raise SystemExit(f"append failed: clips={writer.clips_done} offset={writer.offset}")
    print(f"tokenize merge self-test ok {dest}", flush=True)


def _write_store(out, tokens, index, width, audio_s, encode_s, device) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    all_tok = np.concatenate(tokens, axis=0)
    if all_tok.shape[1] != width:
        raise SystemExit(f"width {all_tok.shape[1]} != {width}")
    (out / "tokens.bin").write_bytes(np.ascontiguousarray(all_tok, dtype=np.uint16).tobytes())
    idx = np.asarray(index, dtype=np.int64)
    np.save(out / "index.npy", idx)
    meta = {"width": int(width), "frames": int(all_tok.shape[0])}
    (out / "meta.json").write_text(json.dumps(meta) + "\n")
    frames_s = meta["frames"] / encode_s if encode_s else 0
    speed = {
        "audio_s": round(audio_s, 3),
        "encode_s": round(encode_s, 3),
        "frames": meta["frames"],
        "clips": int(len(index)),
        "width": width,
        "frames_per_s": round(frames_s, 1),
        "realtime_factor": round(audio_s / encode_s, 3) if encode_s else None,
        "device": device,
    }
    (out / "speed.json").write_text(json.dumps(speed, indent=2) + "\n")
    print(json.dumps(speed), flush=True)
    _verify(out)
    return speed


def write_placeholder(clips: list[dict], out: Path, width: int) -> dict:
    """Structurally valid store used only when the CPU smoke test cannot load Mimi.

    Values are token ids in range, not codec codes. meta.json says so.
    The GPU pilot refuses a store marked placeholder.
    """
    tokens = []
    index = []
    offset = 0
    rng = np.random.default_rng(0)
    audio_s = 0.0
    for clip in clips:
        dur = float(clip.get("duration_s") or 1.0)
        n = max(1, int(round(dur * FPS)))
        frames = rng.integers(0, 2048, size=(n, width), dtype=np.int64)
        tokens.append(frames.astype(np.uint16))
        index.append((offset, n))
        offset += n
        audio_s += dur
    speed = _write_store(out, tokens, index, width, audio_s, 0.0, "placeholder")
    meta = json.loads((out / "meta.json").read_text())
    meta["placeholder_tokens"] = True
    (out / "meta.json").write_text(json.dumps(meta) + "\n")
    speed["placeholder_tokens"] = True
    (out / "speed.json").write_text(json.dumps(speed, indent=2) + "\n")
    print("WARNING: placeholder tokens, not Mimi codes", flush=True)
    return speed


def _verify(out: Path) -> None:
    meta = json.loads((out / "meta.json").read_text())
    idx = np.load(out / "index.npy")
    tok = np.memmap(out / "tokens.bin", dtype=np.uint16, mode="r", shape=(meta["frames"], meta["width"]))
    if idx.ndim != 2 or idx.shape[1] != 2:
        raise SystemExit("index.npy must be [clips, 2]")
    if int(idx[-1, 0] + idx[-1, 1]) > meta["frames"]:
        raise SystemExit("index points past tokens.bin")
    _ = tok[0, 0]
    print(f"store ok {out} frames={meta['frames']} width={meta['width']} clips={len(idx)}", flush=True)


def _merge(parts: list[Path], dest: Path, width: int) -> dict:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    tokens = []
    index = []
    offset = 0
    audio_s = 0.0
    encode_s = 0.0
    for part in parts:
        meta = json.loads((part / "meta.json").read_text())
        if meta.get("placeholder_tokens"):
            raise SystemExit("refusing to merge placeholder tokens")
        arr = np.fromfile(part / "tokens.bin", dtype=np.uint16).reshape(-1, width)
        idx = np.load(part / "index.npy").copy()
        idx[:, 0] += offset
        tokens.append(arr)
        index.append(idx)
        offset += arr.shape[0]
        speed = json.loads((part / "speed.json").read_text())
        audio_s += float(speed.get("audio_s") or 0)
        encode_s = max(encode_s, float(speed.get("encode_s") or 0))
    # Wall time of a parallel pair is the slower worker. Audio is the sum.
    return _write_store(dest, tokens, [tuple(map(int, row)) for idx in index for row in idx], width, audio_s, encode_s, "cuda")


def encode_maybe_parallel(clips, out, width, chunk_s, deadline_epoch, reserve_s) -> dict:
    import torch

    gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    force = os.environ.get("PILOT_TOKENIZE_FORCE_PARALLEL") == "1"
    out.mkdir(parents=True, exist_ok=True)
    if not force and (gpus < 2 or len(clips) < 4 or os.environ.get("PILOT_TOKENIZE_ONE_GPU")):
        device = "cuda" if gpus else "cpu"
        return encode_clips(
            clips, out, width, device=device, chunk_s=chunk_s,
            deadline_epoch=deadline_epoch, reserve_s=reserve_s,
        )
    mid = len(clips) // 2
    parts = []
    procs = []
    for i, subset in enumerate((clips[:mid], clips[mid:])):
        part = out.parent / f".{out.name}-gpu{i}"
        clip_path = part / "clips.jsonl"
        part.mkdir(parents=True, exist_ok=True)
        with clip_path.open("w", encoding="utf-8") as fh:
            for row in subset:
                fh.write(json.dumps(row) + "\n")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        device = "cuda" if gpus else "cpu"
        if gpus:
            env["CUDA_VISIBLE_DEVICES"] = str(i)
        cmd = [
            sys.executable, str(Path(__file__).resolve()),
            "--clips", str(clip_path), "--out", str(part), "--width", str(width),
            "--chunk-s", str(chunk_s), "--device", device,
        ]
        if deadline_epoch:
            cmd += ["--deadline-epoch", str(deadline_epoch), "--reserve-s", str(reserve_s)]
        print(f"tokenize worker gpu {i}: {len(subset)} clips", flush=True)
        procs.append(subprocess.Popen(cmd, env=env))
        parts.append(part)
    rc = 0
    for proc in procs:
        rc = proc.wait() or rc
    if rc != 0:
        raise SystemExit(f"tokenize worker failed ({rc})")
    return _merge(parts, out, width)


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clips", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--width", type=int, choices=(8, 16))
    p.add_argument("--chunk-s", type=float, default=0.0, help="0 picks 30s on CPU and 120s on GPU")
    p.add_argument("--device", default="")
    p.add_argument("--deadline-epoch", type=int, default=0)
    p.add_argument("--reserve-s", type=int, default=0)
    p.add_argument("--placeholder", action="store_true")
    p.add_argument("--parallel", action="store_true", help="use both GPUs when present")
    p.add_argument("--self-test", action="store_true", help="CPU merge regression, no model and no pod")
    args = p.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if not args.clips or not args.out or args.width is None:
        raise SystemExit("--clips, --out, and --width are required")
    clips = _load_clips(args.clips)
    if not clips:
        raise SystemExit(f"no clips in {args.clips}")
    if args.placeholder:
        write_placeholder(clips, args.out, args.width)
        return 0
    chunk = args.chunk_s
    if chunk <= 0:
        try:
            import torch
            chunk = 120.0 if torch.cuda.is_available() else 30.0
        except Exception:
            chunk = 30.0
    if args.parallel:
        encode_maybe_parallel(
            clips, args.out, args.width, chunk, args.deadline_epoch, args.reserve_s,
        )
        return 0
    device = args.device
    if not device:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    encode_clips(
        clips, args.out, args.width, device=device, chunk_s=chunk,
        deadline_epoch=args.deadline_epoch, reserve_s=args.reserve_s,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
