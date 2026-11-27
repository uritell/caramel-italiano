#!/usr/bin/env python3
"""Encode mono or stereo audio with Kyutai Mimi into a caramel TokenStore.

Matches third_party/caramel/generate.py:

- MimiModel.from_pretrained("kyutai/mimi")
- audio at 24 kHz, resampled with the same bicubic interpolate as generate.py
- model.encode(..., num_quantizers=8)
- stereo is one speaker per channel, passed as a batch of two mono waves
  (generate.py's ``x[:2, None]``), not as a 2-channel mix

Matches third_party/caramel/train.py TokenStore:

- tokens.bin  uint16  [frames, 8] mono or [frames, 16] stereo
  stereo column order is speaker 1 codebooks 0-7, then speaker 2 codebooks 0-7
  (left / caller, then right / callee)
- index.npy   int64   [clips, 2] of (offset, length) in frames
- meta.json   {"width": W, "frames": total}

Long files are encoded in chunks with left context, because Mimi's
transformer looks back a sliding window (20 s at 12.5 Hz). --check-chunking
compares a chunked encode to a one-shot encode on a short waveform.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))

os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))
os.environ.setdefault("HF_HUB_CACHE", str(ROOT / ".hf-cache" / "hub"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from audio import TARGET_SR, read_audio, resample_mono_like_generate  # noqa: E402

SR = TARGET_SR
N_Q = 8
FRAME_SAMPLES = int(round(SR / 12.5))  # 1920


def load_mimi(device: str = "cpu"):
    import torch
    from transformers import MimiModel

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    print("loading kyutai/mimi", flush=True)
    t0 = time.perf_counter()
    model = MimiModel.from_pretrained("kyutai/mimi")
    model.eval()
    model.to(device)
    print(f"mimi ready in {time.perf_counter() - t0:.1f}s on {device}", flush=True)
    return model


def _encode_batch(model, waves: list[np.ndarray]):
    import torch

    wav = np.stack([np.ascontiguousarray(w, dtype=np.float32) for w in waves], axis=0)
    tensor = torch.from_numpy(wav)[:, None, :]  # [N, 1, T], generate.py's layout
    with torch.inference_mode():
        out = model.encode(tensor, num_quantizers=N_Q)
    codes = out.audio_codes.detach().cpu().numpy()
    return codes


def encode_waves(
    model,
    waves: list[np.ndarray],
    chunk_s: float = 60.0,
    context_s: float = 60.0,
    force_chunk: bool = False,
) -> np.ndarray:
    """Return int64 codes [n_channels, 8, frames] for equal-length mono waves.

    Up to 90 seconds is encoded in one call. Longer audio is cut into
    ``chunk_s`` of new audio with ``context_s`` of left context. Mimi's
    early convolutions keep an activation per sample, so a half-hour
    stereo clip asks for tens of gigabytes on CPU; 90 seconds stays
    near 1 GB. A short context does not reproduce a one-shot encode.
    With 60 s of context the leftover token differences on a test clip
    are about one percent, mostly in the higher codebooks.
    ``force_chunk`` is for that check.
    """
    n = max(len(w) for w in waves)
    padded = [np.pad(w, (0, n - len(w))) for w in waves]
    extra = (FRAME_SAMPLES - (n % FRAME_SAMPLES)) % FRAME_SAMPLES
    if extra:
        padded = [np.pad(w, (0, extra)) for w in padded]
        n = len(padded[0])
    total_in_frames = n // FRAME_SAMPLES
    # 90 s of stereo is about 1 GB of conv activation on this CPU build.
    # A 30 minute clip asked for a 20 GB tensor and was killed.
    if not force_chunk and n <= int(90 * SR):
        return _encode_batch(model, padded)
    ctx_frames = int(round(context_s * 12.5))
    hop_frames = int(round(chunk_s * 12.5))
    parts = []
    start = 0
    while start < total_in_frames:
        end = min(total_in_frames, start + hop_frames)
        ctx = min(ctx_frames, start)
        a0 = (start - ctx) * FRAME_SAMPLES
        a1 = end * FRAME_SAMPLES
        codes = _encode_batch(model, [w[a0:a1] for w in padded])
        need = end - start
        if codes.shape[-1] >= ctx + need:
            take = codes[:, :, ctx : ctx + need]
        elif codes.shape[-1] >= need:
            take = codes[:, :, -need:]
        else:
            raise RuntimeError(
                f"Mimi returned {codes.shape[-1]} frames for {need} content frames "
                f"(context {ctx})"
            )
        parts.append(take)
        start = end
    return np.concatenate(parts, axis=-1)


def codes_to_tokens(codes: np.ndarray) -> np.ndarray:
    """[channels, 8, frames] -> uint16 [frames, channels*8]."""
    if codes.ndim != 3 or codes.shape[1] != N_Q:
        raise RuntimeError(f"expected [C, 8, T], got {codes.shape}")
    flat = np.transpose(codes, (2, 0, 1)).reshape(codes.shape[2], codes.shape[0] * N_Q)
    if flat.min() < 0 or flat.max() >= 65536:
        raise RuntimeError(f"code range {flat.min()}..{flat.max()} does not fit uint16")
    return np.ascontiguousarray(flat.astype(np.uint16))


def write_store(out_dir: Path, rows: list[np.ndarray], width: int) -> dict:
    if not rows:
        raise RuntimeError("no clips to write")
    total = sum(int(r.shape[0]) for r in rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    mm = np.memmap(out_dir / "tokens.bin", dtype=np.uint16, mode="w+", shape=(total, width))
    index = np.zeros((len(rows), 2), dtype=np.int64)
    offset = 0
    for i, row in enumerate(rows):
        if row.ndim != 2 or row.shape[1] != width:
            raise RuntimeError(f"clip {i} shape {row.shape}, width {width}")
        mm[offset : offset + row.shape[0]] = row
        index[i, 0] = offset
        index[i, 1] = row.shape[0]
        offset += row.shape[0]
    mm.flush()
    del mm
    np.save(out_dir / "index.npy", index)
    meta = {"width": int(width), "frames": int(total)}
    (out_dir / "meta.json").write_text(json.dumps(meta) + "\n", encoding="utf-8")
    return meta


def _mono_24k(path: Path, max_seconds: float | None) -> list[np.ndarray]:
    audio, sr = read_audio(path)
    if max_seconds is not None:
        n = int(max_seconds * sr)
        audio = audio[:n]
    if audio.shape[1] == 1:
        channels = [audio[:, 0]]
    else:
        channels = [audio[:, i] for i in range(audio.shape[1])]
    return [resample_mono_like_generate(c, sr, SR) for c in channels]


def _iter_items(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def check_chunking(model) -> dict:
    rng = np.random.default_rng(0)
    seconds = 150.0
    wave = (0.05 * rng.standard_normal(int(seconds * SR))).astype(np.float32)
    # One shot.
    t0 = time.perf_counter()
    full = _encode_batch(model, [wave])
    one_shot_s = time.perf_counter() - t0
    # Force the chunk path used for files longer than 40 minutes.
    t1 = time.perf_counter()
    chunked = encode_waves(model, [wave], chunk_s=60.0, context_s=60.0, force_chunk=True)
    chunk_s = time.perf_counter() - t1
    n = min(full.shape[-1], chunked.shape[-1])
    mismatch = int(np.sum(full[0, :, :n] != chunked[0, :, :n]))
    total = int(n * full.shape[1])
    return {
        "seconds": seconds,
        "one_shot_frames": int(full.shape[-1]),
        "chunked_frames": int(chunked.shape[-1]),
        "compared_frames": n,
        "mismatched_tokens": mismatch,
        "mismatch_fraction": mismatch / total if total else 1.0,
        "one_shot_wall_s": one_shot_s,
        "chunked_wall_s": chunk_s,
        "one_shot_rtf": one_shot_s / seconds,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--items", type=Path, help="jsonl with audio, mode (mono|stereo), optional max_seconds, id")
    p.add_argument("--out", type=Path, help="TokenStore directory to create")
    p.add_argument("--width", type=int, choices=(8, 16))
    p.add_argument("--check-chunking", action="store_true")
    p.add_argument("--benchmark-seconds", type=float, default=0.0, help="warmup-excluded encode of this many seconds")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    model = load_mimi(args.device)
    if args.check_chunking:
        report = check_chunking(model)
        print(json.dumps(report, indent=2), flush=True)
        if not args.items:
            # 60 s of left context leaves about a 1% token difference, almost all
            # in the higher codebooks. A jump well past that means the chunk
            # alignment broke.
            return 0 if report["mismatch_fraction"] < 0.02 else 1
    if not args.items or not args.out or args.width is None:
        if args.check_chunking:
            return 0
        p.error("--items, --out and --width are required to write a store")
    items = _iter_items(args.items)
    rows = []
    channel_seconds = 0.0
    encode_wall = 0.0
    manifest = []
    bench = None
    if args.benchmark_seconds > 0 and items:
        waves = _mono_24k(Path(items[0]["audio"]), args.benchmark_seconds)
        if args.width == 8:
            waves = waves[:1]
        else:
            if len(waves) == 1:
                waves = [waves[0], np.zeros_like(waves[0])]
            waves = waves[:2]
        # warmup
        _ = encode_waves(model, [w[: int(2 * SR)] for w in waves], chunk_s=30, context_s=0)
        times = []
        for _ in range(2):
            t0 = time.perf_counter()
            _ = encode_waves(model, waves, chunk_s=60, context_s=0)
            times.append(time.perf_counter() - t0)
        audio_s = len(waves[0]) / SR * len(waves)
        wall = float(np.mean(times))
        bench = {
            "audio_channel_seconds": audio_s,
            "wall_s_mean": wall,
            "rtf": wall / audio_s,
            "repeats": 2,
            "warmup_excluded": True,
        }
        print(f"benchmark RTF {bench['rtf']:.4f} over {audio_s:.1f} channel-seconds", flush=True)
    for item in items:
        mode = item["mode"]
        waves = _mono_24k(Path(item["audio"]), item.get("max_seconds"))
        if mode == "mono":
            if args.width != 8:
                raise RuntimeError("mono items require --width 8")
            waves = [waves[0]]
        elif mode == "stereo":
            if args.width != 16:
                raise RuntimeError("stereo items require --width 16")
            if len(waves) == 1:
                raise RuntimeError(f"{item['audio']} is mono, stereo store needs two channels")
            waves = waves[:2]
            # Equal length after resample (rates match, but rounding can differ by 1).
            n = min(len(waves[0]), len(waves[1]))
            waves = [waves[0][:n], waves[1][:n]]
        else:
            raise RuntimeError(f"unknown mode {mode}")
        t0 = time.perf_counter()
        codes = encode_waves(model, waves)
        dt = time.perf_counter() - t0
        tokens = codes_to_tokens(codes)
        if tokens.shape[1] != args.width:
            raise RuntimeError(f"got width {tokens.shape[1]}, expected {args.width}")
        rows.append(tokens)
        ch_s = sum(len(w) for w in waves) / SR
        channel_seconds += ch_s
        encode_wall += dt
        manifest.append(
            {
                "id": item.get("id"),
                "audio": item["audio"],
                "mode": mode,
                "frames": int(tokens.shape[0]),
                "channel_seconds": ch_s,
                "encode_wall_s": dt,
                "rtf": dt / ch_s if ch_s else None,
            }
        )
        print(
            f"encoded {item.get('id') or item['audio']}: {tokens.shape[0]} frames, "
            f"RTF {dt / ch_s:.4f}",
            flush=True,
        )
    meta = write_store(args.out, rows, args.width)
    report = {
        "store": str(args.out),
        "meta": meta,
        "clips": len(rows),
        "channel_seconds": channel_seconds,
        "encode_wall_s": encode_wall,
        "rtf": encode_wall / channel_seconds if channel_seconds else None,
        "benchmark": bench,
        "items": manifest,
    }
    (args.out / "encode_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {args.out} width {meta['width']} frames {meta['frames']} "
        f"RTF {report['rtf']:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
