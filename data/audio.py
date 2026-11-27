"""Read speech audio and resample it the way caramel's generate.py does.

generate.py reads with soundfile and, when the rate is not 24 kHz, resamples
with torch.nn.functional.interpolate (bicubic, antialias). Stereo is resampled
here one channel at a time, using that same mono call, so the two tracks do
not bleed into each other. Mimi itself is invoked in tokenize/, not here.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

TARGET_SR = 24000


def probe(path: Path) -> dict:
    """Duration, sample rate and channels via ffprobe (works for mp3)."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate,channels,duration",
        "-of",
        "json",
        str(path),
    ]
    raw = subprocess.check_output(cmd)
    stream = json.loads(raw)["streams"][0]
    duration = stream.get("duration")
    return {
        "sample_rate": int(stream["sample_rate"]),
        "channels": int(stream["channels"]),
        "duration_s": float(duration) if duration not in (None, "N/A") else None,
    }


def _ffmpeg_decode(path: Path, sample_rate: int | None, channels: int) -> tuple[np.ndarray, int]:
    info = probe(path)
    sr = int(sample_rate or info["sample_rate"])
    ch = int(channels)
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-ar",
        str(sr),
        "-ac",
        str(ch),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-",
    ]
    raw = subprocess.check_output(cmd)
    usable = len(raw) - (len(raw) % (4 * ch))
    x = np.frombuffer(raw[:usable], dtype=np.float32).reshape(-1, ch).copy()
    return x, sr


def read_audio(path: Path, sample_rate: int | None = None) -> tuple[np.ndarray, int]:
    """Return float32 audio shaped [samples, channels], and the sample rate.

    wav and flac go through soundfile, matching generate.py. mp3 (the DataHive
    sample) goes through ffmpeg because this environment's soundfile build
    does not reliably decode it. Pass sample_rate to decode straight to that
    rate (ffmpeg path only); None keeps the file's own rate.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".wav", ".flac", ".ogg"} and sample_rate is None:
        import soundfile as sf

        audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return np.ascontiguousarray(audio, dtype=np.float32), int(sr)
    if suffix in {".wav", ".flac", ".ogg"} and sample_rate is not None:
        audio, sr = read_audio(path, sample_rate=None)
        if sr == sample_rate:
            return audio, sr
        columns = [
            resample_mono_like_generate(audio[:, i], sr, sample_rate) for i in range(audio.shape[1])
        ]
        n = min(len(c) for c in columns)
        return np.stack([c[:n] for c in columns], axis=1), int(sample_rate)
    channels = probe(path)["channels"] if sample_rate is None else probe(path)["channels"]
    return _ffmpeg_decode(path, sample_rate, channels)


def header_info(path: Path) -> dict:
    """Duration and format from the file header, without decoding samples."""
    path = Path(path)
    if path.suffix.lower() in {".wav", ".flac", ".ogg"}:
        import soundfile as sf

        info = sf.info(str(path))
        return {
            "sample_rate": int(info.samplerate),
            "channels": int(info.channels),
            "duration_s": float(info.duration),
        }
    return probe(path)


def _resample_block(x: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """One block, the mono branch of generate.py's resample."""
    import torch
    import torch.nn.functional as F

    wave = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)[None, :])
    new_t = max(1, int(round(wave.shape[1] * target_sr / sr)))
    with torch.inference_mode():
        y = F.interpolate(
            wave[None, None],
            size=(1, new_t),
            mode="bicubic",
            antialias=True,
        )[0, 0]
    return np.ascontiguousarray(y[0].detach().cpu().numpy(), dtype=np.float32)


def resample_mono_like_generate(x: np.ndarray, sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    """Resample one mono channel the way generate.encode resamples a mono file.

    generate.py does::

        x = torch.from_numpy(waveform[None, :])          # [channels=1, time]
        y = F.interpolate(x[None, None], size=(1, new_t),
                          mode="bicubic", antialias=True)[0, 0]

    which is ``_resample_block``. A whole DataHive conversation at 48 kHz
    makes that call allocate far more than the waveform (bicubic antialias
    on tens of millions of samples), enough to be killed on a 15 GB
    machine. Blocks of 30 s with 1 s of overlap, discarding the overlap,
    use the same filter without holding the whole file as one tensor.
    Channels of a stereo file are resampled separately by the caller so
    the filter cannot mix left and right.
    """
    x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
    if sr == target_sr or x.size == 0:
        return x
    chunk = int(sr * 30)
    overlap = int(sr)  # 1 s, longer than the antialias kernel
    if x.size <= chunk + overlap:
        return _resample_block(x, sr, target_sr)
    pieces = []
    pos = 0
    scale = target_sr / sr
    while pos < x.size:
        end = min(x.size, pos + chunk)
        left = max(0, pos - overlap) if pos else 0
        right = min(x.size, end + (overlap if end < x.size else 0))
        block = _resample_block(x[left:right], sr, target_sr)
        keep_from = int(round((pos - left) * scale))
        keep_to = int(round((end - left) * scale))
        keep_from = min(max(keep_from, 0), block.size)
        keep_to = min(max(keep_to, keep_from), block.size)
        pieces.append(block[keep_from:keep_to])
        pos = end
    return np.concatenate(pieces) if pieces else np.zeros(0, np.float32)
