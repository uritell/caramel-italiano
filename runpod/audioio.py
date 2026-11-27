"""Read, resample, and write audio. soundfile first, ffmpeg if a file is mp3."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 24000


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    """Return float32 audio as [samples, channels] and the sample rate."""
    path = Path(path)
    try:
        audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return np.ascontiguousarray(audio), int(sr)
    except Exception:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "clip.wav"
            proc = subprocess.run(
                [ffmpeg, "-y", "-v", "error", "-i", str(path), "-ac", "2", str(wav)],
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"ffmpeg could not read {path}: {proc.stderr[-400:]}")
            audio, sr = sf.read(str(wav), dtype="float32", always_2d=True)
            return np.ascontiguousarray(audio), int(sr)


def resample(audio: np.ndarray, sr: int, target: int = SR) -> np.ndarray:
    if sr == target:
        return np.ascontiguousarray(audio, dtype=np.float32)
    try:
        import soxr

        out = soxr.resample(audio, sr, target, quality="HQ")
        return np.ascontiguousarray(out, dtype=np.float32)
    except Exception:
        n = max(1, int(round(audio.shape[0] * target / sr)))
        old = np.linspace(0.0, 1.0, audio.shape[0], endpoint=False)
        new = np.linspace(0.0, 1.0, n, endpoint=False)
        if audio.ndim == 1 or audio.shape[1] == 1:
            col = np.interp(new, old, np.asarray(audio).reshape(-1))
            return col.astype(np.float32).reshape(-1, 1)
        cols = [np.interp(new, old, audio[:, i]) for i in range(audio.shape[1])]
        return np.stack(cols, axis=1).astype(np.float32)


def to_mono(audio: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(audio.mean(axis=1), dtype=np.float32)


def to_stereo(audio: np.ndarray) -> np.ndarray:
    if audio.shape[1] == 1:
        return np.repeat(audio, 2, axis=1)
    return np.ascontiguousarray(audio[:, :2], dtype=np.float32)


def write_audio(path: Path, audio: np.ndarray, sr: int = SR) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio, -1.0, 1.0)
    if path.suffix.lower() == ".flac":
        sf.write(str(path), clipped, sr, format="FLAC", subtype="PCM_16")
    else:
        sf.write(str(path), clipped, sr, subtype="PCM_16")


def duration_s(path: Path) -> float:
    info = sf.info(str(path))
    if info.samplerate <= 0:
        return 0.0
    return float(info.frames) / float(info.samplerate)
