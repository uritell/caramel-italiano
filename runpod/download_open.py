#!/usr/bin/env python3
"""Download the open Italian pilot audio. No accounts.

Included: Multilingual LibriSpeech Italian (OpenSLR 94), FLEURS Italian,
and the DataHive Italian Conversations sample. All three are CC BY 4.0.

Common Voice is refused. It needs a Mozilla Data Collective account, and
this pilot does not create one.

A byte cap and an audio-seconds cap let a local smoke test take a couple
of minutes of audio without pulling the 15 GB MLS archive.
"""
from __future__ import annotations

import argparse
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

UA = "caramel-italiano-pilot/1.0"
AUDIO_EXT = {".flac", ".wav", ".mp3", ".ogg", ".opus"}

MLS_URL = "https://dl.fbaipublicfiles.com/mls/mls_italian.tar.gz"
FLEURS_SPLITS = ("train", "dev", "test")
DATAHIVE_REPO = "https://huggingface.co/datasets/datahiveai/italian-conversations-sample/resolve/main/preview"

ATTRIBUTION = {
    "mls_it": (
        "Multilingual LibriSpeech Italian (Pratap et al., 2020), CC BY 4.0, "
        "https://www.openslr.org/94/. Derived from public-domain LibriVox and Project Gutenberg."
    ),
    "fleurs_it": (
        "FLEURS Italian (Conneau et al., 2022), CC BY 4.0, "
        "https://huggingface.co/datasets/google/fleurs. A product of Google."
    ),
    "datahive": (
        "Italian Conversations sample by DataHive, CC BY 4.0, "
        "https://huggingface.co/datasets/datahiveai/italian-conversations-sample. "
        "Caller is left, callee is right."
    ),
}

# Public URLs a dry run can HEAD. None of these need a login.
HEAD_URLS = [
    MLS_URL,
    "https://huggingface.co/datasets/google/fleurs/resolve/main/data/it_it/train.tsv",
    "https://huggingface.co/datasets/google/fleurs/resolve/main/data/it_it/audio/train.tar.gz",
    "https://huggingface.co/datasets/google/fleurs/resolve/main/data/it_it/dev.tsv",
    "https://huggingface.co/datasets/google/fleurs/resolve/main/data/it_it/audio/dev.tar.gz",
    "https://huggingface.co/datasets/google/fleurs/resolve/main/data/it_it/test.tsv",
    "https://huggingface.co/datasets/google/fleurs/resolve/main/data/it_it/audio/test.tar.gz",
    f"{DATAHIVE_REPO}/metadata.jsonl",
    f"{DATAHIVE_REPO}/it_000002.mp3",
]

BANNED = {"common_voice", "common_voice_it_27", "commonvoice"}


class LimitedReader:
    """File-like that stops a download after `limit` bytes. 0 means no limit."""

    def __init__(self, raw, limit: int):
        self.raw = raw
        self.limit = int(limit)
        self.read_bytes = 0
        self.hit_cap = False

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = 1 << 20
        if self.limit and self.read_bytes >= self.limit:
            self.hit_cap = True
            return b""
        if self.limit:
            size = min(size, self.limit - self.read_bytes)
        chunk = self.raw.read(size)
        self.read_bytes += len(chunk)
        if self.limit and self.read_bytes >= self.limit:
            self.hit_cap = True
        return chunk


def _request(url: str, method: str = "GET"):
    req = urllib.request.Request(url, method=method, headers={"User-Agent": UA})
    return urllib.request.urlopen(req, timeout=120)


def head_urls(urls: list[str] | None = None) -> list[dict]:
    rows = []
    for url in urls or HEAD_URLS:
        row = {"url": url, "ok": False, "status": None, "bytes": None, "error": None}
        try:
            with _request(url, "HEAD") as resp:
                row["status"] = getattr(resp, "status", 200)
                length = resp.headers.get("Content-Length")
                row["bytes"] = int(length) if length and length.isdigit() else None
                row["ok"] = 200 <= int(row["status"]) < 300
        except Exception as exc:  # noqa: BLE001 — report and let the caller decide
            code = getattr(exc, "code", None)
            row["status"] = code
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        flag = "ok" if row["ok"] else "FAIL"
        print(f"HEAD {flag} {row['status']} {url}", flush=True)
    return rows


def _consume(fh) -> None:
    if fh is None:
        return
    while fh.read(1 << 20):
        pass


def _duration(path: Path) -> float:
    try:
        import soundfile as sf

        info = sf.info(str(path))
        if info.samplerate <= 0:
            return 0.0
        return float(info.frames) / float(info.samplerate)
    except Exception:
        return 0.0


def stream_audio(
    url: str,
    dest: Path,
    *,
    source: str,
    split: str,
    max_audio_s: float,
    max_bytes: int,
    speaker_from_name,
) -> list[dict]:
    """Pull audio members until the audio cap or the byte cap."""
    dest.mkdir(parents=True, exist_ok=True)
    print(f"stream {source} {split} {url}", flush=True)
    records = []
    audio_s = 0.0
    with _request(url) as resp:
        limited = LimitedReader(resp, max_bytes)
        try:
            tar = tarfile.open(fileobj=limited, mode="r|gz")
        except tarfile.ReadError as exc:
            raise RuntimeError(f"not a gzip tar: {url}") from exc
        for member in tar:
            if audio_s >= max_audio_s > 0:
                break
            if limited.hit_cap:
                break
            suffix = Path(member.name).suffix.lower()
            if not member.isfile() or suffix not in AUDIO_EXT:
                _consume(tar.extractfile(member) if member.isfile() else None)
                continue
            before = limited.read_bytes
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            data = extracted.read()
            if limited.hit_cap and limited.read_bytes > before and max_bytes:
                # The cap landed inside this member. Drop the partial file.
                print(f"byte cap {max_bytes} reached inside {member.name}", flush=True)
                break
            safe = Path(member.name).name.replace("..", "_")
            path = dest / safe
            if path.exists():
                path = dest / f"{source}_{split}_{safe}"
            path.write_bytes(data)
            dur = _duration(path)
            if dur <= 0:
                path.unlink(missing_ok=True)
                continue
            if max_audio_s > 0 and audio_s + dur > max_audio_s and audio_s > 0:
                path.unlink(missing_ok=True)
                break
            audio_s += dur
            records.append({
                "source": source,
                "licence": "CC BY 4.0",
                "attribution": ATTRIBUTION[source],
                "split": split,
                "path": str(path),
                "duration_s": round(dur, 3),
                "speaker_id": speaker_from_name(member.name),
                "channels": _channels(path),
                "role": "stereo" if source == "datahive" else "mono",
            })
            if limited.read_bytes // (50 << 20) != before // (50 << 20):
                print(f"  {limited.read_bytes / 1e6:.0f} MB, {audio_s:.1f} s audio", flush=True)
        print(
            f"  got {len(records)} files, {audio_s:.1f} s, {limited.read_bytes / 1e6:.1f} MB"
            f"{' (byte cap)' if limited.hit_cap else ''}",
            flush=True,
        )
    return records


def _channels(path: Path) -> int:
    try:
        import soundfile as sf

        return int(sf.info(str(path)).channels)
    except Exception:
        return 0


def _mls_speaker(name: str) -> str:
    parts = Path(name).parts
    if "audio" in parts:
        i = parts.index("audio")
        if i + 1 < len(parts):
            return f"mls_{parts[i + 1]}"
    return "mls_unknown"


def _fleurs_speaker(name: str) -> str:
    # The public TSV has a gender column, not a stable speaker id.
    # Each file is its own voice id so two turns are never the same clip.
    return "fleurs_" + Path(name).stem


def _trim_to(path: Path, seconds: float, dest: Path) -> Path:
    """Write a WAV prefix. Uses ffmpeg when the file is not a WAV soundfile likes."""
    import shutil
    import subprocess

    dest.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        proc = subprocess.run(
            [ffmpeg, "-y", "-v", "error", "-i", str(path), "-t", str(seconds), "-ac", "2", str(dest)],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0 and dest.exists():
            return dest
    from audioio import read_audio, write_audio

    audio, sr = read_audio(path)
    n = min(len(audio), int(seconds * sr))
    write_audio(dest, audio[:n], sr)
    return dest


def download_datahive(dest: Path, max_audio_s: float) -> list[dict]:
    dest.mkdir(parents=True, exist_ok=True)
    meta_path = dest / "metadata.jsonl"
    with _request(f"{DATAHIVE_REPO}/metadata.jsonl") as resp:
        meta_path.write_bytes(resp.read())
    rows = [json.loads(line) for line in meta_path.read_text().splitlines() if line.strip()]
    rows.sort(key=lambda r: float(r.get("duration_seconds") or 0))
    records = []
    audio_s = 0.0
    for row in rows:
        if max_audio_s > 0 and audio_s >= max_audio_s:
            break
        name = row["file_name"]
        raw = dest / name
        if not raw.exists():
            print(f"GET datahive {name}", flush=True)
            with _request(f"{DATAHIVE_REPO}/{name}") as resp:
                raw.write_bytes(resp.read())
        dur = float(row.get("duration_seconds") or 0) or _duration(raw)
        keep = dur
        path = raw
        if max_audio_s > 0:
            room = max_audio_s - audio_s
            if dur > room:
                trimmed = dest / (Path(name).stem + "_trim.wav")
                path = _trim_to(raw, room, trimmed)
                keep = _duration(path)
        audio_s += keep
        records.append({
            "source": "datahive",
            "licence": "CC BY 4.0",
            "attribution": ATTRIBUTION["datahive"],
            "split": "sample",
            "path": str(path),
            "duration_s": round(keep, 3),
            "speaker_id": f"datahive_{Path(name).stem}",
            "channels": 2,
            "role": "stereo",
            "topic": row.get("topic"),
        })
    print(f"datahive {len(records)} files, {audio_s:.1f} s", flush=True)
    return records


def download(
    out: Path,
    *,
    sources: list[str],
    max_audio_s: float,
    max_bytes: int,
    max_mono_s: float = 0.0,
    fleurs_splits: tuple[str, ...] = FLEURS_SPLITS,
) -> list[dict]:
    for name in sources:
        if name in BANNED or "common_voice" in name:
            raise SystemExit(
                "Common Voice is not part of this run. It needs a Mozilla Data "
                "Collective account, and this pilot does not create one."
            )
    out.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    mono_left = max_mono_s

    def mono_cap() -> float:
        if mono_left > 0:
            return mono_left
        return max_audio_s

    if "mls_it" in sources and mono_cap() != 0:
        got = stream_audio(
            MLS_URL, out / "mls",
            source="mls_it", split="italian",
            max_audio_s=mono_cap(), max_bytes=max_bytes,
            speaker_from_name=_mls_speaker,
        )
        records += got
        if mono_left > 0:
            mono_left -= sum(r["duration_s"] for r in got)
    if "fleurs_it" in sources and mono_cap() != 0:
        splits = ("test",) if (max_audio_s > 0 and max_mono_s <= 0) else fleurs_splits
        for split in splits:
            if mono_left < 0:
                break
            url = (
                "https://huggingface.co/datasets/google/fleurs/resolve/main/"
                f"data/it_it/audio/{split}.tar.gz"
            )
            got = stream_audio(
                url, out / "fleurs" / split,
                source="fleurs_it", split=split,
                max_audio_s=mono_cap(), max_bytes=max_bytes,
                speaker_from_name=_fleurs_speaker,
            )
            records += got
            if mono_left > 0:
                mono_left -= sum(r["duration_s"] for r in got)
    if "datahive" in sources:
        records += download_datahive(out / "datahive", max_audio_s)
    manifest = out / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as fh:
        for row in records:
            fh.write(json.dumps(row) + "\n")
    (out / "ATTRIBUTION.txt").write_text(
        "\n".join(ATTRIBUTION[k] for k in ("mls_it", "fleurs_it", "datahive")) + "\n"
    )
    print(f"manifest {manifest} ({len(records)} clips)", flush=True)
    return records


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, help="required unless --head")
    p.add_argument("--source", action="append", default=None,
                   choices=["mls_it", "fleurs_it", "datahive"])
    p.add_argument("--max-audio-s", type=float, default=0.0,
                   help="stop each source after this many seconds; 0 downloads the source")
    p.add_argument("--max-bytes", type=int, default=0,
                   help="stop a tar stream after this many bytes; 0 means the whole file")
    p.add_argument("--max-mono-hours", type=float, default=0.0,
                   help="stop MLS+FLEURS once this many hours are saved; DataHive is always complete")
    p.add_argument("--head", action="store_true", help="HEAD the public URLs and exit")
    args = p.parse_args(argv)
    if args.head:
        rows = head_urls()
        bad = [r for r in rows if r["status"] in (404, 410)]
        return 1 if bad else 0
    if args.out is None:
        print("download_open.py: --out is required", file=sys.stderr)
        return 2
    download(
        args.out,
        sources=args.source or ["mls_it", "fleurs_it", "datahive"],
        max_audio_s=args.max_audio_s,
        max_bytes=args.max_bytes,
        max_mono_s=args.max_mono_hours * 3600.0,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
