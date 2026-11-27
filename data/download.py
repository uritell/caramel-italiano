#!/usr/bin/env python3
"""Download the openly available clean-pilot Italian speech sources.

No account, no Common Voice, no paid host. Audio is written under data/raw/
(gitignored). Re-running skips files that are already complete.

Sources:

- datahiveai/italian-conversations-sample (CC BY 4.0, stereo)
- google/fleurs config it_it (CC BY 4.0)
- Multilingual LibriSpeech Italian, the public tarball on
  dl.fbaipublicfiles.com (CC BY 4.0), md5 from the published md5sum.txt

Common Voice is refused: it needs a Mozilla Data Collective account.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from sources import (  # noqa: E402
    DATAHIVE_REPO,
    FLEURS_REPO,
    MLS_BYTES,
    MLS_MD5,
    MLS_URL,
    OPEN_SOURCES,
    SOURCES,
)

RAW = ROOT / "raw"
DATAHIVE_FILES = (
    "preview/metadata.jsonl",
    "preview/it_000001.mp3",
    "preview/it_000002.mp3",
    "preview/it_000003.mp3",
    "preview/it_000004.mp3",
    "preview/it_000005.mp3",
    "preview/it_000006.mp3",
)
FLEURS_FILES = (
    "data/it_it/train.tsv",
    "data/it_it/dev.tsv",
    "data/it_it/test.tsv",
    "data/it_it/audio/train.tar.gz",
    "data/it_it/audio/dev.tar.gz",
    "data/it_it/audio/test.tar.gz",
)


def _hf_download(repo: str, filename: str, dest: Path) -> Path:
    from huggingface_hub import hf_hub_download

    dest.mkdir(parents=True, exist_ok=True)
    print(f"fetch {repo} {filename}", flush=True)
    return Path(
        hf_hub_download(
            repo_id=repo,
            filename=filename,
            repo_type="dataset",
            local_dir=str(dest),
        )
    )


def download_datahive(raw: Path) -> None:
    dest = raw / "datahive_it_conversations_sample"
    for name in DATAHIVE_FILES:
        path = dest / name
        if path.is_file() and path.stat().st_size > 0:
            print(f"already present {path}", flush=True)
            continue
        _hf_download(DATAHIVE_REPO, name, dest)


def download_fleurs(raw: Path) -> None:
    dest = raw / "fleurs_it_it"
    for name in FLEURS_FILES:
        path = dest / name
        if path.is_file() and path.stat().st_size > 0:
            print(f"already present {path}", flush=True)
            continue
        _hf_download(FLEURS_REPO, name, dest)
    extract_fleurs(dest)


def extract_fleurs(dest: Path) -> None:
    for split in ("train", "dev", "test"):
        tsv = dest / "data" / "it_it" / f"{split}.tsv"
        archive = dest / "data" / "it_it" / "audio" / f"{split}.tar.gz"
        out = dest / "audio" / split
        n_expected = sum(1 for line in tsv.read_text(encoding="utf-8").splitlines() if line.strip())
        n_have = len(list(out.glob("*.wav"))) if out.is_dir() else 0
        if n_have >= n_expected:
            print(f"fleurs {split}: {n_have} wavs already extracted", flush=True)
            continue
        if not archive.is_file():
            raise FileNotFoundError(archive)
        out.mkdir(parents=True, exist_ok=True)
        print(f"extracting fleurs {split} ({n_expected} clips)", flush=True)
        with tarfile.open(archive, "r:gz") as tf:
            for member in tf:
                if not member.isfile() or not member.name.endswith(".wav"):
                    continue
                name = Path(member.name).name
                target = (out / name).resolve()
                if not str(target).startswith(str(out.resolve())):
                    raise RuntimeError(f"refusing path in fleurs archive: {member.name}")
                if target.is_file() and target.stat().st_size > 0:
                    continue
                src = tf.extractfile(member)
                if src is None:
                    continue
                target.write_bytes(src.read())
        n_have = len(list(out.glob("*.wav")))
        if n_have < n_expected:
            raise RuntimeError(f"fleurs {split}: extracted {n_have}, tsv lists {n_expected}")
        print(f"fleurs {split}: extracted {n_have} wavs", flush=True)


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_mls(raw: Path) -> None:
    dest = raw / "mls_it"
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / "mls_italian.tar.gz"
    if not archive.is_file() or archive.stat().st_size != MLS_BYTES:
        print(f"downloading {MLS_URL}", flush=True)
        print(f"  -> {archive} ({MLS_BYTES / 1e9:.2f} GB)", flush=True)
        cmd = [
            "curl",
            "-L",
            "--fail",
            "--retry",
            "5",
            "--retry-delay",
            "2",
            "-C",
            "-",
            "-o",
            str(archive),
            MLS_URL,
        ]
        subprocess.run(cmd, check=True)
    size = archive.stat().st_size
    if size != MLS_BYTES:
        raise RuntimeError(f"MLS archive is {size} bytes, expected {MLS_BYTES}. Leave it and re-run to resume.")
    sidecar = Path(str(archive) + ".md5")
    recorded = sidecar.read_text(encoding="utf-8") if sidecar.is_file() else ""
    if MLS_MD5 in recorded:
        print(f"MLS md5 already recorded ({MLS_MD5})", flush=True)
    else:
        print("checking MLS md5 (published dc77f5805aecc7182aa20786032e5dc1)", flush=True)
        got = _md5(archive)
        if got != MLS_MD5:
            raise RuntimeError(f"MLS md5 {got} does not match published {MLS_MD5}")
        sidecar.write_text(f"{got}  {archive.name}\n", encoding="utf-8")
        print(f"MLS md5 ok {got}", flush=True)
    extract_mls(dest, archive)


def extract_mls(dest: Path, archive: Path) -> None:
    marker = dest / "extracted" / "mls_italian" / "train" / "transcripts.txt"
    # Some releases unpack as mls_italian/, others with the language folder
    # already at the top. Accept either once train transcripts exist.
    if marker.is_file() or any((dest / "extracted").glob("**/train/transcripts.txt")):
        print("MLS already extracted", flush=True)
        return
    out = dest / "extracted"
    out.mkdir(parents=True, exist_ok=True)
    print(f"extracting {archive.name} (this is the slow part)", flush=True)
    subprocess.run(["tar", "-xzf", str(archive), "-C", str(out)], check=True)
    found = list((dest / "extracted").glob("**/train/transcripts.txt"))
    if not found:
        raise RuntimeError("MLS extract finished but train/transcripts.txt was not found")
    print(f"MLS extracted, transcripts at {found[0]}", flush=True)


def refuse_common_voice() -> int:
    print(
        "Refusing common_voice_it_27. Common Voice Scripted Speech has to be "
        "obtained from Mozilla Data Collective, which needs an account, and "
        "this pilot step does not create one. The other three sources are "
        "open downloads. See DATA_LICENSES.md.",
        file=sys.stderr,
    )
    return 2


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=RAW, help="raw download directory")
    p.add_argument(
        "--source",
        choices=tuple(SOURCES),
        action="append",
        help="source id to fetch (repeatable). Default: the three open sources.",
    )
    args = p.parse_args()
    wanted = args.source or list(OPEN_SOURCES)
    if "common_voice_it_27" in wanted:
        return refuse_common_voice()
    raw = args.out
    raw.mkdir(parents=True, exist_ok=True)
    for source in wanted:
        print(f"== {source}: {SOURCES[source]['licence']} {SOURCES[source]['url']}", flush=True)
        if source == "datahive_it_conversations_sample":
            download_datahive(raw)
        elif source == "fleurs_it_it":
            download_fleurs(raw)
        elif source == "mls_it":
            download_mls(raw)
        else:
            raise AssertionError(source)
    print("download finished", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
