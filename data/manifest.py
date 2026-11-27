#!/usr/bin/env python3
"""Build a clip-level manifest from data/raw.

Each JSONL row records the path, the licence, the attribution sentence,
the split, the duration measured from the file header, and the speaker id
when the source actually publishes one. FLEURS publishes gender and not a
speaker id; the speaker_id field is null there on purpose. Common Voice is
not scanned.

Writes data/work/manifest.jsonl and data/work/hours.json (both gitignored).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from audio import header_info  # noqa: E402
from sources import SOURCES  # noqa: E402

DEFAULT_OUT = ROOT / "work" / "manifest.jsonl"


def _row(source_id: str, **fields) -> dict:
    meta = SOURCES[source_id]
    base = {
        "source": source_id,
        "licence": meta["licence"],
        "licence_url": meta["licence_url"],
        "attribution": meta["attribution"],
        "source_url": meta["url"],
        "speaker_id": None,
        "gender": None,
        "text": None,
    }
    base.update(fields)
    return base


def _fleurs(raw: Path) -> list[dict]:
    root = raw / "fleurs_it_it"
    rows = []
    for split in ("train", "dev", "test"):
        tsv = root / "data" / "it_it" / f"{split}.tsv"
        audio_dir = root / "audio" / split
        if not tsv.is_file():
            print(f"skip fleurs {split}: no {tsv}", flush=True)
            continue
        for line in tsv.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 7:
                raise ValueError(f"unexpected fleurs tsv row ({len(parts)} columns)")
            filename = parts[1]
            wav = audio_dir / filename
            if not wav.is_file():
                raise FileNotFoundError(wav)
            info = header_info(wav)
            rows.append(
                _row(
                    "fleurs_it_it",
                    id=f"fleurs_it_it/{split}/{Path(filename).stem}",
                    split=split,
                    path=str(wav.relative_to(raw)),
                    duration_s=info["duration_s"],
                    sample_rate=info["sample_rate"],
                    channels=info["channels"],
                    layout="mono",
                    gender=parts[6],
                    text=parts[3],
                    extra={
                        "sentence_id": parts[0],
                        "tsv_num_samples_16k": int(parts[5]),
                    },
                )
            )
    return rows


def _datahive(raw: Path) -> list[dict]:
    root = raw / "datahive_it_conversations_sample" / "preview"
    meta_path = root / "metadata.jsonl"
    if not meta_path.is_file():
        print(f"skip datahive: no {meta_path}", flush=True)
        return []
    rows = []
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        audio = root / rec["file_name"]
        if not audio.is_file():
            raise FileNotFoundError(audio)
        info = header_info(audio)
        rows.append(
            _row(
                "datahive_it_conversations_sample",
                id=f"datahive_it_conversations_sample/{Path(rec['file_name']).stem}",
                split="sample",
                path=str(audio.relative_to(raw)),
                duration_s=info["duration_s"],
                sample_rate=info["sample_rate"],
                channels=info["channels"],
                layout="stereo_caller_left_callee_right",
                extra={
                    "topic": rec.get("topic"),
                    "card_duration_seconds": rec.get("duration_seconds"),
                    "card_turns_per_minute": rec.get("turns_per_minute"),
                    "card_turn_taking_gap_ms": rec.get("turn_taking_gap_ms"),
                    "card_overlap_percent": rec.get("overlap_percent"),
                    "card_speech_balance_percent": rec.get("speech_balance_percent"),
                    "channel_map": "caller left, callee right, shared timeline",
                },
            )
        )
    return rows


def _load_transcripts(split_dir: Path) -> dict[str, str]:
    path = split_dir / "transcripts.txt"
    if not path.is_file():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        if "\t" in line:
            utt, text = line.split("\t", 1)
        else:
            utt, text = line.split(" ", 1)
        out[utt.strip()] = text.strip()
    return out


def _mls(raw: Path) -> list[dict]:
    extracted = raw / "mls_it" / "extracted"
    if not extracted.is_dir():
        print(f"skip mls: no {extracted}", flush=True)
        return []
    rows = []
    # Train / dev / test only. limited_supervision lists are subsets of train
    # and would double-count hours if their audio were walked as well.
    split_dirs = []
    for transcripts in extracted.glob("**/transcripts.txt"):
        split_dir = transcripts.parent
        if split_dir.name not in {"train", "dev", "test"}:
            continue
        if "limited_supervision" in split_dir.parts:
            continue
        split_dirs.append(split_dir)
    if not split_dirs:
        print("skip mls: no train/dev/test transcripts", flush=True)
        return []
    for split_dir in sorted(split_dirs, key=lambda p: p.name):
        texts = _load_transcripts(split_dir)
        flacs = sorted(split_dir.glob("audio/**/*.flac"))
        print(f"mls {split_dir.name}: {len(flacs)} flac files", flush=True)
        for flac in flacs:
            if "limited_supervision" in flac.parts:
                continue
            rel_parts = flac.relative_to(split_dir / "audio").parts
            if len(rel_parts) < 2:
                raise RuntimeError(f"unexpected MLS path {flac}")
            speaker_id = rel_parts[0]
            info = header_info(flac)
            rows.append(
                _row(
                    "mls_it",
                    id=f"mls_it/{split_dir.name}/{flac.stem}",
                    split=split_dir.name,
                    path=str(flac.relative_to(raw)),
                    duration_s=info["duration_s"],
                    sample_rate=info["sample_rate"],
                    channels=info["channels"],
                    layout="mono",
                    speaker_id=str(speaker_id),
                    text=texts.get(flac.stem),
                    extra={"chapter_id": rel_parts[1] if len(rel_parts) > 2 else None},
                )
            )
    return rows


def summarise(rows: list[dict]) -> dict:
    by_source: dict[str, dict] = {}
    for row in rows:
        bucket = by_source.setdefault(
            row["source"],
            {"clips": 0, "hours": 0.0, "splits": {}, "speakers": set(), "channels": {}},
        )
        bucket["clips"] += 1
        bucket["hours"] += float(row["duration_s"]) / 3600.0
        split = bucket["splits"].setdefault(row["split"], {"clips": 0, "hours": 0.0})
        split["clips"] += 1
        split["hours"] += float(row["duration_s"]) / 3600.0
        if row.get("speaker_id"):
            bucket["speakers"].add(row["speaker_id"])
        ch = str(row["channels"])
        bucket["channels"][ch] = bucket["channels"].get(ch, 0) + 1
    out = {}
    for source, bucket in by_source.items():
        out[source] = {
            "clips": bucket["clips"],
            "hours": bucket["hours"],
            "splits": bucket["splits"],
            "n_speaker_ids": len(bucket["speakers"]),
            "channels": bucket["channels"],
            "licence": SOURCES[source]["licence"],
        }
    out["total_hours"] = sum(v["hours"] for v in out.values())
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, default=ROOT / "raw")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    rows: list[dict] = []
    rows.extend(_datahive(args.raw))
    rows.extend(_fleurs(args.raw))
    rows.extend(_mls(args.raw))
    if not rows:
        print("no clips found under the raw directory", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = summarise(rows)
    hours_path = args.out.parent / "hours.json"
    hours_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} rows to {args.out}", flush=True)
    for source, info in summary.items():
        if source == "total_hours":
            continue
        print(
            f"  {source}: {info['hours']:.3f} h, {info['clips']} clips, "
            f"speaker ids {info['n_speaker_ids']}, splits { {k: round(v['hours'], 3) for k, v in info['splits'].items()} }",
            flush=True,
        )
    print(f"total hours: {summary['total_hours']:.3f}", flush=True)
    print(f"wrote {hours_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
