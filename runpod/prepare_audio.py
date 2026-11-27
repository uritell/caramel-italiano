#!/usr/bin/env python3
"""Build the mono clips, the synthetic stereo dialogues, and the held-out sets.

Real DataHive conversations stay real: the shortest one is dev, the next
shortest is used only to fit timing bands, and the rest train. Synthetic
dialogues alternate two different single-speaker clips with a short gap and
silence on the other channel. They do not invent overlap or backchannels.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from audioio import SR, read_audio, resample, to_mono, to_stereo, write_audio

GAP_S = 0.25
TURN_CAP_S = 12.0
DIALOGUE_S = 90.0


def _load_manifest(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _as_24k(row: dict, channels: int):
    audio, sr = read_audio(Path(row["path"]))
    audio = resample(audio, sr, SR)
    if channels == 1:
        return to_mono(audio).reshape(-1, 1)
    return to_stereo(audio)


def _split_datahive(rows: list[dict], work: Path) -> dict[str, list[dict]]:
    items = []
    for row in rows:
        audio = _as_24k(row, 2)
        items.append((row, audio))
    items.sort(key=lambda pair: len(pair[1]))
    if len(items) >= 3:
        dev_rows = items[:1]
        score_rows = items[1:2]
        train_rows = items[2:]
    elif len(items) == 2:
        dev_rows, train_rows, score_rows = items[:1], items[1:], []
    elif len(items) == 1:
        row, audio = items[0]
        mid = len(audio) // 2
        dev_rows = [({**row, "split": "dev-half"}, audio[mid:])]
        train_rows = [({**row, "split": "train-half"}, audio[:mid])]
        score_rows = []
    else:
        dev_rows = train_rows = score_rows = []

    def _save(name, pairs):
        saved = []
        for i, (row, audio) in enumerate(pairs):
            if len(audio) < SR // 2:
                continue
            path = work / name / f"{i:03d}.flac"
            write_audio(path, audio, SR)
            saved.append({
                "path": str(path),
                "duration_s": round(len(audio) / SR, 3),
                "channels": 2,
                "source": row.get("source", "datahive"),
                "role": name,
            })
        return saved

    return {
        "train": _save("real_train", train_rows),
        "dev": _save("real_dev", dev_rows),
        "score": _save("real_score", score_rows),
    }


def build_dialogues(mono_rows: list[dict], dest: Path, dialogue_s: float, max_total_s: float = 0.0) -> list[dict]:
    import numpy as np

    queues: dict[str, list] = defaultdict(list)
    for row in mono_rows:
        audio = _as_24k(row, 1).reshape(-1)
        cap = int(TURN_CAP_S * SR)
        while len(audio) > int(0.4 * SR):
            piece = audio[:cap]
            audio = audio[cap:]
            queues[str(row.get("speaker_id") or "unk")].append({
                "audio": piece,
                "source": row.get("path"),
                "speaker": str(row.get("speaker_id") or "unk"),
            })
    speakers = [s for s, q in queues.items() if q]
    if len(speakers) < 2:
        print(f"synth skipped: {len(speakers)} speaker(s)", flush=True)
        return []
    dest.mkdir(parents=True, exist_ok=True)
    made = []
    index = 0
    guard = 0
    total_s = 0.0
    while True:
        remaining = sorted(
            ((sum(len(c["audio"]) for c in queues[s]), s) for s in speakers if queues[s]),
            reverse=True,
        )
        if len(remaining) < 2:
            break
        sp_a, sp_b = remaining[0][1], remaining[1][1]
        left_parts = []
        right_parts = []
        turns = []
        t = 0.0
        side = 0
        while t < dialogue_s and queues[sp_a] and queues[sp_b]:
            sp = sp_a if side == 0 else sp_b
            clip = queues[sp].pop(0)
            dur = len(clip["audio"]) / SR
            left_parts.append(clip["audio"] if side == 0 else np.zeros(len(clip["audio"]), np.float32))
            right_parts.append(clip["audio"] if side == 1 else np.zeros(len(clip["audio"]), np.float32))
            turns.append({
                "speaker": sp,
                "channel": "left" if side == 0 else "right",
                "start_s": round(t, 3),
                "end_s": round(t + dur, 3),
                "source": clip["source"],
            })
            t += dur
            gap = np.zeros(int(GAP_S * SR), np.float32)
            left_parts.append(gap)
            right_parts.append(gap)
            t += GAP_S
            side = 1 - side
        if not turns:
            break
        stereo = np.stack([np.concatenate(left_parts), np.concatenate(right_parts)], 1)
        path = dest / f"synth_{index:04d}.flac"
        write_audio(path, stereo, SR)
        side_path = dest / f"synth_{index:04d}.json"
        side_path.write_text(json.dumps({"turns": turns}, indent=2) + "\n")
        made.append({
            "path": str(path),
            "duration_s": round(len(stereo) / SR, 3),
            "channels": 2,
            "source": "synth",
            "role": "synth",
            "turns": str(side_path),
        })
        index += 1
        guard += 1
        total_s += made[-1]["duration_s"]
        if max_total_s and total_s >= max_total_s:
            print(f"synth cap {total_s:.1f}s", flush=True)
            break
        if guard > 100000:
            break
    print(f"synth dialogues: {len(made)}", flush=True)
    return made


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def prepare(manifest: Path, work: Path, dialogue_s: float = DIALOGUE_S, max_synth_s: float = 0.0) -> dict:
    rows = _load_manifest(manifest)
    mono = [r for r in rows if r.get("role") != "stereo" and r.get("source") != "datahive"]
    hive = [r for r in rows if r.get("source") == "datahive" or r.get("role") == "stereo"]
    # A mono file that was labelled stereo by mistake stays out of the mono store.
    mono = [r for r in mono if int(r.get("channels") or 1) == 1 or r.get("source") != "datahive"]
    real = _split_datahive(hive, work)
    synth = build_dialogues(mono, work / "synth", dialogue_s, max_synth_s)
    mono_out = []
    for i, row in enumerate(mono):
        audio = _as_24k(row, 1)
        if len(audio) < SR // 5:
            continue
        path = work / "mono" / f"{i:05d}.flac"
        write_audio(path, audio, SR)
        mono_out.append({
            "path": str(path),
            "duration_s": round(len(audio) / SR, 3),
            "channels": 1,
            "source": row.get("source"),
            "speaker_id": row.get("speaker_id"),
            "role": "mono",
        })
    stereo_train = real["train"] + synth
    # The dev set has to exist for stage 3. If DataHive did not download,
    # hold out the last synthetic dialogue and say so.
    dev = list(real["dev"])
    if len(hive) == 1:
        dev_note = "second half of the only DataHive file; the first half is in the stereo training set"
    elif len(hive) >= 2:
        dev_note = "shortest DataHive conversation, held out of training"
    else:
        dev_note = "no DataHive audio"
    if not dev and synth:
        dev = [synth.pop()]
        stereo_train = real["train"] + synth
        dev_note = "synthetic hold-out because no DataHive audio was available"
    plan = {
        "mono": mono_out,
        "stereo": stereo_train,
        "dev": dev,
        "score": real["score"],
        "dev_note": dev_note,
        "gap_s": GAP_S,
        "turn_cap_s": TURN_CAP_S,
    }
    _write_jsonl(work / "mono.jsonl", mono_out)
    _write_jsonl(work / "stereo.jsonl", stereo_train)
    _write_jsonl(work / "dev.jsonl", dev)
    _write_jsonl(work / "score.jsonl", real["score"])
    (work / "plan.json").write_text(json.dumps({
        "n_mono": len(mono_out),
        "n_stereo": len(stereo_train),
        "n_dev": len(dev),
        "n_score": len(real["score"]),
        "mono_s": round(sum(r["duration_s"] for r in mono_out), 1),
        "stereo_s": round(sum(r["duration_s"] for r in stereo_train), 1),
        "dev_s": round(sum(r["duration_s"] for r in dev), 1),
        "dev_note": dev_note,
    }, indent=2) + "\n")
    print(json.dumps(json.loads((work / "plan.json").read_text())), flush=True)
    return plan


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--dialogue-s", type=float, default=DIALOGUE_S)
    p.add_argument("--max-synth-hours", type=float, default=0.0)
    args = p.parse_args(argv)
    prepare(args.manifest, args.work, args.dialogue_s, args.max_synth_hours * 3600.0)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
