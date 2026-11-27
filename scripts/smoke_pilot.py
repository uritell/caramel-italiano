#!/usr/bin/env python3
"""End-to-end smoke of the pilot pipeline.

Downloads nothing that is not already handled by data/download.py. Builds
about one hour of synthetic stereo from MLS, scores it against DataHive,
tokenizes that hour plus a DataHive slice plus a few minutes of FLEURS
mono, and checks each store with upstream TokenStore.

Audio, tokens and checkpoints stay under gitignored directories.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(args: list[str]) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_items(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hours", type=float, default=1.0)
    p.add_argument("--mono-minutes", type=float, default=10.0)
    p.add_argument("--datahive-slice-minutes", type=float, default=8.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-download", action="store_true")
    args = p.parse_args()

    if not args.skip_download:
        run([PY, "data/download.py", "--source", "datahive_it_conversations_sample", "--source", "fleurs_it_it", "--source", "mls_it"])

    run([PY, "score/activity.py", "--self-test"])
    run([PY, "tokenize/verify_store.py", "--self-test"])
    run([PY, "score/measure.py"])
    run([PY, "data/manifest.py"])

    run(
        [
            PY,
            "synth/synthesize.py",
            "--hours",
            str(args.hours),
            "--seed",
            str(args.seed),
        ]
    )

    synth_manifest = _rows(ROOT / "synth" / "out" / "manifest.jsonl")
    wavs = [row["path"] for row in synth_manifest]
    run([PY, "score/score_files.py", "--audio", *wavs, "--out", "score/out/synthetic.json"])
    run([PY, "score/compare.py", "--synthetic", "score/out/synthetic.json"])

    manifest = _rows(ROOT / "data" / "work" / "manifest.jsonl")
    raw = ROOT / "data" / "raw"
    work = ROOT / "data" / "work"
    fleurs = [r for r in manifest if r["source"] == "fleurs_it_it" and r["split"] == "train"]
    fleurs.sort(key=lambda r: r["id"])
    mono_items = []
    acc = 0.0
    for row in fleurs:
        if acc >= args.mono_minutes * 60.0:
            break
        mono_items.append(
            {"id": row["id"], "audio": str(raw / row["path"]), "mode": "mono"}
        )
        acc += float(row["duration_s"])
    hive = [r for r in manifest if r["source"] == "datahive_it_conversations_sample"]
    hive.sort(key=lambda r: float(r["duration_s"]))
    if len(hive) < 2:
        raise SystemExit("need at least two DataHive files")
    dev = hive[0]
    # A different conversation from the dev file, sliced from the start.
    sliced = max(hive[1:], key=lambda r: float(r["duration_s"]))
    dev_items = [
        {
            "id": dev["id"],
            "audio": str(raw / dev["path"]),
            "mode": "stereo",
        }
    ]
    stereo_items = [
        {
            "id": row["id"],
            "audio": row["path"],
            "mode": "stereo",
        }
        for row in synth_manifest
    ]
    stereo_items.append(
        {
            "id": sliced["id"] + "_slice",
            "audio": str(raw / sliced["path"]),
            "mode": "stereo",
            "max_seconds": args.datahive_slice_minutes * 60.0,
        }
    )
    _write_items(work / "items_mono.jsonl", mono_items)
    _write_items(work / "items_dev.jsonl", dev_items)
    _write_items(work / "items_stereo.jsonl", stereo_items)
    (work / "splits.json").write_text(
        json.dumps(
            {
                "dev_datahive_id": dev["id"],
                "stereo_slice_datahive_id": sliced["id"],
                "note": (
                    "Timing priors in synth/timing_config.json were measured on all "
                    "six DataHive conversations, because that is the entire open "
                    "stereo sample. The dev id is held out of the stereo training "
                    "store and written to tokenize/out/dev for a later train.py --dev."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    t0 = time.perf_counter()
    run(
        [
            PY,
            "tokenize/encode_store.py",
            "--items",
            str(work / "items_mono.jsonl"),
            "--out",
            "tokenize/out/mono",
            "--width",
            "8",
            "--benchmark-seconds",
            "60",
            "--check-chunking",
        ]
    )
    run(
        [
            PY,
            "tokenize/encode_store.py",
            "--items",
            str(work / "items_dev.jsonl"),
            "--out",
            "tokenize/out/dev",
            "--width",
            "16",
        ]
    )
    run(
        [
            PY,
            "tokenize/encode_store.py",
            "--items",
            str(work / "items_stereo.jsonl"),
            "--out",
            "tokenize/out/stereo",
            "--width",
            "16",
        ]
    )
    tokenize_wall = time.perf_counter() - t0
    run(
        [
            PY,
            "tokenize/verify_store.py",
            "--store",
            "tokenize/out/mono",
            "--store",
            "tokenize/out/dev",
            "--store",
            "tokenize/out/stereo",
        ]
    )

    reports = []
    for name in ("mono", "dev", "stereo"):
        reports.append(json.loads((ROOT / "tokenize" / "out" / name / "encode_report.json").read_text()))
    channel_s = sum(r["channel_seconds"] for r in reports)
    encode_wall = sum(r["encode_wall_s"] for r in reports)
    rtf = encode_wall / channel_s if channel_s else float("nan")
    bench = reports[0].get("benchmark") or {}
    chunk = None
    # check-chunking prints JSON but does not save it. Re-read is unnecessary;
    # the mono report is enough for throughput. Chunk result is in the log.
    speed = {
        "channel_seconds_encoded": channel_s,
        "encode_wall_s": encode_wall,
        "rtf_encode_over_channel_audio": rtf,
        "benchmark_60s": bench,
        "tokenize_step_wall_s_including_load": tokenize_wall,
        "cpu_hours_for_channel_hours": {
            "300": (300 * 3600 * rtf) / 3600.0,
            "600": (600 * 3600 * rtf) / 3600.0,
        },
        "cpu_hours_for_stereo_hours": {
            "300": (300 * 2 * 3600 * rtf) / 3600.0,
            "600": (600 * 2 * 3600 * rtf) / 3600.0,
        },
        "gpu": (
            "Not measured. This run has no GPU and does not rent one. "
            "A published Candle Mimi encode on an M4 with Metal is about 9.1 times "
            "realtime, which is a different implementation and a laptop GPU, so it "
            "is not a quote for an H100 running this PyTorch path."
        ),
    }
    (ROOT / "score" / "out" / "speed.json").write_text(json.dumps(speed, indent=2) + "\n", encoding="utf-8")
    comparison = ROOT / "score" / "comparison.md"
    extra = [
        "",
        "## Mimi encode speed on this CPU",
        "",
        f"Encoded {channel_s / 3600:.3f} channel-hours in {encode_wall / 60:.1f} min of encode time "
        f"(model load excluded from that figure; the whole tokenize step was {tokenize_wall / 60:.1f} min).",
        f"Real-time factor (encode wall / channel-audio) = {rtf:.4f}, "
        f"which is {1 / rtf:.1f}× realtime." if rtf else "RTF unavailable.",
        "",
        "A stereo hour is two channel-hours, because each speaker is encoded.",
        "",
        "| Audio | CPU wall time at this measured rate |",
        "|---|---:|",
        f"| 300 h mono | {speed['cpu_hours_for_channel_hours']['300']:.1f} h |",
        f"| 600 h mono | {speed['cpu_hours_for_channel_hours']['600']:.1f} h |",
        f"| 300 h stereo | {speed['cpu_hours_for_stereo_hours']['300']:.1f} h |",
        f"| 600 h stereo | {speed['cpu_hours_for_stereo_hours']['600']:.1f} h |",
        "",
        speed["gpu"],
        "",
    ]
    if bench:
        extra.insert(
            6,
            f"A separate 60 s benchmark, warmup excluded, two repeats: RTF {bench.get('rtf'):.4f}.",
        )
    comparison.write_text(comparison.read_text(encoding="utf-8") + "\n".join(extra), encoding="utf-8")
    print(json.dumps({k: v for k, v in speed.items() if k != "gpu"}, indent=2), flush=True)
    print("smoke finished", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
