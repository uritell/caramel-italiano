#!/usr/bin/env python3
"""Score stereo files with a VAD config already measured on DataHive.

The config's VAD block is used unchanged, so real and synthetic numbers
come from the same detector.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

def _plain(obj):
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _plain(obj.tolist())
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from activity import VadConfig, analyse_prepared, pool, prepare  # noqa: E402
from audio import read_audio  # noqa: E402


def score_paths(paths: list[Path], cfg: VadConfig) -> dict:
    stats_list = []
    per_file = []
    for path in paths:
        print(f"scoring {path}", flush=True)
        audio, sr = read_audio(path, sample_rate=16000)
        if audio.shape[1] == 1:
            raise RuntimeError(f"{path} is mono; this scorer wants one speaker per channel")
        prep = prepare(audio[:, 0], audio[:, 1], sr, cfg)
        stats, _, _ = analyse_prepared(prep, cfg)
        stats_list.append(stats)
        row = stats.headline()
        row["path"] = str(path)
        per_file.append(row)
        del audio
    pooled = pool(stats_list)
    return {
        "per_file": per_file,
        "corpus": {k: v for k, v in pooled.items() if not isinstance(v, list)},
        "n_gaps": len(pooled["gap_ms"]),
        "n_backchannels": len(pooled["backchannel_duration_ms"]),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "synth" / "timing_config.json")
    p.add_argument("--audio", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    document = json.loads(args.config.read_text(encoding="utf-8"))
    cfg = VadConfig(**document["vad"])
    result = _plain(score_paths(list(args.audio), cfg))
    result["config"] = str(args.config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    c = result["corpus"]
    print(
        f"turns/min {c['turns_per_min']:.2f}  both {100 * c['pct_both']:.1f}%  "
        f"nobody {100 * c['pct_nobody']:.1f}%  median gap {c['median_gap_ms']:.0f} ms  "
        f"overlapping changes/min {c['overlapping_turn_changes_per_min']:.2f}",
        flush=True,
    )
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
