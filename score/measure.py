#!/usr/bin/env python3
"""Measure DataHive timing and write the synthesiser's config.

The VAD knobs that are not themselves conversation targets (threshold,
bridge, intra-turn pause) are chosen by a small grid. The score is how
close that VAD lands to the turns/minute, overlap percent and median gap
already printed on the DataHive card. Those card numbers are a calibration
check, not the samples the synthesiser draws.

What the synthesiser draws — the gap list, the backchannel durations and
positions, the backchannel rate, the active-frame RMS, the conversation
lengths — is then measured with the chosen VAD. The backchannel duration
cutoff is the valley in that VAD's overlapped-region histogram. If there
is no valley, backchannels are left empty and the file says so.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
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

from activity import (  # noqa: E402
    VadConfig,
    analyse_prepared,
    backchannel_cutoff_seconds,
    overlapped_durations,
    pool,
    prepare,
)
from audio import read_audio  # noqa: E402

DEFAULT_CONFIG = ROOT / "synth" / "timing_config.json"


def _load_datahive(raw: Path) -> list[dict]:
    meta = raw / "datahive_it_conversations_sample" / "preview" / "metadata.jsonl"
    rows = []
    for line in meta.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _prepare_files(raw: Path, frame_cfg: VadConfig) -> list[dict]:
    prepared = []
    for rec in _load_datahive(raw):
        path = raw / "datahive_it_conversations_sample" / "preview" / rec["file_name"]
        print(f"decoding {path.name}", flush=True)
        audio, sr = read_audio(path, sample_rate=16000)
        if audio.shape[1] < 2:
            raise RuntimeError(f"{path} is not stereo")
        prep = prepare(audio[:, 0], audio[:, 1], sr, frame_cfg)
        prep["id"] = Path(rec["file_name"]).stem
        prep["card"] = {
            "turns_per_minute": float(rec["turns_per_minute"]),
            "turn_taking_gap_ms": float(rec["turn_taking_gap_ms"]),
            "overlap_percent": float(rec["overlap_percent"]),
            "duration_seconds": float(rec["duration_seconds"]),
        }
        prepared.append(prep)
        del audio
    return prepared


def _fit_error(stats, card: dict) -> float:
    tpm = abs(stats.turns_per_min - card["turns_per_minute"]) / max(card["turns_per_minute"], 1.0)
    overlap = abs(100.0 * stats.pct_both - card["overlap_percent"]) / 100.0
    gap = abs(stats.median_gap_ms - card["turn_taking_gap_ms"]) / 500.0
    if stats.median_gap_ms != stats.median_gap_ms:  # NaN
        gap = 2.0
    return tpm + overlap + gap


def _sane(pooled: dict) -> bool:
    return (
        2.0 <= pooled["turns_per_min"] <= 20.0
        and 0.02 <= pooled["pct_both"] <= 0.40
        and pooled["pct_nobody"] <= 0.55
    )


def calibrate(prepared: list[dict], base: VadConfig) -> tuple[VadConfig, dict]:
    fractions = (0.25, 0.35, 0.50)
    margins = (8.0, 12.0)
    intras = (300.0, 600.0, 1000.0)
    bridges = (120.0, 250.0)
    trials = []
    for fraction in fractions:
        for margin in margins:
            for intra in intras:
                for bridge in bridges:
                    cfg = replace(
                        base,
                        threshold_fraction=fraction,
                        min_margin_db=margin,
                        intra_turn_pause_ms=intra,
                        bridge_ms=bridge,
                        backchannel_max_s=None,
                    )
                    durations: list[float] = []
                    for prep in prepared:
                        durations.extend(overlapped_durations(prep, cfg))
                    cutoff, rule = backchannel_cutoff_seconds(durations)
                    cfg = replace(cfg, backchannel_max_s=cutoff)
                    per_file = []
                    stats_list = []
                    err = 0.0
                    for prep in prepared:
                        stats, _, _ = analyse_prepared(prep, cfg)
                        stats_list.append(stats)
                        err += _fit_error(stats, prep["card"])
                        per_file.append(
                            {
                                "id": prep["id"],
                                "ours": stats.headline(),
                                "card": prep["card"],
                            }
                        )
                    err /= len(prepared)
                    pooled = pool(stats_list)
                    trials.append(
                        {
                            "error": err,
                            "sane": _sane(pooled),
                            "cfg": cfg,
                            "cutoff_rule": rule,
                            "pooled": {k: v for k, v in pooled.items() if not isinstance(v, list)},
                            "per_file": per_file,
                            "stats_list": stats_list,
                            "pooled_full": pooled,
                        }
                    )
    sane = [t for t in trials if t["sane"]]
    pool_to_search = sane or trials
    best = min(pool_to_search, key=lambda t: t["error"])
    summary = {
        "n_trials": len(trials),
        "n_sane": len(sane),
        "used_sane_subset": bool(sane),
        "error": best["error"],
        "cutoff_rule": best["cutoff_rule"],
        "chosen_vad": best["cfg"].to_json(),
        "per_file": best["per_file"],
        "note": (
            "error is the mean over the six files of "
            "|turns/min error| / card turns/min + "
            "|overlap percentage-point error| / 100 + "
            "|median gap error| / 500 ms. "
            "Card figures are DataHive's own automatic labels."
        ),
    }
    turn_durs: list[float] = []
    for prep in prepared:
        _stats, floor, _backchannels = analyse_prepared(prep, best["cfg"])
        turn_durs.extend(float(region.duration_s) for region in floor)
    best["pooled_full"]["floor_turn_duration_s"] = turn_durs
    return best["cfg"], {**summary, "_stats": best["stats_list"], "_pooled": best["pooled_full"]}


def build_config(cfg: VadConfig, calibration: dict) -> dict:
    pooled = calibration["_pooled"]
    return {
        "version": 1,
        "measured_from": (
            "DataHive Italian Conversations sample, all 6 stereo files "
            "(caller left, callee right). Distributions below are empirical "
            "samples from the energy VAD, not hand-picked targets."
        ),
        "vad": cfg.to_json(),
        "backchannel_cutoff_rule": calibration["cutoff_rule"],
        "calibration_to_datahive_card": {
            k: v for k, v in calibration.items() if not k.startswith("_")
        },
        "corpus_stats": {k: v for k, v in pooled.items() if not isinstance(v, list)},
        "empirical": {
            "gap_ms": pooled["gap_ms"],
            "backchannel_duration_ms": pooled["backchannel_duration_ms"],
            "backchannel_rel_position": pooled["backchannel_rel_position"],
            "backchannels_per_min": pooled["backchannels_per_min"],
            "target_active_rms": pooled["target_active_rms"],
            "conversation_duration_s": pooled["conversation_duration_s"],
            "floor_turn_duration_s": pooled.get("floor_turn_duration_s", []),
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    p.add_argument("--out", type=Path, default=DEFAULT_CONFIG)
    args = p.parse_args()
    base = VadConfig()
    prepared = _prepare_files(args.raw, base)
    if len(prepared) != 6:
        print(f"expected 6 DataHive files, found {len(prepared)}", file=sys.stderr)
        return 1
    cfg, calibration = calibrate(prepared, base)
    document = _plain(build_config(cfg, calibration))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    stats = document["corpus_stats"]
    print(
        "DataHive energy VAD: "
        f"turns/min {stats['turns_per_min']:.2f}, "
        f"both {100 * stats['pct_both']:.1f}%, "
        f"nobody {100 * stats['pct_nobody']:.1f}%, "
        f"median gap {stats['median_gap_ms']:.0f} ms, "
        f"overlapping changes/min {stats['overlapping_turn_changes_per_min']:.2f}, "
        f"backchannels/min {stats['backchannels_per_min']:.2f}",
        flush=True,
    )
    print(f"calibration error {calibration['error']:.3f}; cutoff: {calibration['cutoff_rule']}", flush=True)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
