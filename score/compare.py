#!/usr/bin/env python3
"""Write the real-vs-synthetic timing comparison.

Reads the DataHive measurement (synth/timing_config.json) and a score JSON
produced for the synthetic set. The prose stays short; the numbers are the
point.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _fmt(x, nd=2):
    if x is None or (isinstance(x, float) and x != x):
        return "n/a"
    return f"{x:.{nd}f}"


def render(real: dict, synthetic: dict) -> str:
    r = real["corpus_stats"]
    s = synthetic["corpus"]
    cal = real.get("calibration_to_datahive_card", {})
    lines = []
    lines.append("# Real DataHive timing vs synthetic two-track dialogues")
    lines.append("")
    lines.append(
        "Both columns use the same energy voice-activity detector. "
        "Its threshold, bridge and intra-turn pause were chosen so the detector "
        "lands near the turns/minute, overlap and median gap printed on the "
        "DataHive dataset card; that fit is a calibration check, not the target "
        "the synthesiser copies. The synthesiser draws each gap, each backchannel "
        "duration and each backchannel position from the empirical lists measured "
        "on all six DataHive conversations. Read-speech clips supply the sound. "
        "They are not real backchannels, and the comparison is expected to show that."
    )
    lines.append("")
    lines.append(
        f"Real audio: {r['n_files']} conversations, {_fmt(r['hours'], 3)} h. "
        f"Synthetic audio: {s['n_files']} conversations, {_fmt(s['hours'], 3)} h."
    )
    lines.append("")
    lines.append("| Statistic | DataHive (measured here) | Synthetic |")
    lines.append("|---|---:|---:|")
    rows = [
        ("Turns per minute", "turns_per_min", 2),
        ("Both talking (%)", "pct_both", "pct"),
        ("Nobody talking (%)", "pct_nobody", "pct"),
        ("Median gap between turns (ms)", "median_gap_ms", 0),
        ("Overlapping turn changes per minute", "overlapping_turn_changes_per_min", 2),
        ("Backchannels per minute", "backchannels_per_min", 2),
    ]
    for label, key, kind in rows:
        rv, sv = r[key], s[key]
        if kind == "pct":
            left, right = f"{100 * rv:.1f}", f"{100 * sv:.1f}"
        elif kind == 0:
            left, right = f"{rv:.0f}", f"{sv:.0f}"
        else:
            left, right = f"{rv:.{kind}f}", f"{sv:.{kind}f}"
        lines.append(f"| {label} | {left} | {right} |")
    lines.append("")
    lines.append("## How the detector was set")
    lines.append("")
    rule = real.get("backchannel_cutoff_rule", "")
    if rule:
        lines.append("Cutoff rule: " + rule[0].upper() + rule[1:])
    lines.append("")
    vad = real["vad"]
    lines.append(
        f"Chosen VAD: threshold fraction {vad['threshold_fraction']}, "
        f"minimum margin {vad['min_margin_db']} dB, "
        f"bridge {vad['bridge_ms']} ms, "
        f"intra-turn pause {vad['intra_turn_pause_ms']} ms, "
        f"backchannel cutoff {vad['backchannel_max_s']} s, "
        f"overlap ratio for a backchannel {vad['backchannel_overlap_ratio']}."
    )
    bc = real.get("empirical", {}).get("backchannel_duration_ms") or []
    if bc:
        lines.append("")
        lines.append(
            f"The cutoff is the long edge of that histogram. The durations actually "
            f"sampled are shorter: median {statistics.median(bc):.0f} ms "
            f"across {len(bc)} measured backchannels."
        )
    if cal:
        lines.append("")
        lines.append(
            f"Grid search: {cal.get('n_trials')} settings, "
            f"{cal.get('n_sane')} inside a loose sanity window, "
            f"mean card-fit error {cal.get('error'):.3f}. "
            "Per file, the card's own labels against this detector:"
        )
        lines.append("")
        lines.append("| File | Card turns/min | Measured turns/min | Card overlap % | Measured both % | Card gap ms | Measured gap ms |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for row in cal.get("per_file", []):
            card = row["card"]
            ours = row["ours"]
            lines.append(
                f"| {row['id']} | {card['turns_per_minute']:.1f} | {ours['turns_per_min']:.1f} | "
                f"{card['overlap_percent']:.1f} | {100 * ours['pct_both']:.1f} | "
                f"{card['turn_taking_gap_ms']:.0f} | {ours['median_gap_ms']:.0f} |"
            )
    lines.append("")
    lines.append("## Reading the gap")
    lines.append("")
    lines.append(
        "A negative median gap means the next speaker usually starts before the "
        "previous one has finished. Overlapping turn changes per minute count only "
        "those speaker changes, not backchannels. Backchannels are short regions "
        "that sit inside the other person's turn; their audio in the synthetic set "
        "is a slice of read speech, cut at a low-energy point, placed with the "
        "measured rate and the measured position along the other turn."
    )
    lines.append("")
    lines.append(
        "The synthetic column is not expected to copy the DataHive column. "
        "MLS utterances are mostly 10–20 seconds and are kept whole, so a "
        "conversation built from them has fewer turns per minute than DataHive "
        "even when every gap is drawn from the measured list. The median gap "
        "in the table is what the detector finds between those longer turns, "
        "not a second hand-set pause. Some drawn backchannels are skipped when "
        "they would land on speech already placed for that speaker, and some "
        "slices are not labelled as backchannels by the detector, so the "
        "synthetic backchannel rate comes out lower than the measured rate."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--real", type=Path, default=ROOT / "synth" / "timing_config.json")
    p.add_argument("--synthetic", type=Path, required=True)
    p.add_argument("--out", type=Path, default=ROOT / "score" / "comparison.md")
    args = p.parse_args()
    real = json.loads(args.real.read_text(encoding="utf-8"))
    synthetic = json.loads(args.synthetic.read_text(encoding="utf-8"))
    text = render(real, synthetic)
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
