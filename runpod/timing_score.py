#!/usr/bin/env python3
"""Timing statistics for stereo Italian audio (one speaker per channel).

caramel published the names of these measurements and did not publish the
code or the English band values. This file measures the same ideas from the
two tracks. Bands are fitted only from held-out real conversations, as the
10th to 90th percentile across 60 s windows. With fewer than four windows
the score is left empty. No band number is typed in by hand.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HOP_S = 0.010
WIN_S = 0.020
MIN_TURN_S = 0.150
BACKCHANNEL_S = 0.500
DEAD_AIR_S = 1.5
BAND_WINDOW_S = 60.0
BAND_HOP_S = 30.0
MIN_WINDOWS = 4

STAT_KEYS = (
    "turns_per_minute",
    "overlap_fraction",
    "silence_fraction",
    "median_gap_ms",
    "overlapping_turn_changes_per_minute",
    "backchannels_per_minute",
    "speech_balance",
    "dead_air_fraction",
    "crosstalk_db",
    "noise_floor_dbfs",
    "median_turn_s",
    "active_fraction",
    "words_per_minute",
)


def _rms_frames(x: np.ndarray, sr: int) -> np.ndarray:
    win = max(1, int(round(WIN_S * sr)))
    hop = max(1, int(round(HOP_S * sr)))
    if x.ndim == 1:
        x = x[:, None]
    n = x.shape[0]
    if n < win:
        pad = np.zeros((win - n, x.shape[1]), dtype=np.float32)
        x = np.concatenate([x, pad], axis=0)
        n = x.shape[0]
    starts = range(0, n - win + 1, hop)
    frames = []
    for s in starts:
        chunk = x[s:s + win]
        frames.append(np.sqrt(np.mean(chunk * chunk, axis=0) + 1e-12))
    return np.asarray(frames, dtype=np.float64)


def _mask_from_rms(rms: np.ndarray) -> np.ndarray:
    """True where that channel is treated as talking."""
    active = np.zeros(rms.shape, dtype=bool)
    for ch in range(rms.shape[1]):
        col = rms[:, ch]
        floor = float(np.percentile(col, 20))
        thresh = max(3.0 * floor, 0.01)
        active[:, ch] = col >= thresh
    hop = max(1, int(round(0.050 / HOP_S)))
    # Drop blips shorter than 50 ms and fill holes shorter than 50 ms.
    for ch in range(active.shape[1]):
        a = active[:, ch].copy()
        a = _fill_short(a, hop, fill=False)
        a = _fill_short(~a, hop, fill=False)
        active[:, ch] = ~a
    return active


def _fill_short(mask: np.ndarray, min_len: int, fill: bool) -> np.ndarray:
    """Replace runs of True shorter than min_len with `fill`."""
    out = mask.copy()
    n = len(mask)
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i + 1
        while j < n and mask[j]:
            j += 1
        if j - i < min_len:
            out[i:j] = fill
        i = j
    return out


def _runs(active: np.ndarray) -> list[tuple[str, int, int]]:
    """Exclusive-speaker runs: ('L' or 'R', start_frame, end_frame)."""
    left = active[:, 0] & ~active[:, 1]
    right = active[:, 1] & ~active[:, 0]
    label = np.full(len(active), "", dtype=object)
    label[left] = "L"
    label[right] = "R"
    runs = []
    i = 0
    n = len(label)
    while i < n:
        if label[i] == "":
            i += 1
            continue
        j = i + 1
        while j < n and label[j] == label[i]:
            j += 1
        runs.append((str(label[i]), i, j))
        i = j
    return runs


def stats_from_audio(x: np.ndarray, sr: int, word_count: int | None = None) -> dict:
    if x.ndim != 2 or x.shape[1] < 2:
        raise ValueError("timing stats need stereo audio, one speaker per channel")
    x = np.asarray(x[:, :2], dtype=np.float32)
    rms = _rms_frames(x, sr)
    active = _mask_from_rms(rms)
    n = len(active)
    dur_s = max(n * HOP_S, 1e-6)
    minutes = dur_s / 60.0
    both = active[:, 0] & active[:, 1]
    neither = ~active[:, 0] & ~active[:, 1]
    exclusive = active[:, 0] ^ active[:, 1]
    runs = _runs(active)
    long_runs = [(sp, a, b) for sp, a, b in runs if (b - a) * HOP_S >= MIN_TURN_S]
    gaps = []
    overlap_changes = 0
    changes = 0
    for (sp1, _a1, b1), (sp2, a2, _b2) in zip(long_runs, long_runs[1:]):
        if sp1 == sp2:
            continue
        changes += 1
        gap_s = (a2 - b1) * HOP_S
        gaps.append(gap_s * 1000.0)
        if gap_s < 0:
            overlap_changes += 1
    backchannels = 0
    for sp, a, b in runs:
        length = (b - a) * HOP_S
        if length >= BACKCHANNEL_S or length < 0.08:
            continue
        other = 1 if sp == "L" else 0
        pad = int(round(0.5 / HOP_S))
        lo = max(0, a - pad)
        hi = min(n, b + pad)
        if active[lo:hi, other].mean() > 0.5:
            backchannels += 1
    turn_lengths = [(b - a) * HOP_S for _sp, a, b in long_runs]
    left_s = float((active[:, 0] & ~active[:, 1]).sum() * HOP_S)
    right_s = float((active[:, 1] & ~active[:, 0]).sum() * HOP_S)
    balance_den = left_s + right_s
    balance = max(left_s, right_s) / balance_den if balance_den > 0 else 1.0

    dead = 0.0
    i = 0
    while i < n:
        if not neither[i]:
            i += 1
            continue
        j = i + 1
        while j < n and neither[j]:
            j += 1
        length = (j - i) * HOP_S
        if length >= DEAD_AIR_S:
            dead += length
        i = j

    crosstalk = []
    for ch, other in ((0, 1), (1, 0)):
        sel = active[:, ch] & ~active[:, other]
        if not np.any(sel):
            continue
        ratio = rms[sel, other] / np.maximum(rms[sel, ch], 1e-8)
        crosstalk.append(20.0 * np.log10(np.maximum(ratio, 1e-8)))
    if crosstalk:
        cross_db = float(np.median(np.concatenate(crosstalk)))
    else:
        cross_db = -80.0
    floor = float(np.percentile(rms, 20))
    noise_db = 20.0 * np.log10(max(floor, 1e-8))
    wpm = None if word_count is None else float(word_count) / minutes
    return {
        "duration_s": round(dur_s, 3),
        "turns_per_minute": round(len(long_runs) / minutes, 3),
        "overlap_fraction": round(float(both.mean()), 4),
        "silence_fraction": round(float(neither.mean()), 4),
        "median_gap_ms": None if not gaps else round(float(np.median(gaps)), 1),
        "overlapping_turn_changes_per_minute": round(overlap_changes / minutes, 3),
        "backchannels_per_minute": round(backchannels / minutes, 3),
        "speech_balance": round(float(balance), 4),
        "dead_air_fraction": round(dead / dur_s, 4),
        "crosstalk_db": round(cross_db, 2),
        "noise_floor_dbfs": round(noise_db, 2),
        "median_turn_s": None if not turn_lengths else round(float(np.median(turn_lengths)), 3),
        "active_fraction": round(float((~neither).mean()) if n else 0.0, 4),
        "exclusive_fraction": round(float(exclusive.mean()), 4),
        "words_per_minute": None if wpm is None else round(wpm, 2),
        "turn_changes": changes,
    }


def windows(x: np.ndarray, sr: int) -> list[dict]:
    win = int(BAND_WINDOW_S * sr)
    hop = int(BAND_HOP_S * sr)
    if x.shape[0] < win:
        return []
    out = []
    for start in range(0, x.shape[0] - win + 1, hop):
        out.append(stats_from_audio(x[start:start + win], sr))
    return out


def fit_bands(rows: list[dict]) -> dict | None:
    if len(rows) < MIN_WINDOWS:
        return None
    bands = {}
    for key in STAT_KEYS:
        vals = [r[key] for r in rows if r.get(key) is not None]
        if len(vals) < MIN_WINDOWS:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        lo, hi = np.percentile(arr, [10, 90])
        bands[key] = {"lo": float(lo), "hi": float(hi), "n": int(len(vals))}
    if len(bands) < 4:
        return None
    return {
        "method": "10th-90th percentile of 60s windows, hop 30s, held-out real stereo only",
        "windows": len(rows),
        "bands": bands,
    }


def closeness(stats: dict, bands: dict | None) -> float | None:
    if not bands or "bands" not in bands:
        return None
    excess = []
    for key, band in bands["bands"].items():
        value = stats.get(key)
        if value is None:
            continue
        lo, hi = float(band["lo"]), float(band["hi"])
        width = max(abs(hi - lo), 1e-6)
        if value < lo:
            excess.append((lo - value) / width)
        elif value > hi:
            excess.append((value - hi) / width)
        else:
            excess.append(0.0)
    if not excess:
        return None
    return float(np.mean(excess))


def score_file(path: Path, word_count: int | None = None) -> dict:
    from audioio import read_audio

    audio, sr = read_audio(path)
    stats = stats_from_audio(audio, sr, word_count=word_count)
    stats["path"] = str(path)
    return stats


def self_check() -> None:
    sr = 24000
    n = sr * 3
    rng = np.random.default_rng(0)
    left = np.zeros(n, dtype=np.float32)
    right = np.zeros(n, dtype=np.float32)
    left[:sr] = rng.normal(0, 0.2, sr).astype(np.float32)
    right[int(1.2 * sr):int(2.2 * sr)] = rng.normal(0, 0.2, sr).astype(np.float32)
    gap = stats_from_audio(np.stack([left, right], 1), sr)
    if gap["overlap_fraction"] > 0.05:
        raise SystemExit(f"expected little overlap, got {gap}")
    if gap["median_gap_ms"] is None or not (50 <= gap["median_gap_ms"] <= 500):
        raise SystemExit(f"expected a gap near 200 ms, got {gap['median_gap_ms']}")
    if gap["turns_per_minute"] <= 0:
        raise SystemExit(gap)
    both = np.stack([left, left], 1)
    over = stats_from_audio(both, sr)
    if over["overlap_fraction"] < 0.2:
        raise SystemExit(f"expected overlap when both channels match, got {over['overlap_fraction']}")
    # One short file must not invent bands.
    if fit_bands([gap, gap, gap]) is not None:
        raise SystemExit("bands were fitted from too few windows")
    print("timing self-check ok", flush=True)
    print(json.dumps({k: gap[k] for k in STAT_KEYS}, indent=2))


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-check", action="store_true")
    p.add_argument("--audio", type=Path, nargs="*", default=[])
    p.add_argument("--fit-from", type=Path, nargs="*", default=[], help="held-out real stereo used only to fit bands")
    p.add_argument("--out", type=Path)
    args = p.parse_args(argv)
    if args.self_check:
        self_check()
        if not args.audio and not args.fit_from:
            return 0
    bands = None
    if args.fit_from:
        rows = []
        from audioio import read_audio

        for path in args.fit_from:
            audio, sr = read_audio(path)
            rows.extend(windows(audio, sr))
        bands = fit_bands(rows)
    reports = []
    for path in args.audio:
        stats = score_file(path)
        stats["closeness"] = closeness(stats, bands)
        reports.append(stats)
        print(json.dumps(stats), flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"bands": bands, "clips": reports}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
