#!/usr/bin/env python3
"""Energy voice activity and two-track timing statistics.

The five headline numbers match the timing rows caramel publishes
(turns per minute, both talking, nobody talking, median gap between turns,
overlapping turn changes per minute). A sixth, backchannels per minute, is
measured because the synthesiser needs a rate and a duration distribution.
Upstream did not ship the scorer, so the definitions below are operational
and are written down rather than copied from hidden code.

Activity is per channel, on a shared hop grid:

- frame RMS, then a threshold between a low percentile (noise) and a high
  percentile (speech), with a minimum margin in dB
- gaps shorter than bridge_ms are filled, then blips shorter than
  min_region_ms are dropped
- a region that is mostly inside the other channel's speech and shorter
  than the measured backchannel cutoff is a backchannel, not a floor turn
- floor turns of the same speaker are merged across a pause of at most
  intra_turn_pause_ms when the other speaker does not take a floor turn
  in that pause
- a turn change is two successive floor turns by different speakers
- the gap is next.start - previous.end (negative when the next person
  starts before the previous one finishes)
- an overlapping turn change is a turn change with a negative gap

Percents are fractions of the hop grid, not fractions of speech time.
"Both talking" and "nobody talking" therefore add up with "exactly one"
to 100%.

The backchannel cutoff is not a fixed 1 second. It is the first valley in
the duration histogram of regions that already overlap the other speaker,
computed by `backchannel_cutoff_seconds`. When that valley is missing the
caller is told and backchannels are not invented.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class VadConfig:
    frame_ms: float = 20.0
    hop_ms: float = 10.0
    noise_percentile: float = 10.0
    speech_percentile: float = 90.0
    threshold_fraction: float = 0.35
    min_margin_db: float = 8.0
    bridge_ms: float = 200.0
    min_region_ms: float = 80.0
    intra_turn_pause_ms: float = 500.0
    backchannel_max_s: float | None = None
    backchannel_overlap_ratio: float = 0.5

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class Region:
    speaker: int
    start_s: float
    end_s: float
    overlap_ratio: float
    kind: str  # "floor" or "backchannel"

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass
class ConversationStats:
    duration_s: float
    turns_per_min: float
    pct_both: float
    pct_nobody: float
    median_gap_ms: float
    overlapping_turn_changes_per_min: float
    backchannels_per_min: float
    n_floor_turns: int
    n_turn_changes: int
    n_overlapping_turn_changes: int
    n_backchannels: int
    gap_ms: list[float] = field(default_factory=list)
    backchannel_duration_ms: list[float] = field(default_factory=list)
    backchannel_rel_position: list[float] = field(default_factory=list)
    active_rms_left: float = 0.0
    active_rms_right: float = 0.0
    pct_left: float = 0.0
    pct_right: float = 0.0

    def headline(self) -> dict:
        return {
            "duration_s": self.duration_s,
            "turns_per_min": self.turns_per_min,
            "pct_both": self.pct_both,
            "pct_nobody": self.pct_nobody,
            "median_gap_ms": self.median_gap_ms,
            "overlapping_turn_changes_per_min": self.overlapping_turn_changes_per_min,
            "backchannels_per_min": self.backchannels_per_min,
            "n_floor_turns": self.n_floor_turns,
            "n_turn_changes": self.n_turn_changes,
            "n_overlapping_turn_changes": self.n_overlapping_turn_changes,
            "n_backchannels": self.n_backchannels,
        }


def frame_rms(x: np.ndarray, sr: int, frame_ms: float, hop_ms: float) -> tuple[np.ndarray, int]:
    """Per-hop RMS of a mono signal. Returns (rms, hop_samples)."""
    x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
    frame = max(1, int(round(sr * frame_ms / 1000.0)))
    hop = max(1, int(round(sr * hop_ms / 1000.0)))
    if x.size < frame:
        x = np.pad(x, (0, frame - x.size))
    n = 1 + (x.size - frame) // hop
    strides = (x.strides[0] * hop, x.strides[0])
    windows = np.lib.stride_tricks.as_strided(x, shape=(n, frame), strides=strides)
    rms = np.sqrt(np.mean(windows.astype(np.float64) ** 2, axis=1) + 1e-12)
    return rms.astype(np.float64), hop


def _runs(mask: np.ndarray) -> list[tuple[bool, int, int]]:
    if mask.size == 0:
        return []
    changes = np.flatnonzero(mask[1:] != mask[:-1]) + 1
    bounds = np.concatenate(([0], changes, [mask.size]))
    return [(bool(mask[a]), int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:])]


def smooth_mask(mask: np.ndarray, bridge_frames: int, min_frames: int) -> np.ndarray:
    m = mask.copy()
    if bridge_frames > 0:
        for val, a, b in _runs(m):
            if not val and (b - a) <= bridge_frames and a > 0 and b < m.size:
                m[a:b] = True
    if min_frames > 1:
        for val, a, b in _runs(m):
            if val and (b - a) < min_frames:
                m[a:b] = False
    return m


def activity_mask(rms: np.ndarray, cfg: VadConfig) -> tuple[np.ndarray, float]:
    """Threshold between the noise percentile and the speech percentile.

    The top of the range is the 99th percentile, so a channel that is quiet
    most of the time still counts as speech on its loud frames. A channel
    whose whole level span is under ``min_margin_db`` is treated as all
    speech or all silence from its median level (there is no gap to find).
    """
    db = 20.0 * np.log10(rms + 1e-8)
    noise = float(np.percentile(db, cfg.noise_percentile))
    speech = float(np.percentile(db, cfg.speech_percentile))
    peak = float(np.percentile(db, 99))
    if peak - noise < cfg.min_margin_db:
        loud = float(np.median(db)) > -42.0
        return np.full(db.shape, loud, dtype=bool), float(np.median(db))
    margin = max(cfg.min_margin_db, cfg.threshold_fraction * max(0.0, speech - noise))
    threshold = min(noise + margin, peak - 1.0)
    return db > threshold, float(threshold)


def _regions(mask: np.ndarray, speaker: int, hop_s: float, other: np.ndarray) -> list[Region]:
    out = []
    for val, a, b in _runs(mask):
        if not val:
            continue
        start = a * hop_s
        end = b * hop_s
        overlap = float(other[a:b].mean()) if b > a else 0.0
        out.append(Region(speaker, start, end, overlap, "floor"))
    return out


def backchannel_cutoff_seconds(overlapped_durations_s: list[float], hi_s: float = 4.0, bin_s: float = 0.1) -> tuple[float | None, str]:
    """First valley after the first peak in the overlapped-region histogram.

    Returns (cutoff_seconds, rule). cutoff is None when there are too few
    regions or the histogram does not fall and rise again.
    """
    durations = [d for d in overlapped_durations_s if 0 < d <= hi_s]
    if len(durations) < 8:
        return None, f"no cutoff: only {len(durations)} overlapped regions at or under {hi_s:.1f}s"
    edges = np.arange(0.0, hi_s + bin_s, bin_s)
    hist, edges = np.histogram(durations, bins=edges)
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    kernel /= kernel.sum()
    smooth = np.convolve(hist.astype(np.float64), kernel, mode="same")
    peak = None
    for i in range(1, len(smooth) - 1):
        if smooth[i] >= smooth[i - 1] and smooth[i] >= smooth[i + 1] and smooth[i] > 0:
            peak = i
            break
    if peak is None:
        return None, "no cutoff: overlapped durations have no peak"
    for i in range(peak + 1, len(smooth) - 1):
        if smooth[i] <= smooth[i - 1] and smooth[i] <= smooth[i + 1] and smooth[i] < smooth[peak]:
            cutoff = float((edges[i] + edges[i + 1]) / 2.0)
            return cutoff, (
                f"valley in the histogram of regions with overlap ratio "
                f">= the configured threshold, bin {bin_s:.2f}s, cutoff {cutoff:.3f}s "
                f"({len(durations)} regions)"
            )
    return None, "no cutoff: overlapped durations do not valley after the short peak"


def _merge_floor(regions: list[Region], intra_s: float) -> list[Region]:
    ordered = sorted(regions, key=lambda r: (r.start_s, r.end_s, r.speaker))
    merged: list[Region] = []
    for region in ordered:
        if (
            merged
            and merged[-1].speaker == region.speaker
            and (region.start_s - merged[-1].end_s) <= intra_s
        ):
            merged[-1].end_s = max(merged[-1].end_s, region.end_s)
            merged[-1].overlap_ratio = max(merged[-1].overlap_ratio, region.overlap_ratio)
        else:
            merged.append(
                Region(region.speaker, region.start_s, region.end_s, region.overlap_ratio, "floor")
            )
    return merged


def _rel_position(bc: Region, floor: list[Region]) -> float | None:
    best = None
    best_overlap = 0.0
    for turn in floor:
        if turn.speaker == bc.speaker:
            continue
        overlap = min(bc.end_s, turn.end_s) - max(bc.start_s, turn.start_s)
        if overlap > best_overlap:
            best_overlap = overlap
            best = turn
    if best is None or best.duration_s <= 0:
        return None
    rel = (bc.start_s - best.start_s) / best.duration_s
    return float(min(1.0, max(0.0, rel)))


def prepare(left: np.ndarray, right: np.ndarray, sr: int, cfg: VadConfig) -> dict:
    rms_l, hop = frame_rms(left, sr, cfg.frame_ms, cfg.hop_ms)
    rms_r, hop_r = frame_rms(right, sr, cfg.frame_ms, cfg.hop_ms)
    if hop != hop_r:
        raise RuntimeError("hop mismatch")
    n = min(rms_l.size, rms_r.size)
    return {
        "rms_l": rms_l[:n],
        "rms_r": rms_r[:n],
        "hop_samples": hop,
        "sr": sr,
        "n_samples": int(min(len(left), len(right))),
    }


def channel_masks(prep: dict, cfg: VadConfig) -> tuple[np.ndarray, np.ndarray, float]:
    raw_l, _thr_l = activity_mask(prep["rms_l"], cfg)
    raw_r, _thr_r = activity_mask(prep["rms_r"], cfg)
    bridge = int(round(cfg.bridge_ms / cfg.hop_ms))
    min_frames = int(round(cfg.min_region_ms / cfg.hop_ms))
    mask_l = smooth_mask(raw_l, bridge, min_frames)
    mask_r = smooth_mask(raw_r, bridge, min_frames)
    hop_s = prep["hop_samples"] / prep["sr"]
    return mask_l, mask_r, hop_s


def overlapped_durations(prep: dict, cfg: VadConfig) -> list[float]:
    """Durations of regions that already overlap the other speaker.

    Used to place the backchannel cutoff. No duration limit is applied here.
    """
    mask_l, mask_r, hop_s = channel_masks(prep, cfg)
    regions = _regions(mask_l, 0, hop_s, mask_r) + _regions(mask_r, 1, hop_s, mask_l)
    return [r.duration_s for r in regions if r.overlap_ratio >= cfg.backchannel_overlap_ratio]


def analyse_prepared(prep: dict, cfg: VadConfig) -> tuple[ConversationStats, list[Region], list[Region]]:
    mask_l, mask_r, hop_s = channel_masks(prep, cfg)
    n = mask_l.size
    both = float(np.mean(mask_l & mask_r)) if n else 0.0
    nobody = float(np.mean(~mask_l & ~mask_r)) if n else 0.0
    regions = _regions(mask_l, 0, hop_s, mask_r) + _regions(mask_r, 1, hop_s, mask_l)
    cutoff = cfg.backchannel_max_s
    backchannels: list[Region] = []
    floor: list[Region] = []
    for region in regions:
        is_bc = (
            cutoff is not None
            and region.duration_s <= cutoff
            and region.overlap_ratio >= cfg.backchannel_overlap_ratio
        )
        if is_bc:
            region.kind = "backchannel"
            backchannels.append(region)
        else:
            floor.append(region)
    floor = _merge_floor(floor, cfg.intra_turn_pause_ms / 1000.0)
    floor.sort(key=lambda r: (r.start_s, r.speaker))
    gaps: list[float] = []
    n_overlap_changes = 0
    for prev, nxt in zip(floor, floor[1:]):
        if prev.speaker == nxt.speaker:
            continue
        gap_ms = (nxt.start_s - prev.end_s) * 1000.0
        gaps.append(float(gap_ms))
        if gap_ms < 0:
            n_overlap_changes += 1
    duration_s = n * hop_s
    minutes = duration_s / 60.0 if duration_s else 0.0
    bc_durations = []
    bc_positions = []
    for bc in backchannels:
        bc_durations.append(bc.duration_s * 1000.0)
        rel = _rel_position(bc, floor)
        if rel is not None:
            bc_positions.append(rel)

    def _active_rms(rms: np.ndarray, mask: np.ndarray) -> float:
        if not np.any(mask):
            return 0.0
        return float(np.median(rms[mask]))

    stats = ConversationStats(
        duration_s=duration_s,
        turns_per_min=(len(floor) / minutes) if minutes else 0.0,
        pct_both=both,
        pct_nobody=nobody,
        median_gap_ms=float(np.median(gaps)) if gaps else float("nan"),
        overlapping_turn_changes_per_min=(n_overlap_changes / minutes) if minutes else 0.0,
        backchannels_per_min=(len(backchannels) / minutes) if minutes else 0.0,
        n_floor_turns=len(floor),
        n_turn_changes=len(gaps),
        n_overlapping_turn_changes=n_overlap_changes,
        n_backchannels=len(backchannels),
        gap_ms=gaps,
        backchannel_duration_ms=bc_durations,
        backchannel_rel_position=bc_positions,
        active_rms_left=_active_rms(prep["rms_l"], mask_l),
        active_rms_right=_active_rms(prep["rms_r"], mask_r),
        pct_left=float(np.mean(mask_l)) if n else 0.0,
        pct_right=float(np.mean(mask_r)) if n else 0.0,
    )
    return stats, floor, backchannels


def analyse(left: np.ndarray, right: np.ndarray, sr: int, cfg: VadConfig) -> ConversationStats:
    prep = prepare(left, right, sr, cfg)
    stats, _, _ = analyse_prepared(prep, cfg)
    return stats


def pool(stats_list: list[ConversationStats]) -> dict:
    """Time-weighted corpus numbers. The median gap is the median of every gap, not the median of per-file medians."""
    duration = sum(s.duration_s for s in stats_list)
    minutes = duration / 60.0 if duration else 0.0
    gaps = [g for s in stats_list for g in s.gap_ms]
    bc_d = [d for s in stats_list for d in s.backchannel_duration_ms]
    bc_p = [p for s in stats_list for p in s.backchannel_rel_position]
    both = sum(s.pct_both * s.duration_s for s in stats_list)
    nobody = sum(s.pct_nobody * s.duration_s for s in stats_list)
    turns = sum(s.n_floor_turns for s in stats_list)
    overlaps = sum(s.n_overlapping_turn_changes for s in stats_list)
    bcs = sum(s.n_backchannels for s in stats_list)
    rms_values = [
        v
        for s in stats_list
        for v in (s.active_rms_left, s.active_rms_right)
        if v > 0
    ]
    return {
        "duration_s": duration,
        "hours": duration / 3600.0,
        "n_files": len(stats_list),
        "turns_per_min": (turns / minutes) if minutes else 0.0,
        "pct_both": (both / duration) if duration else 0.0,
        "pct_nobody": (nobody / duration) if duration else 0.0,
        "median_gap_ms": float(np.median(gaps)) if gaps else float("nan"),
        "overlapping_turn_changes_per_min": (overlaps / minutes) if minutes else 0.0,
        "backchannels_per_min": (bcs / minutes) if minutes else 0.0,
        "n_floor_turns": turns,
        "n_turn_changes": len(gaps),
        "n_overlapping_turn_changes": overlaps,
        "n_backchannels": bcs,
        "gap_ms": gaps,
        "backchannel_duration_ms": bc_d,
        "backchannel_rel_position": bc_p,
        "target_active_rms": float(np.median(rms_values)) if rms_values else 0.0,
        "conversation_duration_s": [s.duration_s for s in stats_list],
    }


def _self_test() -> None:
    sr = 16000
    rng = np.random.default_rng(0)

    def burst(seconds: float, amp: float = 0.2) -> np.ndarray:
        return (amp * rng.standard_normal(int(seconds * sr))).astype(np.float32)

    cfg = VadConfig(backchannel_max_s=0.8, intra_turn_pause_ms=300, bridge_ms=40, min_region_ms=40)

    left = np.zeros(int(3.0 * sr), np.float32)
    right = np.zeros_like(left)
    left[: int(1.0 * sr)] = burst(1.0)
    right[int(1.2 * sr) : int(2.2 * sr)] = burst(1.0)
    stats = analyse(left, right, sr, cfg)
    assert stats.n_floor_turns == 2, stats
    assert stats.n_backchannels == 0, stats
    assert abs(stats.median_gap_ms - 200) < 40, stats.median_gap_ms
    assert stats.n_overlapping_turn_changes == 0, stats
    assert stats.pct_both < 0.02, stats.pct_both
    assert 0.25 < stats.pct_nobody < 0.45, stats.pct_nobody

    left = np.zeros(int(2.5 * sr), np.float32)
    right = np.zeros_like(left)
    left[: int(1.5 * sr)] = burst(1.5)
    right[int(1.0 * sr) :] = burst(1.5)
    stats = analyse(left, right, sr, cfg)
    assert stats.n_floor_turns == 2, stats
    assert stats.median_gap_ms < 0, stats.median_gap_ms
    assert abs(stats.median_gap_ms - (-500)) < 50, stats.median_gap_ms
    assert stats.overlapping_turn_changes_per_min > 0
    assert stats.pct_both > 0.1, stats.pct_both

    left = np.zeros(int(5.0 * sr), np.float32)
    right = np.zeros_like(left)
    left[:] = burst(5.0)
    right[int(2.0 * sr) : int(2.35 * sr)] = burst(0.35)
    stats = analyse(left, right, sr, cfg)
    assert stats.n_backchannels == 1, stats
    assert stats.n_floor_turns == 1, stats
    assert stats.backchannels_per_min > 0
    assert 0.2 < stats.backchannel_rel_position[0] < 0.6, stats.backchannel_rel_position

    cutoff, rule = backchannel_cutoff_seconds([0.2, 0.3, 0.25, 0.4, 0.35, 0.3, 0.2, 0.45, 2.5, 3.0, 2.8, 3.2])
    assert cutoff is not None and 0.4 < cutoff < 2.0, (cutoff, rule)
    missing, why = backchannel_cutoff_seconds([0.2, 0.3])
    assert missing is None and why.startswith("no cutoff"), why
    print("activity self-test ok")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        _self_test()
        return 0
    p.error("pass --self-test, or import analyse() from this module")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
