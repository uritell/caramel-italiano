#!/usr/bin/env python3
"""Build synthetic two-track Italian conversations from MLS utterances.

Each conversation puts one MLS speaker on the left and a different MLS
speaker on the right. Turns alternate. The gap before the next turn is
drawn from the empirical DataHive gap list (negative gaps are overlaps).
Backchannels, when the measurement found any, are short slices of the
listener's unused audio: duration and position along the other person's
turn are drawn from the measured lists, and the count is Poisson with
the measured rate per minute. Slices start and end near low energy so
the cut is not in the middle of a loud frame. The slice is still read
speech, not a real "mhmm".

FLEURS is not used here. Its public files have a gender and no speaker
id, so a track built from them would change voice every turn. FLEURS
stays mono source material.

Defaults come from synth/timing_config.json, written by score/measure.py.
--hours, --conversation-seconds and --seed override the run, not the
measured distributions.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))

from audio import TARGET_SR, read_audio, resample_mono_like_generate  # noqa: E402
from sources import SOURCES  # noqa: E402

SR = TARGET_SR


def _load_manifest(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _match_rms(x: np.ndarray, target: float) -> np.ndarray:
    if target <= 0:
        return np.ascontiguousarray(x, dtype=np.float32)
    peak = float(np.max(np.abs(x))) + 1e-8
    active = np.abs(x) > max(0.02 * peak, 1e-4)
    if int(np.count_nonzero(active)) < 80:
        rms = float(np.sqrt(np.mean(np.square(x.astype(np.float64))))) + 1e-8
    else:
        rms = float(np.sqrt(np.mean(np.square(x[active].astype(np.float64))))) + 1e-8
    y = x.astype(np.float32) * np.float32(target / rms)
    peak = float(np.max(np.abs(y))) + 1e-8
    if peak > 0.98:
        y = y * np.float32(0.98 / peak)
    return np.ascontiguousarray(y, dtype=np.float32)


def _load_clip(raw: Path, row: dict, target_rms: float) -> np.ndarray:
    audio, sr = read_audio(raw / row["path"])
    mono = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]
    y = resample_mono_like_generate(mono, sr, SR)
    return _match_rms(y, target_rms)


def _take(rows: list[dict], raw: Path, seconds: float, target_rms: float):
    clips = []
    used = []
    total = 0.0
    consumed = 0
    for row in rows:
        consumed += 1
        clip = _load_clip(raw, row, target_rms)
        if clip.size < int(0.4 * SR):
            continue
        clips.append(clip)
        used.append(row)
        total += clip.size / SR
        if total >= seconds:
            break
    return clips, used, total, consumed


def _slice_low_energy(audio: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if audio.size <= n:
        return audio
    hop = SR // 100
    n_starts = (audio.size - n) // hop
    if n_starts <= 1:
        return audio[:n]
    edge = hop

    def energy(i: int) -> float:
        a = max(0, i - edge)
        b = min(audio.size, i + edge)
        return float(np.sqrt(np.mean(np.square(audio[a:b].astype(np.float64)))))

    # Score a subsample of starts when the reservoir is long.
    step = 1 if n_starts <= 400 else int(np.ceil(n_starts / 400))
    indexes = np.arange(0, n_starts, step)
    scores = np.array([energy(int(i) * hop) + energy(int(i) * hop + n) for i in indexes])
    k = max(1, indexes.size // 10)
    best = indexes[np.argpartition(scores, k)[:k]]
    start = int(best[int(rng.integers(0, best.size))]) * hop
    return audio[start : start + n]


def _place(buffer: np.ndarray, start: int, clip: np.ndarray) -> tuple[np.ndarray, int, int, bool]:
    """Write clip at start. If that span is already used, shift forward. Returns buffer, start, end, clamped."""
    clamped = False
    if start < 0:
        start = 0
        clamped = True
    end = start + clip.size
    if end > buffer.size:
        grown = np.zeros(end + SR, dtype=np.float32)
        grown[: buffer.size] = buffer
        buffer = grown
    span = buffer[start:end]
    if span.size and float(np.max(np.abs(span))) > 1e-4:
        occupied = np.flatnonzero(np.abs(buffer[:end]) > 1e-4)
        if occupied.size:
            start = int(occupied[-1]) + 1
            end = start + clip.size
            clamped = True
            if end > buffer.size:
                grown = np.zeros(end + SR, dtype=np.float32)
                grown[: buffer.size] = buffer
                buffer = grown
    buffer[start:end] += clip
    return buffer, start, end, clamped


def _build_one(
    clips_a: list[np.ndarray],
    clips_b: list[np.ndarray],
    reservoir_a: np.ndarray,
    reservoir_b: np.ndarray,
    empirical: dict,
    rng: np.random.Generator,
    target_s: float,
) -> dict:
    gaps = np.asarray(empirical["gap_ms"], dtype=np.float64)
    if gaps.size == 0:
        raise RuntimeError("timing config has no measured gaps")
    bc_durations = np.asarray(empirical.get("backchannel_duration_ms") or [], dtype=np.float64)
    bc_pos = np.asarray(empirical.get("backchannel_rel_position") or [], dtype=np.float64)
    bc_rate = float(empirical.get("backchannels_per_min") or 0.0)
    use_bc = bc_durations.size > 0 and bc_pos.size > 0 and bc_rate > 0 and reservoir_a.size > SR and reservoir_b.size > SR

    length = int((target_s + 30.0) * SR)
    left = np.zeros(length, np.float32)
    right = np.zeros(length, np.float32)
    buffers = [left, right]
    pools = [clips_a, clips_b]
    index = [0, 0]
    side = 0
    t = 0.0
    turns = []
    n_clamped = 0
    while t < target_s and index[0] < len(pools[0]) and index[1] < len(pools[1]):
        clip = pools[side][index[side]]
        index[side] += 1
        start_s = max(0.0, t)
        if start_s >= target_s and len(turns) >= 4:
            break
        buf, start_i, end_i, clamped = _place(buffers[side], int(round(start_s * SR)), clip)
        buffers[side] = buf
        if clamped:
            n_clamped += 1
        start_s = start_i / SR
        end_s = end_i / SR
        turns.append(
            {
                "kind": "turn",
                "side": "left" if side == 0 else "right",
                "start_s": start_s,
                "end_s": end_s,
            }
        )
        gap_s = float(gaps[int(rng.integers(0, gaps.size))]) / 1000.0
        t = end_s + gap_s
        side = 1 - side
        if len(turns) > 4000:
            break

    # Trim to the last sample either channel used, but not past the target
    # by more than one turn (the last turn is allowed to finish).
    used = 0
    for buf in buffers:
        nz = np.flatnonzero(np.abs(buf) > 1e-5)
        if nz.size:
            used = max(used, int(nz[-1]) + 1)
    used = max(used, 1)

    def _fit(buf: np.ndarray, n: int) -> np.ndarray:
        if buf.size >= n:
            return buf
        grown = np.zeros(n, dtype=np.float32)
        grown[: buf.size] = buf
        return grown

    width = max(buffers[0].size, buffers[1].size, used)
    buffers = [_fit(buffers[0], width), _fit(buffers[1], width)]

    backchannels = []
    if use_bc:
        minutes = (used / SR) / 60.0
        n_bc = int(rng.poisson(bc_rate * minutes))
        weights = np.array([max(0.0, ev["end_s"] - ev["start_s"]) for ev in turns], dtype=np.float64)
        if weights.sum() <= 0 or n_bc == 0:
            n_bc = 0
        else:
            weights /= weights.sum()
        reservoirs = [reservoir_a, reservoir_b]
        for _ in range(n_bc):
            turn = turns[int(rng.choice(len(turns), p=weights))]
            listener = 0 if turn["side"] == "right" else 1
            dur_s = float(bc_durations[int(rng.integers(0, bc_durations.size))]) / 1000.0
            if dur_s < 0.05 or dur_s > max(0.05, turn["end_s"] - turn["start_s"]):
                continue
            rel = float(bc_pos[int(rng.integers(0, bc_pos.size))])
            room = (turn["end_s"] - turn["start_s"]) - dur_s
            start_s = turn["start_s"] + rel * max(0.0, room)
            n = int(round(dur_s * SR))
            snippet = _slice_low_energy(reservoirs[listener], n, rng)
            start_i = int(round(start_s * SR))
            end_i = start_i + snippet.size
            if end_i > buffers[listener].size:
                continue
            if float(np.max(np.abs(buffers[listener][start_i:end_i]))) > 1e-4:
                continue
            buffers[listener][start_i:end_i] += snippet
            backchannels.append(
                {
                    "kind": "backchannel",
                    "side": "left" if listener == 0 else "right",
                    "start_s": start_i / SR,
                    "end_s": end_i / SR,
                    "during_side": turn["side"],
                }
            )
        used_bc = 0
        for buf in buffers:
            nz = np.flatnonzero(np.abs(buf) > 1e-5)
            if nz.size:
                used_bc = max(used_bc, int(nz[-1]) + 1)
        used = max(used, used_bc)

    left = buffers[0][:used]
    right = buffers[1][:used]
    stereo = np.stack([left, right], axis=1)
    peak = float(np.max(np.abs(stereo))) if stereo.size else 0.0
    if peak > 0.99:
        stereo = stereo * np.float32(0.99 / peak)
    return {
        "audio": stereo,
        "turns": turns + backchannels,
        "n_turns": len(turns),
        "n_backchannels": len(backchannels),
        "n_gap_clamps": n_clamped,
        "duration_s": used / SR,
        "backchannels_drawn": use_bc,
    }


def _plan_lengths(empirical: dict, hours: float, conversation_seconds: float | None) -> list[float]:
    target = hours * 3600.0
    if conversation_seconds is not None:
        lengths = []
        left = target
        while left > 15:
            this = min(conversation_seconds, left)
            if left < conversation_seconds * 1.25:
                this = left
            lengths.append(this)
            left -= this
            if len(lengths) > 50:
                break
        return lengths
    durations = [float(x) for x in empirical.get("conversation_duration_s") or []]
    if not durations:
        raise RuntimeError("timing config has no measured conversation lengths; pass --conversation-seconds")
    median = float(np.median(durations))
    lengths = []
    left = target
    while left > 15:
        this = median if left > median * 1.25 else left
        lengths.append(float(this))
        left -= this
        if len(lengths) > 50:
            break
    return lengths


def _speakers(rows: list[dict]) -> list[tuple[str, list[dict]]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        if row.get("source") != "mls_it" or row.get("split") != "train":
            continue
        if not row.get("speaker_id"):
            continue
        grouped.setdefault(str(row["speaker_id"]), []).append(row)
    for clips in grouped.values():
        clips.sort(key=lambda r: r["path"])
    ranked = sorted(grouped.items(), key=lambda kv: -sum(float(r["duration_s"]) for r in kv[1]))
    return ranked


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=ROOT / "data" / "work" / "manifest.jsonl")
    p.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    p.add_argument("--config", type=Path, default=ROOT / "synth" / "timing_config.json")
    p.add_argument("--out", type=Path, default=ROOT / "synth" / "out")
    p.add_argument("--hours", type=float, default=1.0)
    p.add_argument("--conversation-seconds", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    document = json.loads(args.config.read_text(encoding="utf-8"))
    empirical = document["empirical"]
    target_rms = float(empirical.get("target_active_rms") or 0.0)
    lengths = _plan_lengths(empirical, args.hours, args.conversation_seconds)
    ranked = _speakers(_load_manifest(args.manifest))
    if len(ranked) < 2:
        print("need at least two MLS train speakers in the manifest", file=sys.stderr)
        return 1
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    conv_dir = args.out / "conversations"
    conv_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    cursor = {sid: 0 for sid, _ in ranked}
    pair_i = 0
    licence = SOURCES["mls_it"]
    for n, target_s in enumerate(lengths):
        # Each speaker needs a bit over half the timeline, plus a reservoir.
        need = target_s * 0.65 + 20.0
        chosen = None
        for _try in range(len(ranked)):
            a_id, a_rows = ranked[(pair_i) % len(ranked)]
            b_id, b_rows = ranked[(pair_i + 1) % len(ranked)]
            pair_i += 2
            if a_id == b_id:
                continue
            a_left = sum(float(r["duration_s"]) for r in a_rows[cursor[a_id] :])
            b_left = sum(float(r["duration_s"]) for r in b_rows[cursor[b_id] :])
            if a_left >= need and b_left >= need:
                chosen = (a_id, a_rows, b_id, b_rows)
                break
        if chosen is None:
            print(f"stopping: not enough unused MLS audio for a {target_s:.0f}s conversation", flush=True)
            break
        a_id, a_rows, b_id, b_rows = chosen
        print(f"conversation {n:03d}: speakers {a_id} / {b_id}, target {target_s:.0f}s", flush=True)
        a_turn, a_used, _, a_consumed = _take(a_rows[cursor[a_id] :], args.raw, target_s * 0.7, target_rms)
        b_turn, b_used, _, b_consumed = _take(b_rows[cursor[b_id] :], args.raw, target_s * 0.7, target_rms)
        cursor[a_id] += a_consumed
        cursor[b_id] += b_consumed
        a_res_clips, a_res_used, _, a_res_n = _take(a_rows[cursor[a_id] :], args.raw, 40.0, target_rms)
        b_res_clips, b_res_used, _, b_res_n = _take(b_rows[cursor[b_id] :], args.raw, 40.0, target_rms)
        cursor[a_id] += a_res_n
        cursor[b_id] += b_res_n
        reservoir_a = np.concatenate(a_res_clips) if a_res_clips else np.zeros(0, np.float32)
        reservoir_b = np.concatenate(b_res_clips) if b_res_clips else np.zeros(0, np.float32)
        if len(a_turn) < 2 or len(b_turn) < 2:
            print(f"skip conversation {n}: not enough clips", flush=True)
            continue
        built = _build_one(a_turn, b_turn, reservoir_a, reservoir_b, empirical, rng, target_s)
        import soundfile as sf

        wav_path = conv_dir / f"conv_{n:03d}.wav"
        sf.write(str(wav_path), built["audio"], SR, subtype="PCM_16")
        side = {
            "id": f"synth_mls/{wav_path.stem}",
            "source": "mls_it",
            "derived_from": "mls_it train, two speakers, synthetic timing",
            "licence": licence["licence"],
            "licence_url": licence["licence_url"],
            "attribution": licence["attribution"],
            "source_url": licence["url"],
            "split": "synthetic",
            "path": str(wav_path),
            "duration_s": built["duration_s"],
            "sample_rate": SR,
            "channels": 2,
            "layout": "stereo_speaker_a_left_speaker_b_right",
            "speaker_id_left": a_id,
            "speaker_id_right": b_id,
            "n_turns": built["n_turns"],
            "n_backchannels": built["n_backchannels"],
            "n_gap_clamps": built["n_gap_clamps"],
            "target_s": target_s,
            "timing_config": str(args.config),
            "seed": args.seed,
            "clips_left": [r["id"] for r in a_used],
            "clips_right": [r["id"] for r in b_used],
            "events": built["turns"],
        }
        (conv_dir / f"conv_{n:03d}.json").write_text(json.dumps(side, indent=2) + "\n", encoding="utf-8")
        manifest_rows.append({k: v for k, v in side.items() if k not in {"events", "clips_left", "clips_right"}})
        print(
            f"  wrote {wav_path.name}: {built['duration_s']:.1f}s, "
            f"{built['n_turns']} turns, {built['n_backchannels']} backchannels, "
            f"{built['n_gap_clamps']} gap clamps",
            flush=True,
        )
    man_path = args.out / "manifest.jsonl"
    with man_path.open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row) + "\n")
    total = sum(r["duration_s"] for r in manifest_rows)
    print(f"synthetic total {total / 3600:.3f} h across {len(manifest_rows)} files -> {man_path}", flush=True)
    return 0 if manifest_rows else 1


if __name__ == "__main__":
    sys.exit(main())
