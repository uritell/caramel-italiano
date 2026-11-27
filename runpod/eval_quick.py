#!/usr/bin/env python3
"""Quick look at generated stereo samples.

Timing stats come from timing_score.py. Italian word clarity, when
faster-whisper is installed, is the recogniser's average log-probability
of its own transcript (the number caramel reports; closer to zero is
clearer). Self-play has no reference transcript, so this does not report
a word error rate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from timing_score import closeness, fit_bands, score_file, windows


def _asr(paths: list[Path]) -> dict:
    try:
        from faster_whisper import WhisperModel
    except Exception as exc:
        return {"skipped": f"faster-whisper not installed ({type(exc).__name__})"}
    try:
        import torch
        cuda = torch.cuda.is_available()
    except Exception:
        cuda = False
    device = "cuda" if cuda else "cpu"
    compute = "float16" if cuda else "int8"
    model = WhisperModel("small", device=device, compute_type=compute)
    clips = []
    for path in paths:
        segments, _info = model.transcribe(
            str(path), language="it", beam_size=5, temperature=0.0, vad_filter=False,
        )
        segs = list(segments)
        text = " ".join(s.text.strip() for s in segs).strip()
        weights = [max(0.0, float(s.end - s.start)) for s in segs]
        logs = [float(s.avg_logprob) for s in segs]
        if weights and sum(weights) > 0:
            avg = sum(w * lp for w, lp in zip(weights, logs)) / sum(weights)
        else:
            avg = None
        clips.append({
            "path": str(path),
            "transcript": text,
            "avg_logprob": None if avg is None else round(avg, 4),
            "words": len(text.split()) if text else 0,
        })
        print(f"asr {path.name} logprob={clips[-1]['avg_logprob']} {text[:160]}", flush=True)
    vals = [c["avg_logprob"] for c in clips if c["avg_logprob"] is not None]
    return {
        "model": "faster-whisper small",
        "device": device,
        "language": "it",
        "mean_avg_logprob": None if not vals else round(sum(vals) / len(vals), 4),
        "clips": clips,
    }


def _bands_from(paths: list[Path]) -> dict | None:
    from audioio import read_audio

    rows = []
    for path in paths:
        audio, sr = read_audio(path)
        rows.extend(windows(audio, sr))
    return fit_bands(rows)


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", type=Path, nargs="*", default=[])
    p.add_argument("--real", type=Path, nargs="*", default=[], help="held-out real stereo, scored and used to fit bands")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--skip-asr", action="store_true")
    args = p.parse_args(argv)
    bands = _bands_from(args.real) if args.real else None
    sample_stats = []
    for path in args.samples:
        stats = score_file(path)
        stats["closeness"] = closeness(stats, bands)
        sample_stats.append(stats)
    real_stats = [score_file(path) for path in args.real]
    asr = {"skipped": "not requested"} if args.skip_asr or not args.samples else _asr(args.samples)
    # Word clarity is defined on the generated audio. Also count words on the
    # real clips only when ASR ran; self-play has no reference for WER.
    report = {
        "bands": bands,
        "samples": sample_stats,
        "real": real_stats,
        "asr": asr,
        "note": (
            "Word clarity is the duration-weighted mean of faster-whisper "
            "segment avg_logprob on the generated mix. There is no reference "
            "transcript, so WER is not computed. Closeness is empty until at "
            "least four 60s windows of held-out real stereo exist."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
