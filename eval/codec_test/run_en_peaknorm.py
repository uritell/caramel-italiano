#!/usr/bin/env python3
"""Re-run the English baseline after peak-normalizing each clip to 0.9.

The raw FLEURS en_us test wavs have wildly inconsistent levels (36/80 peaks below
0.01; Italian peaks are all >= 0.16). An unnormalized English run is not a fair
language comparison. This keeps the same 80 clips, seed, Mimi, and whisper-small.
"""
from __future__ import annotations

import gc, json, math, os, sys, time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_followup as fu
import run_codec_test as it

PEAK_TARGET = 0.9
EN_ROOT = fu.EN_ROOT
PN = EN_ROOT / "peaknorm"
ORIG = PN / "originals_24k"
Q8 = PN / "recon_q8_24k"
Q32 = PN / "recon_q32_24k"
ASR_PATH = PN / "asr_results.json"
os.environ.setdefault("HF_HOME", str(it.HF_HOME))


def peak_norm(x, target=PEAK_TARGET):
    peak = float(np.max(np.abs(x)))
    if peak < 1e-8:
        return x.copy(), 1.0, peak
    gain = target / peak
    return np.ascontiguousarray(x * gain, dtype=np.float32), float(gain), peak


def main():
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 8)))
    torch.set_grad_enabled(False)
    man = json.loads((EN_ROOT / "manifest.json").read_text())
    clips = []
    for d in man["clips"]:
        c = it.Clip(**{k: d[k] for k in it.Clip.__dataclass_fields__})
        clips.append(c)
    src_dir = EN_ROOT / "originals_24k"
    ORIG.mkdir(parents=True, exist_ok=True)
    orig_audio = {}
    gains = {}
    for i, c in enumerate(clips, 1):
        x, sr = it.load_mono(src_dir / f"{c.id}.wav")
        assert sr == it.SR
        y, gain, peak = peak_norm(x)
        gains[c.id] = {"raw_peak": peak, "gain": gain, "norm_peak": float(np.max(np.abs(y)))}
        it.save_wav(ORIG / f"{c.id}.wav", y)
        orig_audio[c.id] = y
        if i % 20 == 0 or i == len(clips):
            print(f"Peak-normed {i}/{len(clips)}", flush=True)
    (PN / "gains.json").write_text(json.dumps(gains, indent=2))
    print(
        f"Gains: median {np.median([g['gain'] for g in gains.values()]):.1f}, "
        f"n_gain>2 {sum(1 for g in gains.values() if g['gain']>2)}, "
        f"n_gain>20 {sum(1 for g in gains.values() if g['gain']>20)}",
        flush=True,
    )

    q8_meta, q32_meta, timing = fu.run_mimi(
        clips, orig_audio, Q8, Q32,
        PN / "q8_audio_metrics.json",
        PN / "q32_audio_metrics.json",
        PN / "timing.json",
    )
    asr = fu.load_asr("small", "en")
    store = {}
    if ASR_PATH.exists():
        try:
            store = json.loads(ASR_PATH.read_text())
        except json.JSONDecodeError:
            store = {}
    q32_clips = [c for c in clips if c.q32]
    t0 = time.perf_counter()
    for i, c in enumerate(clips, 1):
        rec = fu.cached_transcribe(asr, store, ASR_PATH, f"orig:{c.id}", ORIG / f"{c.id}.wav", "en")
        print(f"[pn orig {i}/{len(clips)}] lp={rec['avg_logprob']:.3f} {rec['text'][:80]!r}", flush=True)
    for i, c in enumerate(clips, 1):
        rec = fu.cached_transcribe(asr, store, ASR_PATH, f"q8:{c.id}", Q8 / f"{c.id}.wav", "en")
        print(f"[pn q8 {i}/{len(clips)}] lp={rec['avg_logprob']:.3f} {rec['text'][:80]!r}", flush=True)
    for i, c in enumerate(q32_clips, 1):
        rec = fu.cached_transcribe(asr, store, ASR_PATH, f"q32:{c.id}", Q32 / f"{c.id}.wav", "en")
        print(f"[pn q32 {i}/{len(q32_clips)}] lp={rec['avg_logprob']:.3f} {rec['text'][:80]!r}", flush=True)
    print(f"peak-norm English ASR wall {time.perf_counter()-t0:.1f}s", flush=True)
    del asr
    gc.collect()

    metrics = {
        "peak_target": PEAK_TARGET,
        "original": fu.pack_condition(clips, store, lambda c: f"orig:{c.id}"),
        "q8": fu.pack_condition(clips, store, lambda c: f"q8:{c.id}", q8_meta),
        "q32_vs_q8": fu.pack_q32_compare(q32_clips, store, q8_meta, q32_meta, timing),
    }
    sens, dropped = fu.corpus_excluding_collapses(clips, store, lambda c: f"orig:{c.id}", lambda c: f"q8:{c.id}")
    if sens:
        metrics["exclude_collapses"] = {**sens, "n_dropped": dropped}
    # median deltas
    o = {r["id"]: r for r in metrics["original"]["per_utt"]}
    deltas = [metrics["q8"]["per_utt"][i]["wer"] - o[metrics["q8"]["per_utt"][i]["id"]]["wer"] for i in range(80)]
    metrics["median_delta_wer"] = float(np.median(deltas))
    metrics["mean_delta_wer"] = float(np.mean(deltas))
    (PN / "metrics.json").write_text(json.dumps(fu.sanitise(metrics), ensure_ascii=False, indent=2))
    (PN / "timing.json").write_text(json.dumps(timing, indent=2))
    print("EN peak-norm orig", metrics["original"]["corpus"])
    print("EN peak-norm q8", metrics["q8"]["corpus"], "sdr", metrics["q8"]["mean_si_sdr"])
    print("delta WER", metrics["q8"]["corpus"]["wer"] - metrics["original"]["corpus"]["wer"],
          "median", metrics["median_delta_wer"])
    print("32q", {k: metrics["q32_vs_q8"][k]["corpus"] if k!="per_utt" and k!="audio_s" else metrics["q32_vs_q8"][k]
                  for k in metrics["q32_vs_q8"] if k in ("original","q8","q32","audio_s")})
    print("DONE_PEAKNORM", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
