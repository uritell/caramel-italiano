#!/usr/bin/env python3
"""CPU-only English FLEURS baseline + stronger ASR on the existing Italian 8q set.

Does not overwrite Italian 8q/32q wavs or the original report body; appends a section.
No GPU, no accounts, no spending.
"""
from __future__ import annotations

import gc
import json
import math
import os
import random
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_codec_test as it  # noqa: E402

ROOT = it.ROOT
EN_ROOT = ROOT / "en_us"
EN_RAW_WAVS = ROOT / "raw" / "fleurs_en_us_test_wavs"
EN_TSV = ROOT / "raw" / "fleurs_en_us_test.tsv"
EN_TAR = ROOT / "raw" / "fleurs_en_us_test.tar.gz"
EN_ORIG = EN_ROOT / "originals_24k"
EN_Q8 = EN_ROOT / "recon_q8_24k"
EN_Q32 = EN_ROOT / "recon_q32_24k"

N_MALE = 38
N_FEMALE = 42
SEED = 42
TURBO_NAME = "large-v3-turbo"
TURBO_ASR_PATH = ROOT / "asr_italian_large_v3_turbo.json"
EN_ASR_PATH = EN_ROOT / "asr_results.json"
REPORT_PATH = ROOT / "report.md"

os.environ.setdefault("HF_HOME", str(it.HF_HOME))
os.environ.setdefault("TRANSFORMERS_CACHE", str(it.HF_HOME / "transformers"))
os.environ.setdefault("HF_HUB_CACHE", str(it.HF_HOME / "hub"))


def select_en_clips(all_clips, n_male=N_MALE, n_female=N_FEMALE, seed=SEED):
    """Match the Italian sample: seed 42, 38 male + 42 female, 5–25 s, unique sentences."""
    rng = random.Random(seed)
    eligible = [c for c in all_clips if 5.0 <= c.duration_s <= 25.0]
    males = [c for c in eligible if c.gender.upper() == "MALE"]
    females = [c for c in eligible if c.gender.upper() == "FEMALE"]
    if len(males) < n_male:
        raise RuntimeError(f"Only {len(males)} eligible English male clips, need {n_male}")
    if len(females) < n_female:
        raise RuntimeError(f"Only {len(females)} eligible English female clips, need {n_female}")
    selected = it.unique_sentence_sample(males, n_male, rng) + it.unique_sentence_sample(females, n_female, rng)
    selected.sort(key=lambda c: c.id)
    by_dur = sorted(selected, key=lambda c: (c.duration_s, c.id))
    step = max(len(by_dur) // it.N_Q32, 1)
    q32_ids = {by_dur[min(i * step, len(by_dur) - 1)].id for i in range(it.N_Q32)}
    if len(q32_ids) < it.N_Q32:
        for c in by_dur:
            q32_ids.add(c.id)
            if len(q32_ids) == it.N_Q32:
                break
    for c in selected:
        c.q32 = c.id in q32_ids
        c.source = "google/fleurs en_us test (FLEURS-102 archive on Hugging Face)"
        c.license = "CC BY 4.0"
        c.split = "test"
    return selected


def load_or_resample(clips, raw_dir, orig_dir):
    orig_dir.mkdir(parents=True, exist_ok=True)
    audio = {}
    for i, c in enumerate(clips, 1):
        out = orig_dir / f"{c.id}.wav"
        src = raw_dir / c.filename
        x, sr = it.load_mono(src)
        y = it.to_24k_mono(x, sr)
        if not out.exists() or out.stat().st_size == 0:
            it.save_wav(out, y)
        audio[c.id] = y
        if i % 20 == 0 or i == len(clips):
            print(f"Resampled {i}/{len(clips)} English originals to 24 kHz", flush=True)
    return audio


def run_mimi(clips, orig_audio, q8_dir, q32_dir, q8_path, q32_path, timing_path):
    q8_dir.mkdir(parents=True, exist_ok=True)
    q32_dir.mkdir(parents=True, exist_ok=True)
    q32_clips = [c for c in clips if c.q32]
    have_q8 = all((q8_dir / f"{c.id}.wav").exists() for c in clips) and q8_path.exists()
    have_q32 = all((q32_dir / f"{c.id}.wav").exists() for c in q32_clips) and q32_path.exists()
    if have_q8 and have_q32:
        print("English reconstructions already on disk; skipping Mimi.", flush=True)
        q8_meta = json.loads(q8_path.read_text())
        q32_meta = json.loads(q32_path.read_text())
        timing = json.loads(timing_path.read_text()) if timing_path.exists() else {}
        return q8_meta, q32_meta, timing

    mimi = it.load_mimi()
    timing = {
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "n_threads": torch.get_num_threads(),
        "mimi_params": int(sum(p.numel() for p in mimi.parameters())),
    }
    enc_s = dec_s = audio_s = 0.0
    q8_meta = {}
    for i, c in enumerate(clips, 1):
        y = orig_audio[c.id]
        recon, code_shape, t_enc, t_dec = it.mimi_roundtrip(mimi, y, 8)
        enc_s += t_enc
        dec_s += t_dec
        audio_s += len(y) / it.SR
        it.save_wav(q8_dir / f"{c.id}.wav", recon)
        n = min(len(y), len(recon))
        q8_meta[c.id] = {
            "encode_s": t_enc,
            "decode_s": t_dec,
            "code_shape": list(code_shape),
            "si_sdr": it.si_sdr(recon[:n], y[:n]),
            "peak_snr_db": it.peak_snr(recon[:n], y[:n]),
        }
        print(
            f"[en 8q {i}/{len(clips)}] {c.id} {len(y)/it.SR:.2f}s  "
            f"RTF {(t_enc+t_dec)/(len(y)/it.SR):.2f}x  SI-SDR {q8_meta[c.id]['si_sdr']:.2f} dB",
            flush=True,
        )
    timing["q8"] = {
        "encode_s": enc_s,
        "decode_s": dec_s,
        "audio_s": audio_s,
        "encode_rtf": enc_s / audio_s,
        "decode_rtf": dec_s / audio_s,
        "roundtrip_rtf": (enc_s + dec_s) / audio_s,
    }

    enc32 = dec32 = aud32 = 0.0
    q32_meta = {}
    for i, c in enumerate(q32_clips, 1):
        y = orig_audio[c.id]
        recon, code_shape, t_enc, t_dec = it.mimi_roundtrip(mimi, y, 32)
        enc32 += t_enc
        dec32 += t_dec
        aud32 += len(y) / it.SR
        it.save_wav(q32_dir / f"{c.id}.wav", recon)
        n = min(len(y), len(recon))
        q32_meta[c.id] = {
            "encode_s": t_enc,
            "decode_s": t_dec,
            "code_shape": list(code_shape),
            "si_sdr": it.si_sdr(recon[:n], y[:n]),
            "peak_snr_db": it.peak_snr(recon[:n], y[:n]),
        }
        print(
            f"[en 32q {i}/{len(q32_clips)}] {c.id} {len(y)/it.SR:.2f}s  "
            f"RTF {(t_enc+t_dec)/(len(y)/it.SR):.2f}x  SI-SDR {q32_meta[c.id]['si_sdr']:.2f} dB",
            flush=True,
        )
    timing["q32"] = {
        "encode_s": enc32,
        "decode_s": dec32,
        "audio_s": aud32,
        "encode_rtf": enc32 / aud32 if aud32 else float("nan"),
        "decode_rtf": dec32 / aud32 if aud32 else float("nan"),
        "roundtrip_rtf": (enc32 + dec32) / aud32 if aud32 else float("nan"),
        "n": len(q32_clips),
    }
    del mimi
    gc.collect()
    q8_path.write_text(json.dumps(q8_meta, indent=2))
    q32_path.write_text(json.dumps(q32_meta, indent=2))
    timing_path.write_text(json.dumps(timing, indent=2))
    print("English Mimi unloaded.", flush=True)
    return q8_meta, q32_meta, timing


def load_asr(name, language_label):
    from faster_whisper import WhisperModel

    cache = str(it.HF_HOME / "faster-whisper")
    Path(cache).mkdir(parents=True, exist_ok=True)
    n_threads = max(1, min(8, os.cpu_count() or 8))
    print(f"Loading faster-whisper '{name}' int8 CPU ({n_threads} threads) for {language_label}…", flush=True)
    t0 = time.perf_counter()
    model = WhisperModel(
        name,
        device="cpu",
        compute_type="int8",
        cpu_threads=n_threads,
        download_root=cache,
    )
    print(f"ASR '{name}' loaded in {time.perf_counter()-t0:.1f}s", flush=True)
    return model


def transcribe_lang(model, path, language):
    # Temporarily swap language without editing the shared helper more than needed.
    x, sr = it.load_mono(path)
    if sr != 16000:
        import soxr

        x = soxr.resample(x, sr, 16000, quality="VHQ")
    x = np.ascontiguousarray(x, dtype=np.float32)
    segments, info = model.transcribe(
        x,
        language=language,
        beam_size=5,
        vad_filter=False,
        condition_on_previous_text=False,
        temperature=0.0,
        word_timestamps=False,
    )
    segs = list(segments)
    text = " ".join(s.text.strip() for s in segs if s.text).strip()
    dur = 0.0
    wsum = 0.0
    for s in segs:
        d = max(float(s.end) - float(s.start), 1e-3)
        dur += d
        wsum += float(s.avg_logprob) * d
    avg_lp = (wsum / dur) if dur > 0 else float("nan")
    return {"text": text, "avg_logprob": avg_lp, "n_segments": len(segs)}


def cached_transcribe(model, store, store_path, key, wav_path, language):
    if key in store and store[key].get("text") is not None:
        return store[key]
    rec = transcribe_lang(model, wav_path, language)
    rec["path"] = str(wav_path)
    store[key] = rec
    store_path.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
    return rec


def pack_condition(clips, asr_map, key_fn, sdr_map=None):
    per = []
    pairs = []
    for c in clips:
        rec = asr_map[key_fn(c)]
        sc = it.utterance_scores(c.transcription, rec["text"])
        row = {
            "id": c.id,
            "gender": c.gender,
            "duration_s": c.duration_s,
            "wer": sc["wer"],
            "cer": sc["cer"],
            "avg_logprob": rec["avg_logprob"],
            "ref_norm": sc["ref_norm"],
            "hyp_norm": sc["hyp_norm"],
            "hyp_raw": rec["text"],
        }
        if sdr_map is not None:
            row["si_sdr"] = sdr_map[c.id]["si_sdr"]
        per.append(row)
        pairs.append((c.transcription, rec["text"]))
    return {
        "corpus": it.corpus_wer_cer(pairs),
        "macro": {
            "wer": it.mean_ignore_nan(p["wer"] for p in per),
            "cer": it.mean_ignore_nan(p["cer"] for p in per),
        },
        "mean_avg_logprob": it.mean_ignore_nan(p["avg_logprob"] for p in per),
        "mean_si_sdr": it.mean_ignore_nan(p.get("si_sdr", float("nan")) for p in per) if sdr_map is not None else None,
        "per_utt": per,
    }


def pack_q32_compare(clips_q32, asr_map, q8_meta, q32_meta, timing):
    per = []
    pairs_o, pairs_8, pairs_32 = [], [], []
    for c in clips_q32:
        o = asr_map[f"orig:{c.id}"]
        a8 = asr_map[f"q8:{c.id}"]
        a32 = asr_map[f"q32:{c.id}"]
        so = it.utterance_scores(c.transcription, o["text"])
        s8 = it.utterance_scores(c.transcription, a8["text"])
        s32 = it.utterance_scores(c.transcription, a32["text"])
        pairs_o.append((c.transcription, o["text"]))
        pairs_8.append((c.transcription, a8["text"]))
        pairs_32.append((c.transcription, a32["text"]))
        per.append(
            {
                "id": c.id,
                "gender": c.gender,
                "duration_s": c.duration_s,
                "wer_orig": so["wer"],
                "wer_q8": s8["wer"],
                "wer_q32": s32["wer"],
                "cer_orig": so["cer"],
                "cer_q8": s8["cer"],
                "cer_q32": s32["cer"],
                "avg_logprob_orig": o["avg_logprob"],
                "avg_logprob_q8": a8["avg_logprob"],
                "avg_logprob_q32": a32["avg_logprob"],
                "si_sdr_q8": q8_meta[c.id]["si_sdr"],
                "si_sdr_q32": q32_meta[c.id]["si_sdr"],
            }
        )
    enc8 = sum(q8_meta[c.id]["encode_s"] for c in clips_q32)
    dec8 = sum(q8_meta[c.id]["decode_s"] for c in clips_q32)
    # audio duration from timing is overall; use clip durations
    aud8 = sum(c.duration_s for c in clips_q32)
    return {
        "audio_s": aud8,
        "original": {
            "corpus": it.corpus_wer_cer(pairs_o),
            "macro": {
                "wer": it.mean_ignore_nan(p["wer_orig"] for p in per),
                "cer": it.mean_ignore_nan(p["cer_orig"] for p in per),
            },
            "mean_avg_logprob": it.mean_ignore_nan(p["avg_logprob_orig"] for p in per),
        },
        "q8": {
            "corpus": it.corpus_wer_cer(pairs_8),
            "macro": {
                "wer": it.mean_ignore_nan(p["wer_q8"] for p in per),
                "cer": it.mean_ignore_nan(p["cer_q8"] for p in per),
            },
            "mean_avg_logprob": it.mean_ignore_nan(p["avg_logprob_q8"] for p in per),
            "mean_si_sdr": it.mean_ignore_nan(p["si_sdr_q8"] for p in per),
            "encode_rtf": enc8 / aud8 if aud8 else float("nan"),
            "decode_rtf": dec8 / aud8 if aud8 else float("nan"),
        },
        "q32": {
            "corpus": it.corpus_wer_cer(pairs_32),
            "macro": {
                "wer": it.mean_ignore_nan(p["wer_q32"] for p in per),
                "cer": it.mean_ignore_nan(p["cer_q32"] for p in per),
            },
            "mean_avg_logprob": it.mean_ignore_nan(p["avg_logprob_q32"] for p in per),
            "mean_si_sdr": it.mean_ignore_nan(p["si_sdr_q32"] for p in per),
            "encode_rtf": timing.get("q32", {}).get("encode_rtf"),
            "decode_rtf": timing.get("q32", {}).get("decode_rtf"),
        },
        "per_utt": per,
    }


def n_collapses(per_q8, per_orig=None):
    """Count clips where 8q ASR is empty, almost empty, WER>=1, or hugely worse."""
    n_empty = 0
    n_wer1 = 0
    for r in per_q8:
        hyp = (r.get("hyp_norm") or "").strip()
        if not hyp or len(hyp.split()) <= 1:
            n_empty += 1
        if r["wer"] >= 1.0:
            n_wer1 += 1
    return n_empty, n_wer1


def corpus_excluding_collapses(clips, asr_map, orig_key, q8_key):
    keep = []
    for c in clips:
        q = asr_map[q8_key(c)]
        hyp = it.normalize_text(q["text"])
        sc = it.utterance_scores(c.transcription, q["text"])
        if sc["wer"] >= 1.0 or (not hyp) or len(hyp.split()) <= 1:
            continue
        keep.append(c)
    if not keep:
        return None, 0
    pairs_o = [(c.transcription, asr_map[orig_key(c)]["text"]) for c in keep]
    pairs_8 = [(c.transcription, asr_map[q8_key(c)]["text"]) for c in keep]
    return {"original": it.corpus_wer_cer(pairs_o), "q8": it.corpus_wer_cer(pairs_8), "n": len(keep)}, len(clips) - len(keep)


def sanitise(obj):
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: sanitise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitise(v) for v in obj]
    return obj


def pct(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{100.0 * x:.2f}%"


def fnum(x, nd=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def append_report(en_metrics, en_manifest, en_timing, it_small, it_turbo, turbo_meta):
    it_o = it_small["original"]
    it_8 = it_small["q8"]
    en_o = en_metrics["original"]
    en_8 = en_metrics["q8"]
    d_it_wer = it_8["corpus"]["wer"] - it_o["corpus"]["wer"]
    d_en_wer = en_8["corpus"]["wer"] - en_o["corpus"]["wer"]
    d_it_cer = it_8["corpus"]["cer"] - it_o["corpus"]["cer"]
    d_en_cer = en_8["corpus"]["cer"] - en_o["corpus"]["cer"]
    d_it_lp = it_8["mean_avg_logprob"] - it_o["mean_avg_logprob"]
    d_en_lp = en_8["mean_avg_logprob"] - en_o["mean_avg_logprob"]

    # "Meaningfully worse": Italian absolute WER rise larger by >= 5 points,
    # or ~1.5x the English rise, consistently across WER/CER and the 8-vs-32 gap.
    wer_gap = d_it_wer - d_en_wer
    if d_en_wer <= 0 and d_it_wer > 0.05:
        worse = "YES"
        worse_txt = (
            f"Italian 8q degradation is meaningfully worse than English: Italian WER rose "
            f"{d_it_wer*100:.2f} points while English did not rise (or improved)."
        )
    elif wer_gap >= 0.05 or (d_en_wer > 1e-6 and d_it_wer >= 1.5 * d_en_wer):
        worse = "YES"
        worse_txt = (
            f"Italian 8q degradation is meaningfully worse than English. Absolute corpus WER rise "
            f"is {d_it_wer*100:.2f} points (IT) vs {d_en_wer*100:.2f} points (EN), a gap of "
            f"{wer_gap*100:.2f} points; CER rise {d_it_cer*100:.2f} vs {d_en_cer*100:.2f} points."
        )
    elif wer_gap >= 0.02:
        worse = "SLIGHTLY"
        worse_txt = (
            f"Italian 8q degradation is only modestly worse than English "
            f"(+{d_it_wer*100:.2f} vs +{d_en_wer*100:.2f} WER points). The difference is visible but not large."
        )
    else:
        worse = "NO"
        worse_txt = (
            f"Italian 8q degradation is not meaningfully worse than English "
            f"(+{d_it_wer*100:.2f} vs +{d_en_wer*100:.2f} WER points). This looks like a general 8-codebook loss, not an Italian-specific failure."
        )

    # Turbo comparison, if present
    turbo_txt = ""
    turbo_d = None
    if it_turbo:
        to, t8 = it_turbo["original"], it_turbo["q8"]
        turbo_d = t8["corpus"]["wer"] - to["corpus"]["wer"]
        turbo_txt = (
            f"Re-scoring the same 80 Italian originals and 8q reconstructions with faster-whisper "
            f"**large-v3-turbo** int8, language=it, same decoding settings: corpus WER "
            f"{pct(to['corpus']['wer'])} → {pct(t8['corpus']['wer'])} "
            f"({turbo_d*100:+.2f} pts), CER {pct(to['corpus']['cer'])} → {pct(t8['corpus']['cer'])} "
            f"({(t8['corpus']['cer']-to['corpus']['cer'])*100:+.2f} pts), mean avg log-prob "
            f"{fnum(to['mean_avg_logprob'])} → {fnum(t8['mean_avg_logprob'])}. "
        )
        # Is the small-model gap just fragility?
        if turbo_d + 0.02 >= d_it_wer:
            turbo_txt += (
                f"The stronger ASR does **not** shrink the 8q gap (it is {turbo_d*100:.2f} vs "
                f"{d_it_wer*100:.2f} points on whisper-small), so the Italian degradation is not just whisper-small fragility."
            )
        elif turbo_d <= 0.5 * d_it_wer and turbo_d < 0.08:
            turbo_txt += (
                f"The stronger ASR shrinks the gap substantially ({d_it_wer*100:.2f} → {turbo_d*100:.2f} points), "
                f"so a large share of the original Italian result was whisper-small fragility. A residual 8q loss remains."
            )
        else:
            turbo_txt += (
                f"The stronger ASR shrinks the gap somewhat ({d_it_wer*100:.2f} → {turbo_d*100:.2f} points) but a clear 8q loss remains; "
                f"it is not only whisper-small fragility."
            )
    else:
        turbo_txt = turbo_meta.get("error") or "large-v3-turbo was not run."

    # One-line pilot verdict: combine English comparison + turbo (if any) + previous 32q finding
    it32 = it_small.get("q32_vs_q8") or {}
    en32 = en_metrics.get("q32_vs_q8") or {}
    # Pilot OK if residual Italian 8q WER rise (prefer turbo if available) is not huge
    # AND not dramatically worse than English, acknowledging 32q is better.
    residual = turbo_d if turbo_d is not None else d_it_wer
    if residual <= 0.08 and wer_gap < 0.08:
        one = (
            f"YES — 8q Mimi is OK for a small Italian pilot: residual WER rise is "
            f"{residual*100:.1f} points "
            + ("on large-v3-turbo" if turbo_d is not None else "on whisper-small")
            + f", and the Italian-vs-English gap is {wer_gap*100:.1f} points; keep the train short and treat 32q as a later upgrade."
        )
    elif residual <= 0.15 or (turbo_d is not None and turbo_d <= 0.12):
        one = (
            f"YES, WITH CAUTION — 8q Mimi is acceptable for a small Italian pilot, not a long paid train: "
            f"Italian WER still rises {residual*100:.1f} points "
            + ("on large-v3-turbo" if turbo_d is not None else "on whisper-small")
            + f" (English +{d_en_wer*100:.1f} on whisper-small); 32q remains clearly cleaner."
        )
    else:
        one = (
            f"NOT YET — 8q Mimi is too lossy for even a confident Italian pilot until conversational audio is checked "
            f"or more codebooks are tried: Italian WER rises {residual*100:.1f} points "
            + ("on large-v3-turbo" if turbo_d is not None else "on whisper-small")
            + f" vs English +{d_en_wer*100:.1f} on whisper-small."
        )

    lines = []
    a = lines.append
    a("")
    a("---")
    a("")
    a("## Follow-up: English baseline and stronger Italian ASR")
    a("")
    a("_Appended Friday 27 Nov 2026, Europe/Zurich. Still CPU only; no GPU rental, no accounts, no spending._")
    a("")
    a("### One-line verdict")
    a("")
    a(f"**{one}**")
    a("")
    a(f"**Is the Italian 8q degradation meaningfully worse than English?** {worse}. {worse_txt}")
    a("")
    a("### Side-by-side: FLEURS Italian vs English, whisper-small int8")
    a("")
    a("Same codec (Mimi, 8 quantizers, 24 kHz), same ASR class (faster-whisper `small` int8, beam 5, temperature 0, no VAD), same seed 42, same 38 male + 42 female target, same 5–25 s duration filter. Language flag is `it` vs `en`. English originals are shorter on average (FLEURS `en_us` test is a shorter-utterance pool than `it_it`); that is a property of the set, not a sampling bug.")
    a("")
    a("| Language | Condition | Corpus WER | Corpus CER | Macro WER | Macro CER | Mean ASR avg log-prob | Mean SI-SDR (dB) |")
    a("|---|---|---:|---:|---:|---:|---:|---:|")
    a(
        f"| Italian | Original | {pct(it_o['corpus']['wer'])} | {pct(it_o['corpus']['cer'])} | "
        f"{pct(it_o['macro']['wer'])} | {pct(it_o['macro']['cer'])} | {fnum(it_o['mean_avg_logprob'])} | — |"
    )
    a(
        f"| Italian | Mimi 8q | {pct(it_8['corpus']['wer'])} | {pct(it_8['corpus']['cer'])} | "
        f"{pct(it_8['macro']['wer'])} | {pct(it_8['macro']['cer'])} | {fnum(it_8['mean_avg_logprob'])} | {fnum(it_8['mean_si_sdr'], 2)} |"
    )
    a(
        f"| Italian | Change (8q − orig) | {d_it_wer*100:+.2f} pts | {d_it_cer*100:+.2f} pts | "
        f"{(it_8['macro']['wer']-it_o['macro']['wer'])*100:+.2f} pts | "
        f"{(it_8['macro']['cer']-it_o['macro']['cer'])*100:+.2f} pts | {d_it_lp:+.3f} | — |"
    )
    a(
        f"| English | Original | {pct(en_o['corpus']['wer'])} | {pct(en_o['corpus']['cer'])} | "
        f"{pct(en_o['macro']['wer'])} | {pct(en_o['macro']['cer'])} | {fnum(en_o['mean_avg_logprob'])} | — |"
    )
    a(
        f"| English | Mimi 8q | {pct(en_8['corpus']['wer'])} | {pct(en_8['corpus']['cer'])} | "
        f"{pct(en_8['macro']['wer'])} | {pct(en_8['macro']['cer'])} | {fnum(en_8['mean_avg_logprob'])} | {fnum(en_8['mean_si_sdr'], 2)} |"
    )
    a(
        f"| English | Change (8q − orig) | {d_en_wer*100:+.2f} pts | {d_en_cer*100:+.2f} pts | "
        f"{(en_8['macro']['wer']-en_o['macro']['wer'])*100:+.2f} pts | "
        f"{(en_8['macro']['cer']-en_o['macro']['cer'])*100:+.2f} pts | {d_en_lp:+.3f} | — |"
    )
    a("")
    a(
        f"- English sample: **{en_manifest['n_clips']}** FLEURS `en_us` test utterances "
        f"({en_manifest['n_male']} male, {en_manifest['n_female']} female; seed {SEED}), "
        f"**{en_manifest['audio_s']:.1f} s** ({en_manifest['audio_s']/60:.1f} min). "
        f"Source: [google/fleurs](https://huggingface.co/datasets/google/fleurs) `data/en_us` test TSV + `audio/test.tar.gz`. Licence: **CC BY 4.0**. Train/dev were not downloaded."
    )
    a(
        f"- English CPU RTF (8q): encode {fnum(en_timing['q8']['encode_rtf'])}×, "
        f"decode {fnum(en_timing['q8']['decode_rtf'])}×, round-trip {fnum(en_timing['q8']['roundtrip_rtf'])}× "
        f"({fnum(en_timing['q8']['encode_s'], 1)} s + {fnum(en_timing['q8']['decode_s'], 1)} s over {en_timing['q8']['audio_s']:.1f} s audio)."
    )
    # sensitivity
    en_sens = en_metrics.get("exclude_collapses")
    if en_sens:
        a(
            f"- English excluding {en_sens['n_dropped']} ASR-collapse clip(s) (WER ≥ 100% or ≤1 word): "
            f"corpus WER {pct(en_sens['original']['wer'])} → {pct(en_sens['q8']['wer'])} "
            f"({(en_sens['q8']['wer']-en_sens['original']['wer'])*100:+.2f} pts) on {en_sens['n']} clips."
        )
    a("- Absolute WER levels are not comparable across languages: whisper-small is much stronger on English, so the **delta** (and the 8q-vs-32q gap) is the codec signal. Italian originals already sit at 19.4% WER; English originals should be much lower.")
    a("")

    # 8 vs 32 both languages
    a("### 8 vs 32 quantizers, 10 clips each")
    a("")
    def q32_row(lang, block):
        if not block:
            return
        o, q8, q32 = block["original"], block["q8"], block["q32"]
        a(
            f"| {lang} | Original (same 10) | {pct(o['corpus']['wer'])} | {pct(o['corpus']['cer'])} | "
            f"{fnum(o['mean_avg_logprob'])} | — |"
        )
        a(
            f"| {lang} | Mimi 8q | {pct(q8['corpus']['wer'])} | {pct(q8['corpus']['cer'])} | "
            f"{fnum(q8['mean_avg_logprob'])} | {fnum(q8['mean_si_sdr'], 2)} |"
        )
        a(
            f"| {lang} | Mimi 32q | {pct(q32['corpus']['wer'])} | {pct(q32['corpus']['cer'])} | "
            f"{fnum(q32['mean_avg_logprob'])} | {fnum(q32['mean_si_sdr'], 2)} |"
        )
        a(
            f"| {lang} | 8q − orig | {(q8['corpus']['wer']-o['corpus']['wer'])*100:+.2f} pts | "
            f"{(q8['corpus']['cer']-o['corpus']['cer'])*100:+.2f} pts | "
            f"{(q8['mean_avg_logprob']-o['mean_avg_logprob']):+.3f} | — |"
        )
        a(
            f"| {lang} | 32q − orig | {(q32['corpus']['wer']-o['corpus']['wer'])*100:+.2f} pts | "
            f"{(q32['corpus']['cer']-o['corpus']['cer'])*100:+.2f} pts | "
            f"{(q32['mean_avg_logprob']-o['mean_avg_logprob']):+.3f} | — |"
        )

    a("| Language | Condition | Corpus WER | Corpus CER | Mean ASR avg log-prob | Mean SI-SDR (dB) |")
    a("|---|---|---:|---:|---:|---:|")
    q32_row("Italian", it32)
    q32_row("English", en32)
    a("")
    if en32:
        en8g = en32["q8"]["corpus"]["wer"] - en32["original"]["corpus"]["wer"]
        en32g = en32["q32"]["corpus"]["wer"] - en32["original"]["corpus"]["wer"]
        a(
            f"On the English 10-clip diagnostic, 32q WER rise is {en32g*100:+.2f} points vs {en8g*100:+.2f} at 8q, "
            f"and SI-SDR goes from {fnum(en32['q8']['mean_si_sdr'], 2)} dB to {fnum(en32['q32']['mean_si_sdr'], 2)} dB. "
            "If English also gains a lot at 32q, the missing detail is a general 8-codebook limit; if English 8q is already close to 32q while Italian is not, the 8q budget is harsher on Italian."
        )
        a("")
        a("English per-clip 8q vs 32q WER:")
        a("")
        a("| Clip | Duration (s) | Gender | WER orig | WER 8q | WER 32q | SI-SDR 8q | SI-SDR 32q |")
        a("|---|---:|---|---:|---:|---:|---:|---:|")
        for row in en32["per_utt"]:
            a(
                f"| `{row['id']}` | {row['duration_s']:.2f} | {row['gender']} | "
                f"{pct(row['wer_orig'])} | {pct(row['wer_q8'])} | {pct(row['wer_q32'])} | "
                f"{fnum(row['si_sdr_q8'], 2)} | {fnum(row['si_sdr_q32'], 2)} |"
            )
        a("")

    a("### Stronger ASR on the Italian 8q set")
    a("")
    a(turbo_txt)
    a("")
    if it_turbo:
        to, t8 = it_turbo["original"], it_turbo["q8"]
        a("| ASR | Condition | Corpus WER | Corpus CER | Macro WER | Macro CER | Mean ASR avg log-prob |")
        a("|---|---|---:|---:|---:|---:|---:|")
        a(
            f"| whisper-small int8 | Original | {pct(it_o['corpus']['wer'])} | {pct(it_o['corpus']['cer'])} | "
            f"{pct(it_o['macro']['wer'])} | {pct(it_o['macro']['cer'])} | {fnum(it_o['mean_avg_logprob'])} |"
        )
        a(
            f"| whisper-small int8 | Mimi 8q | {pct(it_8['corpus']['wer'])} | {pct(it_8['corpus']['cer'])} | "
            f"{pct(it_8['macro']['wer'])} | {pct(it_8['macro']['cer'])} | {fnum(it_8['mean_avg_logprob'])} |"
        )
        a(
            f"| large-v3-turbo int8 | Original | {pct(to['corpus']['wer'])} | {pct(to['corpus']['cer'])} | "
            f"{pct(to['macro']['wer'])} | {pct(to['macro']['cer'])} | {fnum(to['mean_avg_logprob'])} |"
        )
        a(
            f"| large-v3-turbo int8 | Mimi 8q | {pct(t8['corpus']['wer'])} | {pct(t8['corpus']['cer'])} | "
            f"{pct(t8['macro']['wer'])} | {pct(t8['macro']['cer'])} | {fnum(t8['mean_avg_logprob'])} |"
        )
        a("")
        a(
            f"- large-v3-turbo load + decode ran on CPU, int8, language=`it`. "
            f"Wall time: {fnum(turbo_meta.get('elapsed_s'), 1)} s "
            f"(RTF {fnum(turbo_meta.get('rtf'), 3)}× over {fnum(turbo_meta.get('audio_s'), 1)} s of original+8q audio)."
        )
        if it_turbo.get("exclude_collapses"):
            s = it_turbo["exclude_collapses"]
            a(
                f"- Excluding {s['n_dropped']} turbo collapse clip(s): corpus WER "
                f"{pct(s['original']['wer'])} → {pct(s['q8']['wer'])} "
                f"({(s['q8']['wer']-s['original']['wer'])*100:+.2f} pts) on {s['n']} clips."
            )
        # a couple of remaining worst turbo cases
        worst = sorted(
            t8["per_utt"],
            key=lambda r: (-(r["wer"] - r["orig_wer"]), -r["wer"]),
        )[:3]
        if worst:
            a("- Worst remaining Italian 8q clips under large-v3-turbo (WER orig → 8q):")
            for r in worst:
                a(
                    f"  - `{r['id']}`: {pct(r['orig_wer'])} → {pct(r['wer'])}. "
                    f"Ref: {r['ref_norm'][:120]}… / 8q ASR: {r['hyp_norm'][:120]}"
                )
        a("")

    a("### How to read this for the pilot")
    a("")
    a("- A higher English baseline accuracy is expected and does not by itself mean Mimi is “fine for English, bad for Italian.” Compare rises, SI-SDR, and the 8q-vs-32q diagnostic.")
    a("- Conversation, overlap, and telephone channel are still untested. This follow-up is still read speech only.")
    a("- English wavs, manifest and metrics: `en_us/`. Italian turbo transcripts: `asr_italian_large_v3_turbo.json`.")
    a("")

    body = REPORT_PATH.read_text(encoding="utf-8")
    marker = "\n## Follow-up: English baseline and stronger Italian ASR\n"
    if "\n## Follow-up: English baseline and stronger Italian ASR\n" in body or "## Follow-up: English baseline and stronger Italian ASR" in body:
        # replace previous follow-up if re-run
        idx = body.find("## Follow-up: English baseline and stronger Italian ASR")
        # keep everything before the horizontal rule that precedes it, if present
        pre = body[:idx]
        if pre.rstrip().endswith("---"):
            pre = pre[: pre.rstrip().rfind("---")].rstrip() + "\n"
        body = pre
    new = body.rstrip() + "\n" + "\n".join(lines) + "\n"
    REPORT_PATH.write_text(new, encoding="utf-8")
    print(f"Appended follow-up to {REPORT_PATH}", flush=True)
    print("ONE_LINE_VERDICT:", one, flush=True)
    print("IT_VS_EN:", worse, worse_txt, flush=True)
    return one, worse


def italian_clips_from_manifest():
    man = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    clips = []
    for d in man["clips"]:
        c = it.Clip(
            id=d["id"],
            sentence_id=d["sentence_id"],
            filename=d["filename"],
            raw_transcription=d["raw_transcription"],
            transcription=d["transcription"],
            gender=d["gender"],
            num_samples_16k=d["num_samples_16k"],
            duration_s=d["duration_s"],
            source=d["source"],
            license=d["license"],
            split=d["split"],
            q32=d.get("q32", False),
        )
        clips.append(c)
    return clips


def run_english():
    EN_ROOT.mkdir(parents=True, exist_ok=True)
    all_clips = it.parse_fleurs_tsv(EN_TSV)
    selected = select_en_clips(all_clips)
    it.extract_selected(selected, EN_TAR, EN_RAW_WAVS)
    # drop the large archive once the 80 wavs are extracted
    if EN_TAR.exists() and all((EN_RAW_WAVS / c.filename).exists() for c in selected):
        EN_TAR.unlink()
        print(f"Removed {EN_TAR.name} after extracting 80 wavs.", flush=True)

    manifest = {
        "seed": SEED,
        "n_clips": len(selected),
        "n_male": sum(1 for c in selected if c.gender.upper() == "MALE"),
        "n_female": sum(1 for c in selected if c.gender.upper() != "MALE"),
        "audio_s": sum(c.duration_s for c in selected),
        "sample_rate_hz": it.SR,
        "source": "google/fleurs en_us test",
        "license": "CC BY 4.0",
        "pool": {
            "n_test": len(all_clips),
            "n_unique_sentences": len({c.sentence_id for c in all_clips}),
            "n_male": sum(1 for c in all_clips if c.gender.upper() == "MALE"),
            "n_female": sum(1 for c in all_clips if c.gender.upper() != "MALE"),
        },
        "clips": [asdict(c) for c in selected],
    }
    (EN_ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"English manifest: {len(selected)} clips, {manifest['audio_s']:.1f}s, "
        f"{manifest['n_male']} male / {manifest['n_female']} female, "
        f"{sum(1 for c in selected if c.q32)} for 32q",
        flush=True,
    )
    orig_audio = load_or_resample(selected, EN_RAW_WAVS, EN_ORIG)
    q8_meta, q32_meta, timing = run_mimi(
        selected,
        orig_audio,
        EN_Q8,
        EN_Q32,
        EN_ROOT / "q8_audio_metrics.json",
        EN_ROOT / "q32_audio_metrics.json",
        EN_ROOT / "timing.json",
    )
    asr = load_asr("small", "en")
    asr_store = {}
    if EN_ASR_PATH.exists():
        try:
            asr_store = json.loads(EN_ASR_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            asr_store = {}
    q32_clips = [c for c in selected if c.q32]
    t0 = time.perf_counter()
    for i, c in enumerate(selected, 1):
        rec = cached_transcribe(asr, asr_store, EN_ASR_PATH, f"orig:{c.id}", EN_ORIG / f"{c.id}.wav", "en")
        print(f"[asr en orig {i}/{len(selected)}] lp={rec['avg_logprob']:.3f} {rec['text'][:80]!r}", flush=True)
    for i, c in enumerate(selected, 1):
        rec = cached_transcribe(asr, asr_store, EN_ASR_PATH, f"q8:{c.id}", EN_Q8 / f"{c.id}.wav", "en")
        print(f"[asr en q8 {i}/{len(selected)}] lp={rec['avg_logprob']:.3f} {rec['text'][:80]!r}", flush=True)
    for i, c in enumerate(q32_clips, 1):
        rec = cached_transcribe(asr, asr_store, EN_ASR_PATH, f"q32:{c.id}", EN_Q32 / f"{c.id}.wav", "en")
        print(f"[asr en q32 {i}/{len(q32_clips)}] lp={rec['avg_logprob']:.3f} {rec['text'][:80]!r}", flush=True)
    asr_s = time.perf_counter() - t0
    print(f"English whisper-small ASR wall {asr_s:.1f}s", flush=True)
    del asr
    gc.collect()

    metrics = {
        "original": pack_condition(selected, asr_store, lambda c: f"orig:{c.id}"),
        "q8": pack_condition(selected, asr_store, lambda c: f"q8:{c.id}", q8_meta),
        "q32_vs_q8": pack_q32_compare(q32_clips, asr_store, q8_meta, q32_meta, timing),
    }
    sens, dropped = corpus_excluding_collapses(selected, asr_store, lambda c: f"orig:{c.id}", lambda c: f"q8:{c.id}")
    if sens:
        metrics["exclude_collapses"] = {**sens, "n_dropped": dropped}
    (EN_ROOT / "metrics.json").write_text(json.dumps(sanitise(metrics), ensure_ascii=False, indent=2), encoding="utf-8")
    return selected, manifest, metrics, timing, asr_store


def run_italian_turbo(it_clips):
    meta = {"model": TURBO_NAME, "compute_type": "int8", "language": "it"}
    try:
        asr = load_asr(TURBO_NAME, "it")
    except Exception as e:
        meta["error"] = f"Failed to load {TURBO_NAME} on CPU: {type(e).__name__}: {e}"
        print(meta["error"], flush=True)
        return None, meta

    store = {}
    if TURBO_ASR_PATH.exists():
        try:
            store = json.loads(TURBO_ASR_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            store = {}
    t0 = time.perf_counter()
    audio_s = 0.0
    try:
        for i, c in enumerate(it_clips, 1):
            rec = cached_transcribe(
                asr, store, TURBO_ASR_PATH, f"orig:{c.id}", it.ORIG_DIR / f"{c.id}.wav", "it"
            )
            audio_s += c.duration_s
            print(f"[asr it-turbo orig {i}/{len(it_clips)}] lp={rec['avg_logprob']:.3f} {rec['text'][:80]!r}", flush=True)
        for i, c in enumerate(it_clips, 1):
            rec = cached_transcribe(
                asr, store, TURBO_ASR_PATH, f"q8:{c.id}", it.RECON8_DIR / f"{c.id}.wav", "it"
            )
            audio_s += c.duration_s
            print(f"[asr it-turbo q8 {i}/{len(it_clips)}] lp={rec['avg_logprob']:.3f} {rec['text'][:80]!r}", flush=True)
    except Exception as e:
        meta["error"] = f"{TURBO_NAME} failed mid-run after {time.perf_counter()-t0:.1f}s: {type(e).__name__}: {e}"
        print(meta["error"], flush=True)
        del asr
        gc.collect()
        return None, meta
    elapsed = time.perf_counter() - t0
    meta["elapsed_s"] = elapsed
    meta["audio_s"] = audio_s
    meta["rtf"] = elapsed / audio_s if audio_s else float("nan")
    print(f"Italian large-v3-turbo wall {elapsed:.1f}s RTF {meta['rtf']:.3f}x", flush=True)
    del asr
    gc.collect()

    it_q8_metrics = json.loads((ROOT / "q8_audio_metrics.json").read_text())
    packed_o = pack_condition(it_clips, store, lambda c: f"orig:{c.id}")
    packed_8 = pack_condition(it_clips, store, lambda c: f"q8:{c.id}", it_q8_metrics)
    # attach orig wer to q8 rows for worst-list
    orig_by_id = {r["id"]: r for r in packed_o["per_utt"]}
    for r in packed_8["per_utt"]:
        r["orig_wer"] = orig_by_id[r["id"]]["wer"]
        r["orig_cer"] = orig_by_id[r["id"]]["cer"]
        r["orig_hyp_norm"] = orig_by_id[r["id"]]["hyp_norm"]
    metrics = {"original": packed_o, "q8": packed_8}
    sens, dropped = corpus_excluding_collapses(it_clips, store, lambda c: f"orig:{c.id}", lambda c: f"q8:{c.id}")
    if sens:
        metrics["exclude_collapses"] = {**sens, "n_dropped": dropped}
    (ROOT / "metrics_italian_large_v3_turbo.json").write_text(
        json.dumps(sanitise(metrics), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics, meta


def main() -> int:
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 8)))
    torch.set_grad_enabled(False)

    print("=== English baseline ===", flush=True)
    en_clips, en_manifest, en_metrics, en_timing, _ = run_english()

    print("=== Italian large-v3-turbo re-score ===", flush=True)
    it_clips = italian_clips_from_manifest()
    it_small = json.loads((ROOT / "metrics.json").read_text(encoding="utf-8"))
    it_turbo, turbo_meta = run_italian_turbo(it_clips)

    one, worse = append_report(en_metrics, en_manifest, en_timing, it_small, it_turbo, turbo_meta)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
