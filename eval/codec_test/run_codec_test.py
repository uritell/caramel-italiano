#!/usr/bin/env python3
"""CPU-only test of whether Kyutai Mimi preserves Italian speech.

Recipe match: transformers.MimiModel.from_pretrained('kyutai/mimi'),
encode with num_quantizers=8, audio at 24 kHz mono (caramel-ref/generate.py).

No GPU, no accounts, no paid APIs.

Path note: ROOT and HF_HOME originally pointed at /workspace/caramel-italiano/.
They now default to this directory and $HF_HOME (or ./.hf-cache).
"""
from __future__ import annotations

import csv
import gc
import json
import math
import os
import random
import re
import statistics
import sys
import tarfile
import time
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import torch
from jiwer import cer as jiwer_cer
from jiwer import wer as jiwer_wer

# -----------------------------------------------------------------------------
# Paths and constants
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
ORIG_DIR = ROOT / "originals_24k"
RECON8_DIR = ROOT / "recon_q8_24k"
RECON32_DIR = ROOT / "recon_q32_24k"
EXAMPLES_DIR = ROOT / "examples"
HF_HOME = Path(os.environ.get("HF_HOME", str(ROOT / ".hf-cache")))

TSV_PATH = RAW / "fleurs_it_it_test.tsv"
TAR_PATH = RAW / "fleurs_it_it_test.tar.gz"
MANIFEST_PATH = ROOT / "manifest.json"
METRICS_PATH = ROOT / "metrics.json"
ASR_PATH = ROOT / "asr_results.json"
TIMING_PATH = ROOT / "timing.json"
REPORT_PATH = ROOT / "report.md"

SR = 24000
N_CLIPS = 80
N_Q32 = 10
SEED = 42
NUM_QUANTIZERS_MAIN = 8
NUM_QUANTIZERS_DIAG = 32
ASR_MODEL_NAME = "small"  # faster-whisper; feasible on this 15 GiB CPU box
ASR_LANGUAGE = "it"

os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_HOME / "transformers"))
os.environ.setdefault("HF_HUB_CACHE", str(HF_HOME / "hub"))

# -----------------------------------------------------------------------------
# Audio / text helpers
# -----------------------------------------------------------------------------
# Keep apostrophes (Italian elision: l'incidente, un'altra). Strip other punctuation.
_APOS = {"'", "’", "ʼ", "＇", "`"}
_PUNCT_RE = None


def _punct_re():
    global _PUNCT_RE
    if _PUNCT_RE is None:
        chars = []
        for i in range(0x10000):
            ch = chr(i)
            if unicodedata.category(ch).startswith("P") and ch not in _APOS:
                chars.append(re.escape(ch))
        _PUNCT_RE = re.compile("[" + "".join(chars) + "]")
    return _PUNCT_RE


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    t = unicodedata.normalize("NFC", text).lower().strip()
    t = t.replace("’", "'").replace("ʼ", "'").replace("＇", "'").replace("`", "'")
    t = _punct_re().sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    return x, int(sr)


def to_24k_mono(x: np.ndarray, sr: int) -> np.ndarray:
    if sr != SR:
        x = soxr.resample(x, sr, SR, quality="VHQ")
    return np.ascontiguousarray(x, dtype=np.float32)


def save_wav(path: Path, x: np.ndarray, sr: int = SR) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(x, -1.0, 1.0), sr, subtype="PCM_16")


def si_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    n = min(len(est), len(ref))
    if n == 0:
        return float("nan")
    est = est[:n].astype(np.float64)
    ref = ref[:n].astype(np.float64)
    ref_energy = float(np.dot(ref, ref)) + 1e-12
    alpha = float(np.dot(est, ref)) / ref_energy
    target = alpha * ref
    noise = est - target
    return 10.0 * math.log10((float(np.dot(target, target)) + 1e-12) / (float(np.dot(noise, noise)) + 1e-12))


def peak_snr(est: np.ndarray, ref: np.ndarray) -> float:
    n = min(len(est), len(ref))
    if n == 0:
        return float("nan")
    err = est[:n].astype(np.float64) - ref[:n].astype(np.float64)
    mse = float(np.mean(err * err)) + 1e-12
    peak = max(float(np.max(np.abs(ref[:n]))), 1e-6)
    return 20.0 * math.log10(peak / math.sqrt(mse))


# -----------------------------------------------------------------------------
# Dataset selection (FLEURS it_it test, CC BY 4.0)
# -----------------------------------------------------------------------------
@dataclass
class Clip:
    id: str
    sentence_id: str
    filename: str
    raw_transcription: str
    transcription: str
    gender: str
    num_samples_16k: int
    duration_s: float
    source: str
    license: str
    split: str
    q32: bool = False


def parse_fleurs_tsv(path: Path) -> list[Clip]:
    clips = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 7:
                raise ValueError(f"Unexpected TSV columns on line {i+1}: {len(parts)}")
            num_samples = int(parts[5])
            clips.append(
                Clip(
                    id=Path(parts[1]).stem,
                    sentence_id=parts[0],
                    filename=parts[1],
                    raw_transcription=parts[2],
                    transcription=parts[3],
                    gender=parts[6],
                    num_samples_16k=num_samples,
                    duration_s=num_samples / 16000.0,
                    source="google/fleurs it_it test (FLEURS-102 archive on Hugging Face)",
                    license="CC BY 4.0",
                    split="test",
                )
            )
    return clips


def unique_sentence_sample(items: list[Clip], k: int, rng: random.Random) -> list[Clip]:
    items = list(items)
    rng.shuffle(items)
    chosen, used = [], set()
    for c in items:
        if c.sentence_id in used:
            continue
        used.add(c.sentence_id)
        chosen.append(c)
        if len(chosen) == k:
            return chosen
    for c in items:
        if c in chosen:
            continue
        chosen.append(c)
        if len(chosen) == k:
            return chosen
    return chosen


def select_clips(all_clips: list[Clip], n: int = N_CLIPS, seed: int = SEED) -> list[Clip]:
    rng = random.Random(seed)
    # All FLEURS test clips are >= 5.28 s. Cap very long ones so CPU encode stays tractable.
    eligible = [c for c in all_clips if 5.0 <= c.duration_s <= 25.0]
    males = [c for c in eligible if c.gender.upper() == "MALE"]
    females = [c for c in eligible if c.gender.upper() == "FEMALE"]
    # Test set is ~96% female; take every eligible male, fill the rest with unique-sentence females.
    n_male = min(len(males), n)
    n_female = n - n_male
    selected = unique_sentence_sample(males, n_male, rng) + unique_sentence_sample(females, n_female, rng)
    selected.sort(key=lambda c: c.id)
    # 10-clip 8-vs-32 diagnostic: even duration coverage, seed-stable.
    by_dur = sorted(selected, key=lambda c: (c.duration_s, c.id))
    step = max(len(by_dur) // N_Q32, 1)
    q32_ids = {by_dur[min(i * step, len(by_dur) - 1)].id for i in range(N_Q32)}
    # If collisions (short list), fill from the start.
    if len(q32_ids) < N_Q32:
        for c in by_dur:
            q32_ids.add(c.id)
            if len(q32_ids) == N_Q32:
                break
    for c in selected:
        c.q32 = c.id in q32_ids
    return selected


def extract_selected(clips: list[Clip], tar_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    wanted = {c.filename: c for c in clips}
    missing = {c.filename for c in clips if not (dest_dir / c.filename).exists()}
    if not missing:
        print(f"All {len(clips)} wavs already extracted to {dest_dir}", flush=True)
        return
    if not tar_path.exists():
        raise FileNotFoundError(f"Missing FLEURS test archive: {tar_path}")
    found = set()
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf.getmembers():
            name = Path(member.name).name
            if name in missing and member.isfile():
                src = tf.extractfile(member)
                if src is None:
                    continue
                data = src.read()
                (dest_dir / name).write_bytes(data)
                found.add(name)
    still = missing - found
    if still:
        raise FileNotFoundError(f"{len(still)} selected wavs not in tarball, e.g. {list(still)[:3]}")
    print(f"Extracted {len(found)} wavs to {dest_dir}", flush=True)


# -----------------------------------------------------------------------------
# Mimi encode / decode
# -----------------------------------------------------------------------------
def load_mimi():
    from transformers import MimiModel

    print("Loading MimiModel.from_pretrained('kyutai/mimi') on CPU…", flush=True)
    t0 = time.perf_counter()
    model = MimiModel.from_pretrained("kyutai/mimi")
    model = model.to("cpu").eval()
    print(f"Mimi loaded in {time.perf_counter()-t0:.1f}s; params={sum(p.numel() for p in model.parameters()):,}", flush=True)
    return model


@torch.inference_mode()
def mimi_roundtrip(model, audio_24k: np.ndarray, num_quantizers: int) -> tuple[np.ndarray, tuple, float, float]:
    wav = torch.from_numpy(np.ascontiguousarray(audio_24k)).float().unsqueeze(0).unsqueeze(0)  # [1,1,T]
    t0 = time.perf_counter()
    enc = model.encode(wav, num_quantizers=num_quantizers)
    codes = enc.audio_codes
    t_enc = time.perf_counter() - t0
    t1 = time.perf_counter()
    dec = model.decode(codes)
    recon = dec.audio_values[0, 0].contiguous().cpu().numpy().astype(np.float32)
    t_dec = time.perf_counter() - t1
    return recon, tuple(int(x) for x in codes.shape), t_enc, t_dec


# -----------------------------------------------------------------------------
# ASR
# -----------------------------------------------------------------------------
def load_asr():
    from faster_whisper import WhisperModel

    cache = str(HF_HOME / "faster-whisper")
    Path(cache).mkdir(parents=True, exist_ok=True)
    n_threads = max(1, min(8, os.cpu_count() or 8))
    print(f"Loading faster-whisper '{ASR_MODEL_NAME}' int8 CPU ({n_threads} threads)…", flush=True)
    t0 = time.perf_counter()
    model = WhisperModel(
        ASR_MODEL_NAME,
        device="cpu",
        compute_type="int8",
        cpu_threads=n_threads,
        download_root=cache,
    )
    print(f"ASR loaded in {time.perf_counter()-t0:.1f}s", flush=True)
    return model


def transcribe_wav(model, path: Path) -> dict:
    # Load with soundfile/soxr and pass a 16 kHz float32 array. Avoid faster-whisper's
    # PyAV path, which crashes with av 19 ("metadata_errors" removed).
    x, sr = load_mono(path)
    if sr != 16000:
        x = soxr.resample(x, sr, 16000, quality="VHQ")
    x = np.ascontiguousarray(x, dtype=np.float32)
    segments, info = model.transcribe(
        x,
        language=ASR_LANGUAGE,
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
    no_speech = []
    for s in segs:
        d = max(float(s.end) - float(s.start), 1e-3)
        dur += d
        wsum += float(s.avg_logprob) * d
        no_speech.append(float(s.no_speech_prob))
    avg_lp = (wsum / dur) if dur > 0 else float("nan")
    return {
        "text": text,
        "avg_logprob": avg_lp,
        "n_segments": len(segs),
        "language_prob": float(getattr(info, "language_probability", float("nan")) or float("nan")),
        "mean_no_speech_prob": float(statistics.mean(no_speech)) if no_speech else float("nan"),
    }


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def utterance_scores(ref: str, hyp: str) -> dict:
    r, h = normalize_text(ref), normalize_text(hyp)
    if not r:
        return {"wer": float("nan"), "cer": float("nan"), "n_words": 0, "n_chars": 0}
    return {
        "wer": float(jiwer_wer(r, h)),
        "cer": float(jiwer_cer(r, h)),
        "n_words": len(r.split()),
        "n_chars": len(r.replace(" ", "")),
        "ref_norm": r,
        "hyp_norm": h,
    }


def corpus_wer_cer(pairs: list[tuple[str, str]]) -> dict:
    refs = [normalize_text(r) for r, _ in pairs]
    hyps = [normalize_text(h) for _, h in pairs]
    return {
        "wer": float(jiwer_wer(refs, hyps)),
        "cer": float(jiwer_cer(refs, hyps)),
        "n_words": sum(len(r.split()) for r in refs),
        "n_chars": sum(len(r.replace(" ", "")) for r in refs),
        "n_utts": len(pairs),
    }


def mean_ignore_nan(xs):
    ys = [float(x) for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(statistics.mean(ys)) if ys else float("nan")


# -----------------------------------------------------------------------------
# Report
# -----------------------------------------------------------------------------
def _pct(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{100.0 * x:.2f}%"


def _f(x, nd=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def write_report(manifest, metrics, asr, timing) -> None:
    clips = manifest["clips"]
    n = len(clips)
    n_male = sum(1 for c in clips if c["gender"].upper() == "MALE")
    n_female = n - n_male
    total_dur = sum(c["duration_s"] for c in clips)
    m8 = metrics["q8"]
    m_orig = metrics["original"]
    delta_wer = m8["corpus"]["wer"] - m_orig["corpus"]["wer"]
    delta_cer = m8["corpus"]["cer"] - m_orig["corpus"]["cer"]
    # Verdict thresholds (documented): small absolute WER rise and still-intelligible reconstructions.
    # "Good enough to build on" means the 8-codebook tokens still carry Italian lexical content,
    # not that Mimi is transparent. A modest WER rise is expected for a 1.1 kbps residual codec.
    intelligible = m8["corpus"]["wer"] < 0.35 and delta_wer < 0.12
    severe = delta_wer >= 0.20 or m8["corpus"]["wer"] >= 0.50
    if severe:
        verdict = "NO"
        verdict_reason = (
            f"Mimi at 8 codebooks degrades Italian intelligibility too much for a reliable training substrate "
            f"(corpus WER { _pct(m_orig['corpus']['wer']) } → { _pct(m8['corpus']['wer']) }, "
            f"absolute rise {delta_wer*100:.2f} points)."
        )
    elif intelligible:
        verdict = "YES"
        verdict_reason = (
            f"Mimi at 8 codebooks preserves enough Italian lexical content to build on. "
            f"Corpus WER on this read-speech set goes from { _pct(m_orig['corpus']['wer']) } (original) to "
            f"{ _pct(m8['corpus']['wer']) } (reconstruction), an absolute rise of {delta_wer*100:.2f} points; "
            f"CER from { _pct(m_orig['corpus']['cer']) } to { _pct(m8['corpus']['cer']) }."
        )
    else:
        verdict = "BORDERLINE"
        verdict_reason = (
            f"Mimi at 8 codebooks is usable with caution. Italian content is mostly there, but the WER rise "
            f"({ _pct(m_orig['corpus']['wer']) } → { _pct(m8['corpus']['wer']) }, "
            f"+{delta_wer*100:.2f} points) is large enough that some phonetic detail is being lost. "
            f"A codec check on conversation audio, and the 32-codebook diagnostic, should be weighed before a long train."
        )

    # Worst cases by WER delta (recon - orig), then CER delta, then recon WER.
    per = metrics["q8"]["per_utt"]
    worst = sorted(per, key=lambda r: (-(r["wer"] - r["orig_wer"]), -(r["cer"] - r["orig_cer"]), -r["wer"]))[:5]

    q32 = metrics.get("q32_vs_q8")
    lines = []
    a = lines.append
    a("# Caramel Italiano — Mimi codec test (Italian, CPU)")
    a("")
    a(f"_Run date: Friday 27 Nov 2026, Europe/Zurich. Box CPU only; no GPU rental, no accounts, no spending._")
    a("")
    a("## Verdict")
    a("")
    a(f"**{verdict}: {verdict_reason}**")
    a("")
    a("This is a content-preservation test of the codec caramel-1 uses as its tokenizer, not a listening-test MOS and not a legal or training-data decision. The 8-codebook setting is the one in `caramel-ref/generate.py` (`MimiModel.from_pretrained('kyutai/mimi')`, `num_quantizers=8`, 24 kHz).")
    a("")
    a("## Headline numbers")
    a("")
    a("| Condition | Corpus WER | Corpus CER | Macro WER | Macro CER | Mean ASR avg log-prob | Mean SI-SDR (dB) |")
    a("|---|---:|---:|---:|---:|---:|---:|")
    a(
        f"| Original 24 kHz audio | {_pct(m_orig['corpus']['wer'])} | {_pct(m_orig['corpus']['cer'])} | "
        f"{_pct(m_orig['macro']['wer'])} | {_pct(m_orig['macro']['cer'])} | {_f(m_orig['mean_avg_logprob'], 3)} | — |"
    )
    a(
        f"| Mimi reconstruction, 8 quantizers | {_pct(m8['corpus']['wer'])} | {_pct(m8['corpus']['cer'])} | "
        f"{_pct(m8['macro']['wer'])} | {_pct(m8['macro']['cer'])} | {_f(m8['mean_avg_logprob'], 3)} | {_f(m8['mean_si_sdr'], 2)} |"
    )
    a(
        f"| Absolute change (recon − original) | {delta_wer*100:+.2f} pts | {delta_cer*100:+.2f} pts | "
        f"{(m8['macro']['wer']-m_orig['macro']['wer'])*100:+.2f} pts | "
        f"{(m8['macro']['cer']-m_orig['macro']['cer'])*100:+.2f} pts | "
        f"{(m8['mean_avg_logprob']-m_orig['mean_avg_logprob']):+.3f} | — |"
    )
    a("")
    a(f"- Clips: **{n}** FLEURS Italian test utterances ({n_male} male, {n_female} female; seed {SEED}).")
    a(f"- Audio duration (original): **{total_dur:.1f} s** ({total_dur/60:.1f} min).")
    a(f"- CPU encode RTF (8q): **{_f(timing['q8']['encode_rtf'], 3)}×**  ({_f(timing['q8']['encode_s'], 1)} s encode / {timing['q8']['audio_s']:.1f} s audio).")
    a(f"- CPU decode RTF (8q): **{_f(timing['q8']['decode_rtf'], 3)}×**  ({_f(timing['q8']['decode_s'], 1)} s decode).")
    a(f"- CPU encode+decode RTF (8q): **{_f(timing['q8']['roundtrip_rtf'], 3)}×**.")
    a(f"- RTF is wall-clock / audio duration on this box (8-core Intel Xeon, CPU, no GPU). Values < 1 mean faster than real time.")
    a("")
    a("## Data used")
    a("")
    a("| Set | What was used | Source | Licence | Notes |")
    a("|---|---|---|---|---|")
    a(
        "| FLEURS Italian test (`it_it`) | 80 read utterances with official transcripts, "
        f"drawn from the 865-clip test split (seed {SEED}) | "
        "[google/fleurs](https://huggingface.co/datasets/google/fleurs) test TSV + "
        "`data/it_it/audio/test.tar.gz`; upstream archive also at "
        "`https://storage.googleapis.com/xtreme_translations/FLEURS102/it_it.tar.gz` | "
        "**CC BY 4.0** (dataset card) | Single-speaker read speech, originally 16 kHz mono WAV, "
        "resampled here to 24 kHz mono with soxr VHQ. Train/dev were not downloaded. "
        "The official normalized `transcription` field is the WER/CER reference."
    )
    a(
        "| DataHive Italian Conversations sample | **Not used** | "
        "[datahiveai/italian-conversations-sample](https://huggingface.co/datasets/datahiveai/italian-conversations-sample) | "
        "CC BY 4.0 (card; no login, not gated) | Downloads without signup, but the six files are "
        "full conversations (7–55 min, 7–52 MB) and the published metadata has no utterance-level "
        "transcripts. Skipped to avoid a large download that cannot enter the WER/CER table."
    )
    a("| Multilingual LibriSpeech Italian test | **Not used** | OpenSLR 94 / CC BY 4.0 | CC BY 4.0 | The public Italian archive is a single large language tarball; FLEURS test already gave transcribed read speech in a smaller download.")
    a("")
    a("Attribution: FLEURS (Conneau et al., 2022), CC BY 4.0. Mimi weights: Kyutai, [kyutai/mimi](https://huggingface.co/kyutai/mimi), CC BY 4.0.")
    a("")
    a("### How the 80 clips were chosen")
    a("")
    a(f"The FLEURS `it_it` test TSV has 865 clips / 346 unique sentences, but only 38 male readings (the rest female). Duration range in the full test split is {manifest['pool']['dur_min_s']:.2f}–{manifest['pool']['dur_max_s']:.2f} s. Filters: duration 5–25 s (excludes {manifest['pool']['n_over_25s']} clips longer than 25 s; no clip is under 5 s). Then: every eligible male reading ({n_male}), plus {n_female} female readings sampled with `random.Random({SEED})`, preferring unique sentence IDs. The test is therefore slightly richer in male speech than the raw split, on purpose.")
    a("")
    a("## Method")
    a("")
    a("1. Load each selected 16 kHz FLEURS WAV, mix to mono (already mono), resample to 24 kHz with soxr `VHQ` (not the bicubic `F.interpolate` used in `generate.py`, which is a poorer audio resampler).")
    a("2. Encode and decode with `transformers.MimiModel.from_pretrained('kyutai/mimi')` on CPU in `torch.inference_mode()`, `num_quantizers=8`. Reconstructions are saved as 24 kHz PCM16 WAVs. For scoring, a reconstruction longer than its original is trimmed to the original length (Mimi's causal convs can add a short tail).")
    a("3. On a 10-clip subset spanning the duration range, also encode/decode with `num_quantizers=32` (Mimi's full codebook stack) as a diagnostic.")
    a("4. Run the same ASR on every original and every reconstruction: `faster-whisper` **whisper-small**, `compute_type='int8'`, `language='it'`, `beam_size=5`, `temperature=0`, no VAD, no conditioning on previous text. OpenAI `whisper-small` via faster-whisper was chosen as feasible on this 15 GiB CPU-only box; `large-v3-turbo` was not used, to avoid memory pressure from other processes. The **difference** between original and reconstructed WER is the codec signal; absolute WER also reflects whisper-small, not just Mimi.")
    a("5. WER and CER use the FLEURS normalized transcript as reference. Hypothesis and reference are lowercased, Unicode-normalized, and stripped of punctuation **except apostrophes**. Corpus figures are the primary result (total edits / total words or characters). Macro figures are the unweighted mean of per-utterance scores.")
    a("6. ASR confidence: duration-weighted mean of each clip's faster-whisper segment `avg_logprob`. SI-SDR is a scale-invariant reconstruction metric against the 24 kHz original (not a perceptual MOS).")
    a("")
    a(f"- Mimi parameters loaded: {timing['mimi_params']:,}.")
    a(f"- torch {timing['torch_version']} ({timing['torch_built']}), transformers {timing['transformers_version']}, faster-whisper {timing['faster_whisper_version']}.")
    a(f"- CPU: {timing['cpu_name']}, {timing['n_threads']} threads. CUDA available: {timing['cuda_available']}.")
    a("")
    a("## 8 vs 32 quantizers (10 clips)")
    a("")
    if q32:
        a(f"Same 10 clips, same ASR and references. Duration of this subset: {q32['audio_s']:.1f} s.")
        a("")
        a("| Condition | Corpus WER | Corpus CER | Macro WER | Macro CER | Mean ASR avg log-prob | Mean SI-SDR (dB) | Encode RTF | Decode RTF |")
        a("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        a(
            f"| Original (same 10) | {_pct(q32['original']['corpus']['wer'])} | {_pct(q32['original']['corpus']['cer'])} | "
            f"{_pct(q32['original']['macro']['wer'])} | {_pct(q32['original']['macro']['cer'])} | "
            f"{_f(q32['original']['mean_avg_logprob'], 3)} | — | — | — |"
        )
        a(
            f"| Mimi 8 quantizers | {_pct(q32['q8']['corpus']['wer'])} | {_pct(q32['q8']['corpus']['cer'])} | "
            f"{_pct(q32['q8']['macro']['wer'])} | {_pct(q32['q8']['macro']['cer'])} | "
            f"{_f(q32['q8']['mean_avg_logprob'], 3)} | {_f(q32['q8']['mean_si_sdr'], 2)} | "
            f"{_f(q32['q8']['encode_rtf'], 3)}× | {_f(q32['q8']['decode_rtf'], 3)}× |"
        )
        a(
            f"| Mimi 32 quantizers | {_pct(q32['q32']['corpus']['wer'])} | {_pct(q32['q32']['corpus']['cer'])} | "
            f"{_pct(q32['q32']['macro']['wer'])} | {_pct(q32['q32']['macro']['cer'])} | "
            f"{_f(q32['q32']['mean_avg_logprob'], 3)} | {_f(q32['q32']['mean_si_sdr'], 2)} | "
            f"{_f(q32['q32']['encode_rtf'], 3)}× | {_f(q32['q32']['decode_rtf'], 3)}× |"
        )
        a("")
        a(q32.get("note", ""))
        a("")
        a("Per-clip WER on this diagnostic subset:")
        a("")
        a("| Clip | Duration (s) | Gender | WER orig | WER 8q | WER 32q | SI-SDR 8q | SI-SDR 32q |")
        a("|---|---:|---|---:|---:|---:|---:|---:|")
        for row in q32["per_utt"]:
            a(
                f"| `{row['id']}` | {_f(row['duration_s'], 2)} | {row['gender']} | "
                f"{_pct(row['wer_orig'])} | {_pct(row['wer_q8'])} | {_pct(row['wer_q32'])} | "
                f"{_f(row['si_sdr_q8'], 2)} | {_f(row['si_sdr_q32'], 2)} |"
            )
    else:
        a("Diagnostic subset was not completed.")
    a("")
    a("## Listen notes (worst reconstructions)")
    a("")
    a("These notes come from the largest ASR degradations (recon WER minus original WER) plus the word-level diffs. They are not a formal listening panel.")
    a("")
    for i, row in enumerate(worst, 1):
        a(f"### {i}. `{row['id']}` — {row['gender'].lower()}, {row['duration_s']:.1f} s; WER { _pct(row['orig_wer']) } → { _pct(row['wer']) } (CER { _pct(row['orig_cer']) } → { _pct(row['cer']) })")
        a("")
        a(f"- Reference: {row['ref_norm']}")
        a(f"- ASR on original: {row['orig_hyp_norm']}")
        a(f"- ASR on 8q reconstruction: {row['hyp_norm']}")
        a(f"- SI-SDR: {_f(row['si_sdr'], 2)} dB.")
        if row.get("note"):
            a(f"- {row['note']}")
        a("")
    a("### Patterns")
    a("")
    for note in metrics.get("listen_notes", []):
        a(f"- {note}")
    a("")
    a("## Example pairs for listening")
    a("")
    a("Two 24 kHz WAV pairs are in `examples/`. Each pair is the original and the 8-quantizer reconstruction, trimmed to the same length.")
    a("")
    for ex in metrics.get("examples", []):
        a(f"- **{ex['label']}** (`{ex['id']}`, {ex['gender'].lower()}, {ex['duration_s']:.1f} s, WER { _pct(ex['orig_wer']) } → { _pct(ex['wer']) }, SI-SDR {_f(ex['si_sdr'], 2)} dB): `{ex['orig_path']}` and `{ex['recon_path']}`")
    a("")
    a("## Files")
    a("")
    a("- `manifest.json` — clip IDs, sources, licences, transcripts, 8-vs-32 flag")
    a("- `originals_24k/` — 80 resampled originals")
    a("- `recon_q8_24k/` — 80 reconstructions at 8 quantizers")
    a("- `recon_q32_24k/` — 10 reconstructions at 32 quantizers")
    a("- `asr_results.json`, `metrics.json`, `timing.json` — raw numbers")
    a("- `examples/` — the two listening pairs")
    a("")
    a("## Caveats")
    a("")
    a("- Read speech only (FLEURS). This does **not** test overlapping talk, backchannels, telephone channel, or informal conversation — the genre caramel actually models. A pass here is necessary but not sufficient.")
    a("- whisper-small will itself mishear some Italian; compare original vs recon, not the absolute WER alone.")
    a("- One 80-clip sample, one seed, one ASR. No claim about other Italian accents, spontaneous speech, or music/noise.")
    a("- Mimi was run in full fp32 on CPU via Hugging Face `transformers` (v" + timing["transformers_version"] + "), matching the public API used in caramel, not a quantized runtime.")
    a("")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {REPORT_PATH}", flush=True)


def word_diff_note(ref: str, orig_hyp: str, recon_hyp: str) -> str:
    """Short, factual note about what the ASR lost or changed after the codec."""
    r = normalize_text(ref).split()
    o = normalize_text(orig_hyp).split()
    h = normalize_text(recon_hyp).split()
    rset, oset, hset = set(r), set(o), set(h)
    # Words in the reference that the original ASR got but the recon ASR missed
    lost = [w for w in r if w in oset and w not in hset]
    # Words the recon ASR introduced that are in neither ref nor orig
    invented = [w for w in h if w not in rset and w not in oset]
    # Words orig missed but recon also missed — not codec's fault
    already = [w for w in r if w not in oset and w not in hset]
    bits = []
    if lost:
        show = ", ".join(f"“{w}”" for w in lost[:8])
        extra = f" (+{len(lost)-8} more)" if len(lost) > 8 else ""
        bits.append(f"After the codec, ASR dropped reference words it had gotten on the original: {show}{extra}.")
    if invented:
        show = ", ".join(f"“{w}”" for w in invented[:6])
        bits.append(f"ASR invented new tokens on the reconstruction: {show}.")
    if not lost and not invented:
        if normalize_text(orig_hyp) == normalize_text(recon_hyp):
            bits.append("The original and reconstruction transcripts are identical; the WER change, if any, is a tie at the corpus level or a normalization edge.")
        else:
            bits.append("The transcript changed mainly by substitution or word-order/morphology rather than by dropping words the original ASR had right.")
    if already and not lost:
        bits.append("Several reference words were already missed on the original, so this clip is a noisy probe of the codec.")
    return " ".join(bits)


def collect_listen_notes(worst, all_per) -> list[str]:
    notes = []
    # Pattern: function words vs content words lost
    lost_words = []
    for row in all_per:
        r = set(row["ref_norm"].split())
        o = set(row["orig_hyp_norm"].split())
        h = set(row["hyp_norm"].split())
        lost_words.extend([w for w in row["ref_norm"].split() if w in o and w not in h])
    n_worse = sum(1 for row in all_per if row["wer"] - row["orig_wer"] > 1e-9)
    n_better = sum(1 for row in all_per if row["orig_wer"] - row["wer"] > 1e-9)
    n_same = len(all_per) - n_worse - n_better
    notes.append(
        f"Of {len(all_per)} clips, 8q reconstruction made utterance WER worse on {n_worse}, better on {n_better}, and unchanged on {n_same}."
    )
    deltas = [row["wer"] - row["orig_wer"] for row in all_per]
    notes.append(
        f"Per-clip WER change (recon − original): median {statistics.median(deltas)*100:+.2f} points, "
        f"mean {statistics.mean(deltas)*100:+.2f} points, max {max(deltas)*100:+.2f} points."
    )
    if lost_words:
        freq = {}
        for w in lost_words:
            freq[w] = freq.get(w, 0) + 1
        top = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
        notes.append(
            "Reference words most often lost by ASR only after the codec (i.e. correct on the original, missing on the reconstruction): "
            + ", ".join(f"“{w}” ({c})" for w, c in top)
            + "."
        )
    sdrs = [row["si_sdr"] for row in all_per if not math.isnan(row["si_sdr"])]
    if sdrs:
        notes.append(
            f"SI-SDR of 8q reconstructions vs the 24 kHz originals: mean {statistics.mean(sdrs):.2f} dB, "
            f"median {statistics.median(sdrs):.2f} dB, min {min(sdrs):.2f} dB."
        )
    # Italian-specific: elision and doubled consonants often stressed by lossy codecs
    elision = [w for w in lost_words if "'" in w]
    if elision:
        notes.append(
            "Several dropped tokens are elided Italian forms (apostrophe), which is a useful stress test of short vowels and clitics: "
            + ", ".join(f"“{w}”" for w in list(dict.fromkeys(elision))[:8])
            + "."
        )
    return notes


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> int:
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 8)))
    torch.set_grad_enabled(False)

    ORIG_DIR.mkdir(parents=True, exist_ok=True)
    RECON8_DIR.mkdir(parents=True, exist_ok=True)
    RECON32_DIR.mkdir(parents=True, exist_ok=True)
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)

    all_clips = parse_fleurs_tsv(TSV_PATH)
    selected = select_clips(all_clips)
    assert len(selected) == N_CLIPS, len(selected)
    raw_dir = RAW / "fleurs_test_wavs"
    extract_selected(selected, TAR_PATH, raw_dir)

    pool_durs = [c.duration_s for c in all_clips]
    manifest = {
        "seed": SEED,
        "n_clips": len(selected),
        "sample_rate_hz": SR,
        "num_quantizers_main": NUM_QUANTIZERS_MAIN,
        "num_quantizers_diag": NUM_QUANTIZERS_DIAG,
        "asr": {"name": f"faster-whisper/{ASR_MODEL_NAME}", "language": ASR_LANGUAGE, "compute_type": "int8"},
        "pool": {
            "set": "google/fleurs it_it test",
            "license": "CC BY 4.0",
            "n_test": len(all_clips),
            "n_unique_sentences": len({c.sentence_id for c in all_clips}),
            "n_male": sum(1 for c in all_clips if c.gender.upper() == "MALE"),
            "n_female": sum(1 for c in all_clips if c.gender.upper() != "MALE"),
            "dur_min_s": min(pool_durs),
            "dur_max_s": max(pool_durs),
            "n_over_25s": sum(1 for d in pool_durs if d > 25.0),
        },
        "clips": [asdict(c) for c in selected],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Manifest: {len(selected)} clips, {sum(c.duration_s for c in selected):.1f}s, "
          f"{sum(1 for c in selected if c.gender.upper()=='MALE')} male, "
          f"{sum(1 for c in selected if c.q32)} marked for 32q", flush=True)

    # Resample originals
    orig_audio = {}
    for i, c in enumerate(selected, 1):
        out = ORIG_DIR / f"{c.id}.wav"
        src = raw_dir / c.filename
        x, sr = load_mono(src)
        y = to_24k_mono(x, sr)
        if not out.exists() or out.stat().st_size == 0:
            save_wav(out, y)
        orig_audio[c.id] = y
        if i % 20 == 0 or i == len(selected):
            print(f"Resampled {i}/{len(selected)} originals to 24 kHz", flush=True)

    # ---- Mimi ----
    q8_metrics_path = ROOT / "q8_audio_metrics.json"
    q32_metrics_path = ROOT / "q32_audio_metrics.json"
    have_q8 = all((RECON8_DIR / f"{c.id}.wav").exists() for c in selected) and q8_metrics_path.exists()
    have_q32 = all((RECON32_DIR / f"{c.id}.wav").exists() for c in selected if c.q32) and q32_metrics_path.exists()
    if have_q8 and have_q32:
        print("Reconstructions already on disk; skipping Mimi encode/decode.", flush=True)
        q8_meta = json.loads(q8_metrics_path.read_text())
        q32_meta = json.loads(q32_metrics_path.read_text())
        if TIMING_PATH.exists():
            timing = json.loads(TIMING_PATH.read_text())
        else:
            timing = {}
        timing.setdefault("torch_version", torch.__version__)
        timing.setdefault("torch_built", "cpu")
        timing.setdefault("transformers_version", __import__("transformers").__version__)
        timing.setdefault("faster_whisper_version", None)
        timing.setdefault("cuda_available", bool(torch.cuda.is_available()))
        timing.setdefault("n_threads", torch.get_num_threads())
        timing.setdefault(
            "cpu_name",
            open("/proc/cpuinfo").read().split("model name")[-1].split("\n")[0].split(":")[-1].strip()
            if Path("/proc/cpuinfo").exists()
            else "unknown",
        )
        timing.setdefault("mimi_params", 79308609)
        skip_mimi = True
        print("Mimi unloaded. Starting ASR.", flush=True)
    else:
        skip_mimi = False
        mimi = load_mimi()
        timing = {
            "torch_version": torch.__version__,
            "torch_built": "cpu" if "cpu" in torch.__version__ else str(torch.__version__),
            "transformers_version": __import__("transformers").__version__,
            "faster_whisper_version": None,
            "cuda_available": bool(torch.cuda.is_available()),
            "n_threads": torch.get_num_threads(),
            "cpu_name": open("/proc/cpuinfo").read().split("model name")[-1].split("\n")[0].split(":")[-1].strip()
            if Path("/proc/cpuinfo").exists()
            else "unknown",
            "mimi_params": int(sum(p.numel() for p in mimi.parameters())),
            "q8": {},
            "q32": {},
        }

    q32_clips = [c for c in selected if c.q32]
    if skip_mimi:
        pass
    else:
        enc_s = dec_s = 0.0
        audio_s = 0.0
        q8_meta = {}
        for i, c in enumerate(selected, 1):
            out = RECON8_DIR / f"{c.id}.wav"
            y = orig_audio[c.id]
            recon, code_shape, t_enc, t_dec = mimi_roundtrip(mimi, y, NUM_QUANTIZERS_MAIN)
            enc_s += t_enc
            dec_s += t_dec
            audio_s += len(y) / SR
            save_wav(out, recon)
            n = min(len(y), len(recon))
            q8_meta[c.id] = {
                "encode_s": t_enc,
                "decode_s": t_dec,
                "code_shape": list(code_shape),
                "orig_samples": int(len(y)),
                "recon_samples": int(len(recon)),
                "si_sdr": si_sdr(recon[:n], y[:n]),
                "peak_snr_db": peak_snr(recon[:n], y[:n]),
            }
            print(
                f"[8q {i}/{len(selected)}] {c.id} {len(y)/SR:.2f}s  "
                f"enc {t_enc:.2f}s dec {t_dec:.2f}s  RTF {(t_enc+t_dec)/(len(y)/SR):.2f}x  "
                f"SI-SDR {q8_meta[c.id]['si_sdr']:.2f} dB",
                flush=True,
            )

        timing["q8"] = {
            "encode_s": enc_s,
            "decode_s": dec_s,
            "audio_s": audio_s,
            "encode_rtf": enc_s / audio_s if audio_s else float("nan"),
            "decode_rtf": dec_s / audio_s if audio_s else float("nan"),
            "roundtrip_rtf": (enc_s + dec_s) / audio_s if audio_s else float("nan"),
        }
        TIMING_PATH.write_text(json.dumps(timing, indent=2), encoding="utf-8")
        (ROOT / "q8_audio_metrics.json").write_text(json.dumps(q8_meta, indent=2), encoding="utf-8")

        # 32q diagnostic on 10 clips
        enc32 = dec32 = aud32 = 0.0
        q32_meta = {}
        for i, c in enumerate(q32_clips, 1):
            out = RECON32_DIR / f"{c.id}.wav"
            y = orig_audio[c.id]
            recon, code_shape, t_enc, t_dec = mimi_roundtrip(mimi, y, NUM_QUANTIZERS_DIAG)
            enc32 += t_enc
            dec32 += t_dec
            aud32 += len(y) / SR
            save_wav(out, recon)
            n = min(len(y), len(recon))
            q32_meta[c.id] = {
                "encode_s": t_enc,
                "decode_s": t_dec,
                "code_shape": list(code_shape),
                "si_sdr": si_sdr(recon[:n], y[:n]),
                "peak_snr_db": peak_snr(recon[:n], y[:n]),
            }
            print(
                f"[32q {i}/{len(q32_clips)}] {c.id} {len(y)/SR:.2f}s  "
                f"enc {t_enc:.2f}s dec {t_dec:.2f}s  RTF {(t_enc+t_dec)/(len(y)/SR):.2f}x  "
                f"SI-SDR {q32_meta[c.id]['si_sdr']:.2f} dB",
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
        TIMING_PATH.write_text(json.dumps(timing, indent=2), encoding="utf-8")
        (ROOT / "q32_audio_metrics.json").write_text(json.dumps(q32_meta, indent=2), encoding="utf-8")

        # Free Mimi before ASR
        del mimi
        gc.collect()
        print("Mimi unloaded. Starting ASR.", flush=True)

    asr = load_asr()
    import faster_whisper

    timing["faster_whisper_version"] = getattr(faster_whisper, "__version__", "unknown")

    # Transcribe originals, q8, and q32. Cache to asr_results for resume.
    asr_results = {}
    if ASR_PATH.exists():
        try:
            asr_results = json.loads(ASR_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            asr_results = {}

    def cached_asr(key: str, path: Path):
        if key in asr_results and asr_results[key].get("text") is not None:
            return asr_results[key]
        rec = transcribe_wav(asr, path)
        rec["path"] = str(path)
        asr_results[key] = rec
        ASR_PATH.write_text(json.dumps(asr_results, ensure_ascii=False, indent=2), encoding="utf-8")
        return rec

    for i, c in enumerate(selected, 1):
        rec = cached_asr(f"orig:{c.id}", ORIG_DIR / f"{c.id}.wav")
        print(f"[asr orig {i}/{len(selected)}] {c.id} lp={rec['avg_logprob']:.3f} {rec['text'][:80]!r}", flush=True)
    for i, c in enumerate(selected, 1):
        rec = cached_asr(f"q8:{c.id}", RECON8_DIR / f"{c.id}.wav")
        print(f"[asr q8 {i}/{len(selected)}] {c.id} lp={rec['avg_logprob']:.3f} {rec['text'][:80]!r}", flush=True)
    for i, c in enumerate(q32_clips, 1):
        rec = cached_asr(f"q32:{c.id}", RECON32_DIR / f"{c.id}.wav")
        print(f"[asr q32 {i}/{len(q32_clips)}] {c.id} lp={rec['avg_logprob']:.3f} {rec['text'][:80]!r}", flush=True)

    del asr
    gc.collect()

    # ---- Aggregate ----
    def pack(condition_keys):
        per = []
        pairs = []
        logprobs = []
        sdrs = []
        for c in selected if condition_keys[0].startswith("orig") or condition_keys[0].startswith("q8") else q32_clips:
            # handled below more clearly
            pass
        return None

    per_orig = []
    per_q8 = []
    pairs_orig = []
    pairs_q8 = []
    for c in selected:
        o = asr_results[f"orig:{c.id}"]
        r = asr_results[f"q8:{c.id}"]
        so = utterance_scores(c.transcription, o["text"])
        sq = utterance_scores(c.transcription, r["text"])
        so.update(
            {
                "id": c.id,
                "gender": c.gender,
                "duration_s": c.duration_s,
                "avg_logprob": o["avg_logprob"],
                "hyp_raw": o["text"],
            }
        )
        sq.update(
            {
                "id": c.id,
                "gender": c.gender,
                "duration_s": c.duration_s,
                "avg_logprob": r["avg_logprob"],
                "hyp_raw": r["text"],
                "orig_wer": so["wer"],
                "orig_cer": so["cer"],
                "orig_hyp_norm": so["hyp_norm"],
                "si_sdr": q8_meta[c.id]["si_sdr"],
                "peak_snr_db": q8_meta[c.id]["peak_snr_db"],
            }
        )
        sq["note"] = word_diff_note(c.transcription, o["text"], r["text"])
        per_orig.append(so)
        per_q8.append(sq)
        pairs_orig.append((c.transcription, o["text"]))
        pairs_q8.append((c.transcription, r["text"]))

    def macro(per, key):
        return mean_ignore_nan(p[key] for p in per)

    metrics = {
        "original": {
            "corpus": corpus_wer_cer(pairs_orig),
            "macro": {"wer": macro(per_orig, "wer"), "cer": macro(per_orig, "cer")},
            "mean_avg_logprob": macro(per_orig, "avg_logprob"),
            "per_utt": per_orig,
        },
        "q8": {
            "corpus": corpus_wer_cer(pairs_q8),
            "macro": {"wer": macro(per_q8, "wer"), "cer": macro(per_q8, "cer")},
            "mean_avg_logprob": macro(per_q8, "avg_logprob"),
            "mean_si_sdr": mean_ignore_nan(q8_meta[c.id]["si_sdr"] for c in selected),
            "mean_peak_snr_db": mean_ignore_nan(q8_meta[c.id]["peak_snr_db"] for c in selected),
            "per_utt": per_q8,
        },
    }

    # 8 vs 32 on the 10-clip subset
    pairs32_o, pairs32_8, pairs32_32 = [], [], []
    per32 = []
    for c in q32_clips:
        o = asr_results[f"orig:{c.id}"]
        a8 = asr_results[f"q8:{c.id}"]
        a32 = asr_results[f"q32:{c.id}"]
        so = utterance_scores(c.transcription, o["text"])
        s8 = utterance_scores(c.transcription, a8["text"])
        s32 = utterance_scores(c.transcription, a32["text"])
        pairs32_o.append((c.transcription, o["text"]))
        pairs32_8.append((c.transcription, a8["text"]))
        pairs32_32.append((c.transcription, a32["text"]))
        per32.append(
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
                "ref_norm": so["ref_norm"],
                "hyp_orig": so["hyp_norm"],
                "hyp_q8": s8["hyp_norm"],
                "hyp_q32": s32["hyp_norm"],
            }
        )

    def pack_subset(pairs, per, lp_key, sdr_key=None, rtf_block=None):
        out = {
            "corpus": corpus_wer_cer(pairs),
            "macro": {
                "wer": mean_ignore_nan(p[lp_key.replace("avg_logprob", "wer")] if False else None),
            },
        }
        return out

    metrics["q32_vs_q8"] = {
        "audio_s": sum(c.duration_s for c in q32_clips),
        "original": {
            "corpus": corpus_wer_cer(pairs32_o),
            "macro": {
                "wer": mean_ignore_nan(p["wer_orig"] for p in per32),
                "cer": mean_ignore_nan(p["cer_orig"] for p in per32),
            },
            "mean_avg_logprob": mean_ignore_nan(p["avg_logprob_orig"] for p in per32),
        },
        "q8": {
            "corpus": corpus_wer_cer(pairs32_8),
            "macro": {
                "wer": mean_ignore_nan(p["wer_q8"] for p in per32),
                "cer": mean_ignore_nan(p["cer_q8"] for p in per32),
            },
            "mean_avg_logprob": mean_ignore_nan(p["avg_logprob_q8"] for p in per32),
            "mean_si_sdr": mean_ignore_nan(p["si_sdr_q8"] for p in per32),
            "encode_rtf": timing["q8"]["encode_rtf"],  # overall 80-clip RTF; also store subset times
            "decode_rtf": timing["q8"]["decode_rtf"],
        },
        "q32": {
            "corpus": corpus_wer_cer(pairs32_32),
            "macro": {
                "wer": mean_ignore_nan(p["wer_q32"] for p in per32),
                "cer": mean_ignore_nan(p["cer_q32"] for p in per32),
            },
            "mean_avg_logprob": mean_ignore_nan(p["avg_logprob_q32"] for p in per32),
            "mean_si_sdr": mean_ignore_nan(p["si_sdr_q32"] for p in per32),
            "encode_rtf": timing["q32"]["encode_rtf"],
            "decode_rtf": timing["q32"]["decode_rtf"],
        },
        "per_utt": per32,
    }
    # Subset-specific 8q RTF (more honest for the table)
    enc8_sub = sum(q8_meta[c.id]["encode_s"] for c in q32_clips)
    dec8_sub = sum(q8_meta[c.id]["decode_s"] for c in q32_clips)
    aud8_sub = sum(len(orig_audio[c.id]) / SR for c in q32_clips)
    metrics["q32_vs_q8"]["q8"]["encode_rtf"] = enc8_sub / aud8_sub if aud8_sub else float("nan")
    metrics["q32_vs_q8"]["q8"]["decode_rtf"] = dec8_sub / aud8_sub if aud8_sub else float("nan")

    d8 = metrics["q32_vs_q8"]["q8"]["corpus"]["wer"] - metrics["q32_vs_q8"]["original"]["corpus"]["wer"]
    d32 = metrics["q32_vs_q8"]["q32"]["corpus"]["wer"] - metrics["q32_vs_q8"]["original"]["corpus"]["wer"]
    if d32 + 1e-9 < d8:
        qnote = (
            f"On these 10 clips, 32 quantizers reduced the WER rise versus 8 quantizers "
            f"({d8*100:+.2f} points at 8q, {d32*100:+.2f} points at 32q). "
            "If the gap is large, some Italian detail sits in the upper codebooks that caramel's 8-codebook recipe discards."
        )
    elif abs(d32 - d8) < 0.02:
        qnote = (
            f"On these 10 clips, 32 quantizers did not clearly beat 8 on WER "
            f"({d8*100:+.2f} points at 8q, {d32*100:+.2f} points at 32q). "
            "SI-SDR may still improve at 32q even when ASR WER is flat."
        )
    else:
        qnote = (
            f"On these 10 clips, 32 quantizers did not improve corpus WER over 8 "
            f"({d8*100:+.2f} points at 8q, {d32*100:+.2f} points at 32q). "
            "The subset is small, so treat this as a diagnostic, not a ranking."
        )
    metrics["q32_vs_q8"]["note"] = qnote

    metrics["listen_notes"] = collect_listen_notes(
        sorted(per_q8, key=lambda r: (-(r["wer"] - r["orig_wer"]), -r["wer"])),
        per_q8,
    )

    # Pick 2 listening examples: one typical (WER delta closest to median, preferably not zero-error both),
    # and the worst WER-delta clip. Copy trimmed pairs into examples/.
    deltas = sorted(per_q8, key=lambda r: (r["wer"] - r["orig_wer"], r["id"]))
    median_d = statistics.median([r["wer"] - r["orig_wer"] for r in per_q8])
    typical = min(per_q8, key=lambda r: (abs((r["wer"] - r["orig_wer"]) - median_d), -r["duration_s"]))
    worst_ex = max(per_q8, key=lambda r: ((r["wer"] - r["orig_wer"]), r["wer"], -r["duration_s"]))
    if typical["id"] == worst_ex["id"] and len(per_q8) > 1:
        typical = min(
            [r for r in per_q8 if r["id"] != worst_ex["id"]],
            key=lambda r: (abs((r["wer"] - r["orig_wer"]) - median_d), -r["duration_s"]),
        )

    examples = []
    for label, row in (("typical (median WER change)", typical), ("worst WER change", worst_ex)):
        cid = row["id"]
        y = orig_audio[cid]
        recon, sr = load_mono(RECON8_DIR / f"{cid}.wav")
        if sr != SR:
            recon = to_24k_mono(recon, sr)
        n = min(len(y), len(recon))
        orig_p = EXAMPLES_DIR / f"{label.split()[0]}_{cid}_orig.wav"
        recon_p = EXAMPLES_DIR / f"{label.split()[0]}_{cid}_recon_q8.wav"
        # sanitize filenames
        tag = "typical" if label.startswith("typical") else "worst"
        orig_p = EXAMPLES_DIR / f"{tag}_{cid}_orig.wav"
        recon_p = EXAMPLES_DIR / f"{tag}_{cid}_recon_q8.wav"
        save_wav(orig_p, y[:n])
        save_wav(recon_p, recon[:n])
        examples.append(
            {
                "label": label,
                "id": cid,
                "gender": row["gender"],
                "duration_s": row["duration_s"],
                "orig_wer": row["orig_wer"],
                "wer": row["wer"],
                "si_sdr": row["si_sdr"],
                "orig_path": str(orig_p),
                "recon_path": str(recon_p),
            }
        )
    metrics["examples"] = examples

    # Drop bulky per-utt text from the written metrics? Keep it — it's the record.
    # But write a compact JSON (ensure NaN -> null)
    def sanitise(obj):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: sanitise(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitise(v) for v in obj]
        return obj

    METRICS_PATH.write_text(json.dumps(sanitise(metrics), ensure_ascii=False, indent=2), encoding="utf-8")
    TIMING_PATH.write_text(json.dumps(timing, indent=2), encoding="utf-8")
    write_report(manifest, metrics, asr_results, timing)
    print("DONE", flush=True)
    print("Verdict will be in", REPORT_PATH, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
