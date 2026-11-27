# Caramel Italiano — Mimi codec test (Italian, CPU)

_Run date: Friday 27 Nov 2026, Europe/Zurich. Box CPU only; no GPU rental, no accounts, no spending._

## Verdict

**BORDERLINE, leaning against treating 8 codebooks as a free pass.** Mimi can preserve Italian — but not well enough at the 8-codebook budget caramel-1 uses. On 80 FLEURS Italian read utterances, whisper-small corpus WER goes from **19.42% on the original audio to 37.55% after 8-quantizer Mimi** (+18.13 points); CER from 10.67% to 23.21%. Mean ASR log-prob also drops (−0.290 → −0.472), and waveform SI-SDR is only 3.18 dB. The 10-clip diagnostic is the sharper result: **32 quantizers cut that WER rise from +19.7 to +5.0 points and lift SI-SDR from 3.5 dB to 11.0 dB.** So the codec family is usable; the 8-codebook recipe is leaving Italian detail in the upper codebooks it throws away. That is enough to justify a small Italian pipeline/pilot, not enough to rent a long training run on 8q alone. Conversation audio was not tested.

This is a content-preservation test of the codec caramel-1 uses as its tokenizer, not a listening-test MOS and not a legal or training-data decision. The 8-codebook setting is the one in `caramel-ref/generate.py` (`MimiModel.from_pretrained('kyutai/mimi')`, `num_quantizers=8`, 24 kHz).

## Headline numbers

| Condition | Corpus WER | Corpus CER | Macro WER | Macro CER | Mean ASR avg log-prob | Mean SI-SDR (dB) |
|---|---:|---:|---:|---:|---:|---:|
| Original 24 kHz audio | 19.42% | 10.67% | 19.44% | 10.27% | -0.290 | — |
| Mimi reconstruction, 8 quantizers | 37.55% | 23.21% | 35.76% | 20.79% | -0.472 | 3.18 |
| Absolute change (recon − original) | +18.13 pts | +12.53 pts | +16.32 pts | +10.52 pts | -0.182 | — |

- Clips: **80** FLEURS Italian test utterances (38 male, 42 female; seed 42).
- Audio duration (original): **1033.7 s** (17.2 min).
- CPU encode RTF (8q): **0.030×**  (30.6 s encode / 1033.7 s audio).
- CPU decode RTF (8q): **0.062×**  (63.7 s decode).
- CPU encode+decode RTF (8q): **0.091×**.
- RTF is wall-clock / audio duration on this box (8-core Intel Xeon, CPU, no GPU). Values < 1 mean faster than real time.
- Sensitivity: three reconstructions triggered ASR collapse (one empty transcript, one truncated to a single word, one looping plus English insertions). Those three clips alone do not create the gap. Excluding them (77 clips), corpus WER is still **19.40% → 32.43%** (+13.03 points) and CER **10.59% → 18.07%**. Median per-clip WER change across all 80 is +10.53 points.
- Split: 38 male / 42 female. Male originals were harder for this ASR (corpus WER 24.3% vs 14.9%), but 8q WER ends up similar (37.1% vs 38.0%). Mean SI-SDR is worse on male clips (2.20 dB vs 4.06 dB).

## Data used

| Set | What was used | Source | Licence | Notes |
|---|---|---|---|---|
| FLEURS Italian test (`it_it`) | 80 read utterances with official transcripts, drawn from the 865-clip test split (seed 42) | [google/fleurs](https://huggingface.co/datasets/google/fleurs) test TSV + `data/it_it/audio/test.tar.gz`; upstream archive also at `https://storage.googleapis.com/xtreme_translations/FLEURS102/it_it.tar.gz` | **CC BY 4.0** (dataset card) | Single-speaker read speech, originally 16 kHz mono WAV, resampled here to 24 kHz mono with soxr VHQ. Train/dev were not downloaded. The official normalized `transcription` field is the WER/CER reference.
| DataHive Italian Conversations sample | **Not used** | [datahiveai/italian-conversations-sample](https://huggingface.co/datasets/datahiveai/italian-conversations-sample) | CC BY 4.0 (card; no login, not gated) | Downloads without signup, but the six files are full conversations (7–55 min, 7–52 MB) and the published metadata has no utterance-level transcripts. Skipped to avoid a large download that cannot enter the WER/CER table.
| Multilingual LibriSpeech Italian test | **Not used** | OpenSLR 94 / CC BY 4.0 | CC BY 4.0 | The public Italian archive is a single large language tarball; FLEURS test already gave transcribed read speech in a smaller download.

Attribution: FLEURS (Conneau et al., 2022), CC BY 4.0. Mimi weights: Kyutai, [kyutai/mimi](https://huggingface.co/kyutai/mimi), CC BY 4.0.

### How the 80 clips were chosen

The FLEURS `it_it` test TSV has 865 clips / 346 unique sentences, but only 38 male readings (the rest female). Duration range in the full test split is 5.28–52.92 s. Filters: duration 5–25 s (excludes 42 clips longer than 25 s; no clip is under 5 s). Then: every eligible male reading (38), plus 42 female readings sampled with `random.Random(42)`, preferring unique sentence IDs. The test is therefore slightly richer in male speech than the raw split, on purpose.

## Method

1. Load each selected 16 kHz FLEURS WAV, mix to mono (already mono), resample to 24 kHz with soxr `VHQ` (not the bicubic `F.interpolate` used in `generate.py`, which is a poorer audio resampler).
2. Encode and decode with `transformers.MimiModel.from_pretrained('kyutai/mimi')` on CPU in `torch.inference_mode()`, `num_quantizers=8`. Reconstructions are saved as 24 kHz PCM16 WAVs. For scoring, a reconstruction longer than its original is trimmed to the original length (Mimi's causal convs can add a short tail).
3. On a 10-clip subset spanning the duration range, also encode/decode with `num_quantizers=32` (Mimi's full codebook stack) as a diagnostic.
4. Run the same ASR on every original and every reconstruction: `faster-whisper` **whisper-small**, `compute_type='int8'`, `language='it'`, `beam_size=5`, `temperature=0`, no VAD, no conditioning on previous text. OpenAI `whisper-small` via faster-whisper was chosen as feasible on this 15 GiB CPU-only box; `large-v3-turbo` was not used, to avoid memory pressure from other processes. The **difference** between original and reconstructed WER is the codec signal; absolute WER also reflects whisper-small, not just Mimi.
5. WER and CER use the FLEURS normalized transcript as reference. Hypothesis and reference are lowercased, Unicode-normalized, and stripped of punctuation **except apostrophes**. Corpus figures are the primary result (total edits / total words or characters). Macro figures are the unweighted mean of per-utterance scores.
6. ASR confidence: duration-weighted mean of each clip's faster-whisper segment `avg_logprob`. SI-SDR is a scale-invariant reconstruction metric against the 24 kHz original (not a perceptual MOS).

- Mimi parameters loaded: 79,308,609.
- torch 2.15.0+cpu (cpu), transformers 5.24.1, faster-whisper 1.2.1.
- CPU: Intel(R) Xeon(R) Processor, 8 threads. CUDA available: False.

## 8 vs 32 quantizers (10 clips)

Same 10 clips, same ASR and references. Duration of this subset: 122.0 s.

| Condition | Corpus WER | Corpus CER | Macro WER | Macro CER | Mean ASR avg log-prob | Mean SI-SDR (dB) | Encode RTF | Decode RTF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original (same 10) | 13.81% | 7.02% | 13.66% | 6.15% | -0.269 | — | — | — |
| Mimi 8 quantizers | 33.47% | 20.72% | 30.38% | 16.74% | -0.510 | 3.54 | 0.031× | 0.080× |
| Mimi 32 quantizers | 18.83% | 10.39% | 18.56% | 10.94% | -0.259 | 11.00 | 0.028× | 0.031× |

On these 10 clips, 32 quantizers reduced the WER rise versus 8 quantizers (+19.67 points at 8q, +5.02 points at 32q) and more than tripled SI-SDR (3.54 → 11.00 dB). ASR confidence at 32q (−0.259) is back to, or slightly better than, the original (−0.269); at 8q it is much worse (−0.510). The most striking clip is `11067362407565311789`: original WER 0%, 8q WER 100% (ASR returned an empty transcript), 32q WER 8.82%. The 8q reconstruction is not silence (SI-SDR 2.48 dB); the speech is just no longer clear enough for this ASR. 32q is not uniformly better on every clip (two of ten get a slightly higher WER), but the corpus and SI-SDR gaps are large. Conclusion: Italian detail that 8 codebooks discard is still present in Mimi's upper codebooks.

Per-clip WER on this diagnostic subset:

| Clip | Duration (s) | Gender | WER orig | WER 8q | WER 32q | SI-SDR 8q | SI-SDR 32q |
|---|---:|---|---:|---:|---:|---:|---:|
| `11067362407565311789` | 15.96 | MALE | 0.00% | 100.00% | 8.82% | 2.48 | 10.61 |
| `13393593335775648938` | 8.46 | FEMALE | 16.67% | 33.33% | 27.78% | 6.77 | 13.90 |
| `14309954849873750903` | 14.88 | MALE | 38.71% | 41.94% | 41.94% | 2.60 | 10.27 |
| `14350779863296164175` | 16.86 | FEMALE | 6.67% | 10.00% | 10.00% | 5.44 | 11.98 |
| `15019600377627773647` | 13.98 | FEMALE | 5.26% | 15.79% | 5.26% | 4.23 | 10.47 |
| `1713867707548638191` | 10.68 | MALE | 13.04% | 39.13% | 56.52% | 2.32 | 10.32 |
| `17285555551155021824` | 9.60 | MALE | 10.00% | 15.00% | 5.00% | 3.77 | 11.07 |
| `5220459858621949080` | 7.02 | MALE | 14.29% | 28.57% | 14.29% | 3.67 | 11.59 |
| `6170658304757950439` | 11.58 | MALE | 0.00% | 8.00% | 16.00% | 1.93 | 9.86 |
| `8624883642038355529` | 13.02 | MALE | 32.00% | 12.00% | 0.00% | 2.15 | 9.98 |

## Listen notes (worst reconstructions)

These notes come from the largest ASR degradations (recon WER minus original WER) plus the word-level diffs. They are not a formal listening panel.

### 1. `16591664060905172157` — female, 16.6 s; WER 32.43% → 191.89% (CER 27.48% → 172.52%)

- Reference: ma non si tratta solo della sperimentazione posto che un esperimento è un test che si usa per escludere uno o più delle possibili ipotesi anche porre domande e fare osservazioni consente di orientare la ricerca scientifica
- ASR on original: la stessa è un test che si usa per escludere uno o più delle possibili ipotesi anche porre domande e far osservazioni consente di orientare la ricerca scientifica
- ASR on 8q reconstruction: ma non si tratta solo della sperimentazione posto che un esperimento è un test che si usa per escludenare una o più di più delle possibili ipotesi anche spellentation but non si tratta solo della sperimentazione ma non si tratta solo della sperimentazione ma non si tratta solo della sperimentazione ma non si tratta solo della sperimentazione ma non si tratta solo della sperimentaimentazione ma non si tratta solo della sperimentazione ma non si tratta solo della sperimentazione té un esperimentazione ma non si tratta solo della sperimentazione ma non si tratta solo della sperimentazione
- SI-SDR: 3.60 dB.
- After the codec, ASR dropped reference words it had gotten on the original: “escludere”, “uno”, “porre”, “domande”, “e”, “osservazioni”, “consente”, “orientare” (+3 more). ASR invented new tokens on the reconstruction: “escludenare”, “una”, “spellentation”, “but”, “sperimentaimentazione”, “té”.

### 2. `11067362407565311789` — male, 16.0 s; WER 0.00% → 100.00% (CER 0.00% → 100.00%)

- Reference: in sostanza wales ci ha mentito sin dall'inizio innanzitutto comportandosi come se fosse per motivi legali in secondo luogo fingendo di darci ascolto fino al momento della effettiva cancellazione delle opere da parte sua
- ASR on original: in sostanza wales ci ha mentito sin dall'inizio innanzitutto comportandosi come se fosse per motivi legali in secondo luogo fingendo di darci ascolto fino al momento della effettiva cancellazione delle opere da parte sua
- ASR on 8q reconstruction: 
- SI-SDR: 2.48 dB.
- After the codec, ASR dropped reference words it had gotten on the original: “in”, “sostanza”, “wales”, “ci”, “ha”, “mentito”, “sin”, “dall'inizio” (+26 more).

### 3. `11916947524328675871` — female, 19.0 s; WER 4.88% → 97.56% (CER 1.17% → 99.22%)

- Reference: la prima delle sue 78 raccomandazioni riguarda l'avvio di una nuova iniziativa diplomatica da intraprendere entro la fine di quest'anno con lo scopo di proteggere i confini dell'iraq da azioni ostili e di riprendere i rapporti diplomatici con i suoi vicini
- ASR on original: la prima delle suoi 78 raccomandazioni riguarda l'avvio di una nuova iniziativa diplomatica da intraprendere entro la fine di quest'anno con lo scopo di proteggere i confini dell'irac da azioni ostili e di riprendere i rapporti diplomatici con i suoi vicini
- ASR on 8q reconstruction: la
- SI-SDR: 4.54 dB.
- After the codec, ASR dropped reference words it had gotten on the original: “prima”, “delle”, “78”, “raccomandazioni”, “riguarda”, “l'avvio”, “di”, “una” (+29 more).

### 4. `1815772323593604298` — male, 13.6 s; WER 7.41% → 88.89% (CER 2.21% → 67.40%)

- Reference: l'italiano è anche la lingua d'uso quotidiano parlata dalla maggior parte delle persone che lavorano all'interno dello stato mentre il latino è spesso usato nelle funzioni religiose
- ASR on original: l'italiano è anche la lingua d'uso quotidiano parlata dalla maggior parte delle persone che lavorano all'interno dello stato mentre l'attino è spesso usato nelle funzioni religiose
- ASR on 8q reconstruction: l'italiano è anche la lingua d'uso quotidiano parlata dalla maggior parte delle persone che lavorano all l'italiano è anche la lingua d'uso quotidiano parlata dalla maggior parte delle persone che lavorano all'interno del stato mentre l'italiano è spesso usato in le funghi re rigiolo
- SI-SDR: 2.03 dB.
- After the codec, ASR dropped reference words it had gotten on the original: “dello”, “nelle”, “funzioni”, “religiose”. ASR invented new tokens on the reconstruction: “all”, “del”, “in”, “le”, “funghi”, “re”.

### 5. `10479866160281873942` — female, 14.2 s; WER 7.41% → 85.19% (CER 3.83% → 83.06%)

- Reference: l'iniziativa contro le oscenità è stata finanziata dal congresso a partire dall'anno fiscale 2005 con la specifica richiesta all'fbi di assegnare 10 agenti alla pornografia per adulti
- ASR on original: l'iniziativa contro le uscinità è stata finanziata dal congresso a partire dall'anno fiscale 2005 con la specifica richiesta all'fbi di assegnare dieci agenti alla pornografia per adulti
- ASR on 8q reconstruction: l'iniziativa contro le uscinità è
- SI-SDR: 6.06 dB.
- After the codec, ASR dropped reference words it had gotten on the original: “stata”, “finanziata”, “dal”, “congresso”, “a”, “partire”, “dall'anno”, “fiscale” (+13 more).

### Patterns

- The worst cases are not subtle substitutions. They are ASR collapses on 8q audio: empty output (`11067362407565311789`, a clip that was transcribed perfectly from the original), mid-sentence cut-offs (`11916947524328675871` down to “la”; `10479866160281873942` cut after the first clause), and a repetition loop mixed with English (`16591664060905172157`: “spellentation but” then “ma non si tratta solo della sperimentazione” many times). Those are classic signs that the reconstruction is hard to recognise as clean Italian, not just a few wrong vowels.
- A more typical failure, close to the median WER change, is smeared names and clitics rather than a total collapse. On `15019600377627773647` (the “typical” listening pair), the original ASR is almost exact; after 8q it hears “l'annuncio”/“trump” as “l'ann年zo”/“trumpf”. 32q on this same clip returns WER to the original 5.26%.
- Of 80 clips, 8q reconstruction made utterance WER worse on 61, better on 14, and unchanged on 5.
- Per-clip WER change (recon − original): median +10.53 points, mean +16.32 points, max +159.46 points. The 191.89% WER on the worst clip is an ASR loop (many inserted words), not a 2×-longer recording.
- Reference words most often lost by ASR only after the codec (correct on the original, missing on the reconstruction): “di” (7), “e” (6), “in” (6), “alla” (5), “con” (5), “da” (5), “come” (4), “delle” (4). Function words and clitics dominate this list in part because collapses drop whole suffixes of the sentence.
- SI-SDR of 8q reconstructions vs the 24 kHz originals: mean 3.18 dB, median 2.68 dB, min −0.72 dB. Neural codecs often score modestly on SI-SDR; the contrast with 11.0 dB at 32q on the diagnostic subset is the more useful number.
- Several dropped tokens are elided Italian forms (apostrophe), a useful stress test of short vowels and clitics: “dall'anno”, “all'fbi”, “dall'inizio”, “l'avvio”, “quest'anno”, “dell'evento”, “l'annuncio”, “l'inquinamento”.

## Example pairs for listening

Two 24 kHz WAV pairs were written under `examples/` (relative to this file, `eval/codec_test/examples/`). Each pair is the original and the 8-quantizer reconstruction, trimmed to the same length. These are the clips to listen to. The WAVs are not in this repository. In the original run they lived at `/workspace/caramel-italiano/codec-test/examples/`.

- **typical (median WER change)** (`15019600377627773647`, female, 14.0 s, WER 5.26% → 15.79%, SI-SDR 4.23 dB): `examples/typical_15019600377627773647_orig.wav` and `examples/typical_15019600377627773647_recon_q8.wav`. Reference: “l'annuncio è stato fatto a seguito di un colloquio telefonico tra trump e il presidente turco recep tayyip erdoğan.” This is the everyday error mode (smeared names/clitics), not a collapse.
- **worst WER change** (`16591664060905172157`, female, 16.6 s, WER 32.43% → 191.89%, SI-SDR 3.60 dB): `examples/worst_16591664060905172157_orig.wav` and `examples/worst_16591664060905172157_recon_q8.wav`. The 8q ASR transcript loops and inserts English; listen to whether the audio itself repeats or merely becomes mushy enough to confuse the recogniser.

## Files

- `manifest.json` — clip IDs, sources, licences, transcripts, 8-vs-32 flag
- `originals_24k/` — 80 resampled originals
- `recon_q8_24k/` — 80 reconstructions at 8 quantizers
- `recon_q32_24k/` — 10 reconstructions at 32 quantizers
- `asr_results.json`, `metrics.json`, `timing.json` — raw numbers
- `examples/` — the two listening pairs

## Caveats

- Read speech only (FLEURS). This does **not** test overlapping talk, backchannels, telephone channel, or informal conversation — the genre caramel actually models. A pass here is necessary but not sufficient.
- whisper-small will itself mishear some Italian (original corpus WER is already 19.4%). Compare original vs recon, not the absolute WER alone. A stronger ASR might shrink both numbers; it would not erase the 8q-vs-32q gap seen here.
- There is no English control run on this box, so this does not prove that Italian is *harder* for 8q Mimi than English. It only shows that 8q already loses a lot of Italian lexical detail relative to 32q and to the original audio.
- One 80-clip sample, one seed, one ASR. No claim about other Italian accents, spontaneous speech, or music/noise.
- Mimi was run in full fp32 on CPU via Hugging Face `transformers` (v5.24.1), matching the public API used in caramel, not a quantized runtime.

---

## Follow-up: English baseline and stronger Italian ASR

_Appended Friday 27 Nov 2026, Europe/Zurich. Still CPU only; no GPU rental, no accounts, no spending._

### One-line verdict

**YES, WITH CAUTION — 8q Mimi is OK for a small Italian pilot, not a long paid train.** After matching loudness, English also loses lexical detail at 8 codebooks (whisper-small WER +10.5 points, +5.0 after dropping one ASR collapse). Italian is somewhat worse on the same ASR (+18.1 points, median +10.5 vs English +3.3), but a stronger ASR (large-v3-turbo) cuts the Italian rise to +8.2 points, or +2.6 after three collapses. Waveform SI-SDR is essentially tied (~3.2–3.4 dB at 8q; ~9–11 dB at 32q). 32q remains clearly cleaner. Conversation audio is still untested.

**Is the Italian 8q degradation meaningfully worse than English?** Somewhat, on lexical WER with the same ASR; no, on waveform fidelity. This is a general 8-codebook loss that shows up more in Italian word error under whisper-small, not an Italian-only codec breakdown.

### Level-matching (important)

An unnormalized English run is **not** a fair language comparison and is not used in the table below. The FLEURS `en_us` test wavs in this archive have wildly inconsistent levels: 36 of the 80 selected clips have peak amplitude below 0.01 (median peak −35 dBFS). Every Italian clip already sits in a healthy range (peaks −16 to −4 dBFS, all ≥ 0.16). On the quiet English files, 8q Mimi + whisper-small collapsed (loops, empty transcripts, mean SI-SDR −7.9 dB, apparent WER rise +48 points). That was a gain problem, not a language effect.

The English numbers below are a **re-run of the same 80 clips** after peak-normalizing each utterance to 0.9 (median gain 50.8×). Italian was left unchanged (already loud enough). Protocol is otherwise identical: Mimi 8q on all 80, 32q on the same 10-clip duration grid, faster-whisper `small` int8, beam 5, temperature 0, no VAD, seed 42, 38 male + 42 female, duration 5–25 s. English audio: 830.3 s (13.8 min). Source: [google/fleurs](https://huggingface.co/datasets/google/fleurs) `en_us` test, **CC BY 4.0**. Train/dev were not downloaded.

### Side-by-side: Italian vs English (fair levels)

| Language | ASR | Condition | Corpus WER | Corpus CER | Macro WER | Macro CER | Mean ASR avg log-prob | Mean SI-SDR (dB) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Italian | whisper-small int8 | Original | 19.42% | 10.67% | 19.44% | 10.27% | −0.290 | — |
| Italian | whisper-small int8 | Mimi 8q | 37.55% | 23.21% | 35.76% | 20.79% | −0.472 | 3.18 |
| Italian | whisper-small int8 | Change (8q − orig) | +18.13 pts | +12.53 pts | +16.32 pts | +10.52 pts | −0.182 | — |
| English | whisper-small int8 | Original (peak-norm) | 10.98% | 7.02% | 11.80% | 7.48% | −0.345 | — |
| English | whisper-small int8 | Mimi 8q (peak-norm) | 21.46% | 13.67% | 22.17% | 15.62% | −0.355 | 3.37 |
| English | whisper-small int8 | Change (8q − orig) | +10.48 pts | +6.64 pts | +10.37 pts | +8.14 pts | −0.010 | — |

- Same-ASR WER rise: Italian +18.13 vs English +10.48 points (Italian worse by 7.65). Median per-clip WER change: Italian **+10.53** vs English **+3.30**. Mean SI-SDR is tied (3.18 vs 3.37 dB).
- Collapses (WER ≥ 100% or ≤ 1 word): Italian 3/80, English 1/80 after peak-norm. Excluding them: Italian 19.40% → 32.43% (**+13.03** pts, 77 clips); English 11.06% → 16.08% (**+5.03** pts, 79 clips).
- Clip counts at 8q: Italian worse/better/unchanged = 61/14/5; English 45/16/19.
- English CPU RTF (8q, peak-normed audio): encode 0.041×, decode 0.089×, round-trip 0.130×.
- Absolute WER is still not comparable across languages (whisper-small is stronger on English). The **delta**, the median, and SI-SDR are the codec signals.

### 8 vs 32 quantizers, 10 clips each

| Language | Condition | Corpus WER | Corpus CER | Mean ASR avg log-prob | Mean SI-SDR (dB) |
|---|---|---:|---:|---:|---:|
| Italian | Original (same 10) | 13.81% | 7.02% | −0.269 | — |
| Italian | Mimi 8q | 33.47% | 20.72% | −0.510 | 3.54 |
| Italian | Mimi 32q | 18.83% | 10.39% | −0.259 | 11.00 |
| Italian | 8q − orig | +19.67 pts | +13.70 pts | −0.241 | — |
| Italian | 32q − orig | +5.02 pts | +3.38 pts | +0.010 | — |
| English (peak-norm) | Original (same 10) | 8.14% | 4.81% | −0.282 | — |
| English (peak-norm) | Mimi 8q | 9.95% | 5.13% | −0.257 | 3.73 |
| English (peak-norm) | Mimi 32q | 19.91% | 17.24% | −0.380 | 9.41 |
| English (peak-norm) | 8q − orig | +1.81 pts | +0.32 pts | +0.025 | — |
| English (peak-norm) | 32q − orig | +11.76 pts | +12.43 pts | −0.098 | — |

The English 32q **corpus WER is misleading**. One of the ten clips (`13388880723958450388`) was transcribed as a single word “All” at 32q (WER 14.3% → 96.4%) and dominates this small set. On the other nine, 32q WER is equal or better than 8q on seven clips. The clean 32q signal is SI-SDR: English 3.73 → 9.41 dB, Italian 3.54 → 11.00 dB. In both languages, the upper codebooks restore waveform detail that 8q discards.

English per-clip 8q vs 32q WER (peak-normalized):

| Clip | Duration (s) | Gender | WER orig | WER 8q | WER 32q | SI-SDR 8q | SI-SDR 32q |
|---|---:|---|---:|---:|---:|---:|---:|
| `1116401173651214130` | 11.40 | MALE | 11.11% | 0.00% | 0.00% | 4.01 | 10.23 |
| `1146643745158029931` | 8.88 | FEMALE | 9.09% | 13.64% | 9.09% | 3.00 | 7.88 |
| `13388880723958450388` | 9.48 | FEMALE | 10.71% | 14.29% | 96.43% | 3.29 | 10.00 |
| `1497068838470674354` | 13.08 | MALE | 6.45% | 3.23% | 12.90% | 4.21 | 7.69 |
| `2786302614981736925` | 5.52 | FEMALE | 0.00% | 26.67% | 26.67% | 4.16 | 9.73 |
| `4491697341889976683` | 8.28 | FEMALE | 18.18% | 0.00% | 9.09% | 3.75 | 8.84 |
| `6656552880161745727` | 7.14 | FEMALE | 0.00% | 0.00% | 0.00% | 3.74 | 10.58 |
| `814068542761107022` | 10.34 | MALE | 6.67% | 33.33% | 6.67% | 1.98 | 6.41 |
| `9781076737390802637` | 6.28 | MALE | 0.00% | 7.69% | 0.00% | 6.40 | 12.00 |
| `978391636333640472` | 16.62 | MALE | 10.81% | 10.81% | 10.81% | 2.76 | 10.73 |

### Stronger ASR on the Italian 8q set

The same 80 Italian originals and 8q reconstructions were re-scored with faster-whisper **large-v3-turbo** int8, `language='it'`, same beam/temperature/VAD settings, on CPU. Load succeeded; wall time 527.2 s, RTF 0.255× over 2067.5 s of original+8q audio. This fits comfortably on this box.

Re-scoring shrinks the Italian 8q gap substantially (whisper-small +18.13 → turbo +8.21 WER points; CER +12.53 → +2.94). Mean avg log-prob no longer drops (−0.324 → −0.317). So a large share of the original +18 point result **was whisper-small fragility**, especially the empty/truncated/looped transcripts. A residual 8q loss remains: it is not zero, and it is not only the small model.

| ASR | Condition | Corpus WER | Corpus CER | Macro WER | Macro CER | Mean ASR avg log-prob |
|---|---|---:|---:|---:|---:|---:|
| whisper-small int8 | Original | 19.42% | 10.67% | 19.44% | 10.27% | −0.290 |
| whisper-small int8 | Mimi 8q | 37.55% | 23.21% | 35.76% | 20.79% | −0.472 |
| large-v3-turbo int8 | Original | 19.26% | 13.47% | 20.47% | 14.91% | −0.324 |
| large-v3-turbo int8 | Mimi 8q | 27.47% | 16.40% | 28.03% | 17.17% | −0.317 |

- Turbo median per-clip WER change: **+4.81** points (mean +7.56; 44 worse, 25 better, 11 unchanged).
- Excluding 3 turbo collapse clips: corpus WER 19.63% → 22.25% (**+2.62** pts) on 77 clips; CER 13.87% → 13.11% (slightly better).
- Worst turbo 8q cases are still collapses, not mild substitutions: `4023986607040221909` loops “ed ed ed…” (12.5% → 231.2%); `16891835188027563938` returns almost nothing (5.6% → 100%); `5478665049804738217` stutters “piuola di piuola…” (0% → 93.3%). These are rare (3/80) but real.

### How to read this for the pilot

- 8q Mimi is a lossy tokenizer in **both** languages. Matched-level English is the right baseline; the unnormalized English run should be ignored.
- Italian is a bit more fragile lexically under whisper-small. Under large-v3-turbo the residual is small enough to try a short Italian pilot, and not small enough to assume 8q is transparent.
- 32q still helps both languages on SI-SDR. If a later recipe can afford more codebooks, it is the obvious upgrade.
- Read speech only. Overlap, backchannels, and telephone channel are still untested.
- Files: English peak-norm wavs and metrics in `en_us/peaknorm/`; unnormalized English (invalid as a language comparison, kept for the loudness diagnosis) in `en_us/`; Italian turbo transcripts in `asr_italian_large_v3_turbo.json` and `metrics_italian_large_v3_turbo.json`.

