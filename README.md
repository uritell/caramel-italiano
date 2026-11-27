# Caramel Italiano

Non-commercial, experimental Italian speech-to-speech retrain of [caramel-1](https://github.com/naklecha/caramel).
The model should learn from Italian speech only, with no text in the training
loop, following the caramel recipe: Kyutai [Mimi](https://huggingface.co/kyutai/mimi)
at 12.5 frames/s and 8 codebooks, three-stage training, and a closeness /
timing score fitted on real conversations.

This project is non-commercial and experimental. Speech data may be under
open licences or non-commercial terms such as CC BY-NC. See
[DATA_LICENSES.md](DATA_LICENSES.md).

This repository is the **pipeline and documentation** for that work. The
upstream model, `train.py`, and `generate.py` are not copied here. They
live in a git submodule at `third_party/caramel`, pinned to

`84a9271d265841543287d4044d7ef80991a2b003`

from [https://github.com/naklecha/caramel](https://github.com/naklecha/caramel).
Do not copy `train.py`, `generate.py`, `model.py`, or `kernels.py` out of
that checkout into the rest of this project. Upstream's own README is
[third_party/caramel/README.md](third_party/caramel/README.md) after
`git submodule update --init`.

**Suggested GitHub name:** `caramel-italiano`

**Suggested one-line description:** Non-commercial experimental Italian retrain of caramel-1, starting with a cheap clean-data speech-to-speech pilot.

## Status (27 November 2026, Europe/Zurich)

- This tree is the initial skeleton of the private repository.
- A legal/access audit of Italian speech data is in `docs/data-audit.md`.
  It is research, not legal advice. Nothing was downloaded, bought, or
  signed up for during the audit.
- A **CPU codec test** of Mimi on FLEURS Italian (and an English baseline)
  is in `eval/codec_test/`. Verdict, with caution: 8 codebooks is good
  enough for a short Italian pilot. It is a general 8-codebook trade-off,
  not an Italian-only failure. Conversation audio has not been tested.
- Training has not started. **No GPU time is rented, and nothing that
  costs money happens, without the owner's explicit yes.** The starting
  budget cap is **USD 50**.

## Credits

- **caramel-1**, by Nishant Aklecha ([naklecha/caramel](https://github.com/naklecha/caramel),
  Apache 2.0). Model card: [huggingface.co/naklecha/caramel-1](https://huggingface.co/naklecha/caramel-1).
  This project reuses the architecture, training script, and recipe; it
  does not claim those as original work.
- **Mimi**, the neural audio codec, developed by [Kyutai](https://kyutai.org/)
  ([kyutai/mimi](https://huggingface.co/kyutai/mimi), CC BY 4.0). Please
  cite Défossez et al. (2024), *Moshi: a speech-text foundation model for
  real-time dialogue*, https://arxiv.org/abs/2410.00037.
- Speech data credits and licence conditions are in
  [DATA_LICENSES.md](DATA_LICENSES.md) and [NOTICE](NOTICE).

## Data sources for the clean pilot

The original caramel-1 recipe used on the order of 5,700 hours of
two-person audio with each speaker on their own track, plus about 5,800
hours of one-person speech. Open-licensed Italian conversation audio with
separate tracks comes to only about **3 hours** (one CC BY 4.0 sample).
The owner chose a **clean pilot**, not grey YouTube audio and not paid
corpora:

| Role | Dataset | Hours (as stated at the source) | Licence |
|---|---|---|---|
| Real stereo conversations | [DataHive Italian Conversations sample](https://huggingface.co/datasets/datahiveai/italian-conversations-sample) | 3 h 3 m 4 s (6 conversations) | CC BY 4.0, training allowed, attribution to DataHive |
| Single-speaker source for synthetic two-track dialogues | [FLEURS Italian](https://huggingface.co/datasets/google/fleurs) (`it_it`) | ≈12 h “per language”; IT-specific total not stated | CC BY 4.0 |
| Same | [Multilingual LibriSpeech, Italian](https://www.openslr.org/94/) | 257.83 h (train+dev+test) | CC BY 4.0 |
| Same | [Common Voice Scripted Speech 27.0, Italian](https://mozilladatacollective.com/datasets/cmu5xagm300dyo1075qdhf20m) | 363.9 h validated (430.5 h recorded) | CC0, plus MDC conditions (no speaker identification, no re-hosting, obtain via MDC) |

Together, the open single-speaker sets are about **649 hours** if
FLEURS is counted at the paper's “≈12 h per language” figure. Synthetic
dialogues built from that audio will not reproduce authentic overlap,
backchannels, or informal Italian; they are a bootstrap, mixed with the
3 hours of real stereo speech.

Do not re-host Common Voice. Do not commit audio, tokens, or checkpoints
to this repo (see `.gitignore`).

## Pilot plan

Budget cap: **USD 50**. No rental without a quote and a yes.

1. **Codec check (done, free, CPU).** Mimi 8-codebook reconstructions on
   80 FLEURS Italian clips. With a stronger recogniser (Whisper
   large-v3-turbo), word error went from 19.3% to 27.5% (+8.2 points), or
   **+2.6 points** on 77 clips after dropping three recogniser collapses.
   English, after a fair loudness match, loses comparable audio detail
   (SI-SDR). 32 codebooks is clearly cleaner if a later recipe can afford
   it. See `eval/codec_test/report.md`.
2. **Data.** Download the four pilot sources (scripts in `data/` are
   placeholders). Build a manifest of paths, licences, splits, and
   duration. Common Voice access needs an MDC account, not yet created.
3. **Synthesis (`synth/`).** Build synthetic two-track Italian dialogues
   from the open single-speaker sets (left/right speakers, shared
   timeline), and keep the DataHive sample as the only real stereo
   conversations.
4. **Italian timing score (`score/`).** Rebuild caramel's closeness
   statistics (gaps, overlap, silence, backchannels, and the published
   recording stats) and **fit the target bands on real Italian
   conversations** (the DataHive sample, held-out), not on the English
   Dessa bands. Upstream did not release this scorer.
5. **Tokenize (`tokenize/`).** Encode each speaker with Mimi
   (`kyutai/mimi`, 8 codebooks, 24 kHz) into the `tokens.bin` /
   `index.npy` / `meta.json` layout that `train.py` reads. Format is
   documented in `tokenize/README.md`.
6. **Train (`train/`).** Run the three upstream stages with
   `third_party/caramel/train.py`. The one-pod pilot stops at step 401,
   then 1072, then 1981. See `train/README.md` and `runpod/README.md`.
   `python runpod/launch.py --dry-run` checks the plan and creates
   nothing. `python runpod/launch.py --launch` is the command that rents
   one 2× H100 SXM pod, capped at 2.5 h or about $22. On-demand price
   checks from the planning pass (USD, before tax, 27 Nov 2026): about
   $1.73/h (Vast.ai verified offer), $4.39/h (RunPod H100), $4.89/h
   (Lambda H100). The author's full caramel-1 run was about 4 H100
   GPU-hours. Tokenizing time is measured on the pod before the stages
   run.
7. **Evaluate (`eval/`).** Held-out Italian dev loss, timing / closeness
   stats, ASR word clarity (caramel uses mean ASR log-probability of the
   transcript; this project also reports WER), and listening.

Out of scope for this pilot: matching the full ~11,000-hour English mix,
rewriting GPU kernels, paid or legally grey data, and any spend over $50.

## How to reproduce

Nothing below rents a GPU or downloads a corpus until you run the later,
still-to-be-written download and train commands yourself.

```bash
# 1. Clone and check out the pinned upstream submodule
git clone --recurse-submodules https://github.com/uritell/caramel-italiano.git
cd caramel-italiano
# If the clone omitted submodules:
git submodule update --init third_party/caramel

# 2. The recorded pin (do not move it without a reason)
git -C third_party/caramel rev-parse HEAD
# 84a9271d265841543287d4044d7ef80991a2b003

# 3. Python environment: reuse upstream's setup for train/generate
#    (Python 3.12, PyTorch, transformers). Extra eval deps are listed in
#    eval/codec_test/README.md.
./third_party/caramel/setup.sh ./caramel-1 ./.venv
source ./.venv/bin/activate

# 4. Optional, free, already run once: codec test
python eval/codec_test/run_codec_test.py
# English baseline + stronger Italian ASR (appends to the report):
python eval/codec_test/run_followup.py
python eval/codec_test/run_en_peaknorm.py

# 5. Data, synthesis, tokenization, scoring, training, evaluation:
#    scripts in data/, synth/, tokenize/, score/, train/, and eval/ are
#    stubs or documentation until the pipeline is implemented.
#    Training commands, with pilot step counts left as TODO, are in
#    train/README.md.
```

## Repository layout

| Path | What it is for |
|---|---|
| `data/` | Download and manifest stubs for the four pilot corpora |
| `synth/` | Synthetic two-track dialogue construction |
| `tokenize/` | Mimi encode to the token-store format `train.py` expects |
| `score/` | Italian timing / closeness statistics |
| `eval/` | ASR word clarity, timing stats; `eval/codec_test/` holds the completed codec test |
| `train/` | How to run the three upstream training stages |
| `runpod/` | One-command pilot on a 2× H100 SXM pod, with a dry run that rents nothing |
| `third_party/caramel/` | Upstream caramel, git submodule pinned to `84a9271d265841543287d4044d7ef80991a2b003` |
| `docs/data-audit.md` | Licence and access audit (27 Nov 2026) |
| `DATA_LICENSES.md` | One row per pilot dataset / codec, with attribution text |
| `NOTICE` | Apache 2.0 notice and upstream / Mimi attribution |
| `LICENSE` | Apache License 2.0 |

## Licence

Source code and documentation in this repository are released under the
Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

Dataset audio remains under the licences in `DATA_LICENSES.md` and is
not redistributed here. Mimi weights remain CC BY 4.0 and are downloaded
from Hugging Face at run time, not copied into this tree.
