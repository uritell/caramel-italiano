# eval/codec_test/ — Mimi 8-codebook test (completed)

CPU-only test of Kyutai Mimi as caramel-1 uses it
(`MimiModel.from_pretrained('kyutai/mimi')`, `num_quantizers=8`, 24 kHz)
on FLEURS Italian, with an English FLEURS comparison and a stronger
Italian ASR follow-up.

## Verdict (for the $50 pilot)

**Good enough to proceed, with caution.** 8-codebook Mimi is a lossy
tokenizer in both languages, not an Italian-only failure. On 80 FLEURS
`it_it` test clips, Whisper large-v3-turbo word error went from 19.3%
to 27.5%; excluding three clips that collapsed the recogniser, the rise
is **+2.6 points**. 32 codebooks preserves much more (SI-SDR and WER).
Read speech only; overlap and telephone channel were not tested.

The full write-up is `report.md` (copied from the original run). Example
WAV paths in that report are relative to this folder (`examples/...`).
The original run stored those files under
`/workspace/caramel-italiano/codec-test/examples/`. Example WAVs and JSON
dumps are not in this tree (they are audio / bulky artefacts and are
gitignored).

## Scripts

Copied from the original codec-test directory, with only the working
directory and Hugging Face cache paths changed so they default to this
folder (`Path(__file__).parent`) and `HF_HOME` (or `./.hf-cache`),
instead of hardcoded `/workspace/caramel-italiano/...` paths.

| Script | Role |
|---|---|
| `run_codec_test.py` | Italian FLEURS sample, Mimi 8q (+ 10-clip 32q diagnostic), whisper-small ASR, `report.md` |
| `run_followup.py` | English FLEURS baseline + large-v3-turbo on the existing Italian 8q set; appends to the report |
| `run_en_peaknorm.py` | Re-runs the English comparison after peak-normalizing clips to 0.9 (the fair language baseline) |

Dependencies (on top of a torch / transformers environment):
`soundfile`, `soxr`, `numpy`, `jiwer`, `faster-whisper`.

The scripts will download FLEURS test audio, Mimi weights, and Whisper
models on first run. That is free and uses Hugging Face; it is not a
GPU rental. They do not need to be re-run to use the existing report.
