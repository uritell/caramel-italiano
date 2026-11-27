# data/ — download and manifests

Place raw corpora **outside git**. This directory will hold scripts that
fetch the four clean-pilot sources and write a manifest (path, split,
duration, sample rate, channels, licence, attribution tag). Audio itself
belongs under `data/raw/` or `data/audio/`, both gitignored.

## Sources (pilot only)

| ID | How it is obtained | Notes |
|---|---|---|
| `datahive_it_conversations_sample` | Hugging Face dataset `datahiveai/italian-conversations-sample` | ~3 h stereo, caller left / callee right. CC BY 4.0, attribute DataHive. |
| `fleurs_it_it` | Hugging Face dataset `google/fleurs`, config `it_it` | Read speech, CC BY 4.0. Also used by `eval/codec_test/`. |
| `mls_it` | OpenSLR 94 / the Facebook-hosted MLS Italian tarballs | 257.83 h train+dev+test, CC BY 4.0. |
| `common_voice_it_27` | Mozilla Data Collective, Scripted Speech 27.0 Italian | CC0 plus “no re-host, no speaker identification, obtain via MDC.” **An MDC account is required. Do not create one, and do not download, until the owner says so.** |

Scripts in this folder are placeholders. They must not start a download
in their current form.

Intended later outputs (not present yet):

- `data/work/manifest.jsonl` — one record per clip
- `data/work/splits.json` — train / held-out assignments, including a
  held-out slice of the DataHive conversations for the Italian timing
  score and for `train.py --dev`

See `DATA_LICENSES.md` before adding any fifth source.
