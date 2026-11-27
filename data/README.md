# data/ — download and manifests

Raw audio stays out of git, under `data/raw/` and `data/work/`.

This step downloads three sources. All three are public, need no account,
and are CC BY 4.0. Common Voice is intentionally not fetched: it has to
come from Mozilla Data Collective, which needs an account.

| ID | What the script fetches | Licence |
|---|---|---|
| `datahive_it_conversations_sample` | Hugging Face `datahiveai/italian-conversations-sample`, the six stereo mp3s and `metadata.jsonl` | CC BY 4.0, attribute DataHive. Caller is left, callee is right. |
| `fleurs_it_it` | Hugging Face `google/fleurs`, `data/it_it` train/dev/test tsv and audio archives, then the wavs | CC BY 4.0. Gender is recorded. There is no speaker id in the public tsv, so `speaker_id` is null. |
| `mls_it` | `https://dl.fbaipublicfiles.com/mls/mls_italian.tar.gz` (OpenSLR 94 / the Facebook host). md5 `dc77f5805aecc7182aa20786032e5dc1` from the published `md5sum.txt`. | CC BY 4.0. Speaker id is the directory under `audio/`. |
| `common_voice_it_27` | Refused. | CC0, but an MDC account is required. Do not re-host it. |

```bash
# CPU env. Do not use third_party/caramel/setup.sh for this step:
# that script installs a CUDA wheel and downloads caramel-1 weights.
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -r requirements-pilot.txt

python data/download.py
python data/manifest.py
```

`download.py` with no `--source` fetches the three open sets. Pass
`--source common_voice_it_27` and it exits without downloading.

`manifest.py` writes `data/work/manifest.jsonl`, one JSON object per file:

- `source`, `licence`, `licence_url`, `attribution` (the suggested sentence from `DATA_LICENSES.md`), `source_url`
- `split`, `path` (relative to `data/raw/`), `duration_s` (from the file header), `sample_rate`, `channels`, `layout`
- `speaker_id` when the corpus publishes one, otherwise null
- `text` when a transcript ships with the corpus

`data/work/hours.json` is the hour totals by source and split. Those hours
are measured from the file headers, not copied from the papers. A run on
27 Nov 2026 measured:

| Source | Hours measured | Splits |
|---|---:|---|
| DataHive sample | 3.051 | 6 stereo conversations |
| FLEURS `it_it` | 14.066 | train 8.998, dev 1.547, test 3.521 |
| MLS Italian | 257.826 | train 247.378, dev 5.176, test 5.273 |
| **Total** | **274.944** | |

MLS matches the paper's 257.83 h. FLEURS is the header total; the paper
only says about 12 h per language. DataHive's card says 3 h 3 m 4 s.
Common Voice (about 364 validated hours) is not in this total.
