# tokenize/ — Mimi encode to the caramel `train.py` store format

`caramel-1` never trains on raw audio. Each speaker is encoded with
Kyutai Mimi and written to a folder that
`third_party/caramel/train.py` (`TokenStore`) memory-maps. This folder
imports that class to check a store. It does not copy it.

Same settings as `third_party/caramel/generate.py`:

- weights: `kyutai/mimi` (CC BY 4.0; see NOTICE)
- audio: 24 kHz, resampled with the same bicubic `interpolate` call
- `num_quantizers=8`
- frame rate: 12.5 frames per second (80 ms per frame)
- a stereo file is encoded as a batch of two mono waveforms, one per
  channel (`x[:2, None]` in `generate.py`), not as a mix

One hour of one-person speech is `12.5 * 8 * 3600 = 360,000` tokens;
a two-person conversation is twice that.

## Store format (from `train.py`, do not invent fields)

A store is **one folder** containing exactly these three files:

| File | Type and shape | Meaning |
|---|---|---|
| `tokens.bin` | `uint16`, frame-major, shape `[frames, W]` | Codec tokens. **W = 16** for stereo: speaker 1 codebooks 0–7, then speaker 2 codebooks 0–7. **W = 8** for mono (one-person speech). |
| `index.npy` | `int64`, shape `[clips, 2]` | Each row is `(offset, length)` in **frames**, indexing into the first axis of `tokens.bin`. |
| `meta.json` | JSON object | `{"width": W, "frames": <total frames in tokens.bin>}`. `width` must equal `W`. |

For DataHive and for the synthetic wavs, speaker 1 is the left channel
(the caller, on DataHive) and speaker 2 is the right channel (the callee).

`train.py` then windows clips, and `delay_window()` turns a raw
`[L, W]` slice into the delayed 16-wide training sequence. The tokenizer
does not apply that delay. `verify_store.py` imports `TokenStore` and
`delay_window` and runs both.

Up to 90 seconds is encoded in one call. Longer files are encoded in
60 s chunks with 60 s of left context. Mimi's early convolutions keep
an activation per audio sample, so a half-hour stereo waveform asks for
about 20 GB on this CPU build and is killed. Chunking avoids that.
The chunked codes are not identical to a one-shot encode: with 60 s of
context, a 150 s test clip disagrees on about one percent of tokens,
almost all in the higher codebooks. `--check-chunking` prints that
fraction. The smoke conversations are longer than 90 s, so they are
chunked.

```bash
python tokenize/encode_store.py --items items.jsonl --out tokenize/out/mono --width 8
python tokenize/encode_store.py --items items.jsonl --out tokenize/out/stereo --width 16
python tokenize/encode_store.py --check-chunking
python tokenize/verify_store.py --store tokenize/out/mono
```

An items JSONL row is `{"audio": "...", "mode": "mono"|"stereo", "id": "...", "max_seconds": optional}`.
`max_seconds` keeps a slice from the start of a long file. Stereo mode
requires two channels and writes width 16.

The script used to be `tokenize/tokenize.py`. That name shadows the
Python standard library when anything in this directory imports torch,
so the entry point is `encode_store.py`.

Stores are written under `tokenize/out/` and are gitignored.
