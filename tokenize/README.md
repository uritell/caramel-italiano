# tokenize/ — Mimi encode to the caramel `train.py` store format

`caramel-1` never trains on raw audio. Each speaker is encoded with
Kyutai Mimi and written to a folder that
`third_party/caramel/train.py` (`TokenStore`) memory-maps.

Use the same settings as `third_party/caramel/generate.py`:

- weights: `kyutai/mimi` (CC BY 4.0; see NOTICE)
- audio: 24 kHz
- `num_quantizers=8`
- frame rate: 12.5 frames per second (80 ms per frame)

One hour of one-person speech is `12.5 * 8 * 3600 = 360,000` tokens;
a two-person conversation is twice that.

## Store format (from `train.py`, do not invent fields)

A store is **one folder** containing exactly these three files:

| File | Type and shape | Meaning |
|---|---|---|
| `tokens.bin` | `uint16`, frame-major, shape `[frames, W]` | Codec tokens. **W = 16** for stereo: speaker 1 codebooks 0–7, then speaker 2 codebooks 0–7. **W = 8** for mono (one-person speech). |
| `index.npy` | `int64`, shape `[clips, 2]` | Each row is `(offset, length)` in **frames**, indexing into the first axis of `tokens.bin`. |
| `meta.json` | JSON object | `{"width": W, "frames": <total frames in tokens.bin>}`. `width` must equal `W`. |

`TokenStore` does:

```python
meta = json.load(open(os.path.join(d, "meta.json")))
self.W, self.index = meta["width"], np.load(os.path.join(d, "index.npy"))
self.tok = np.memmap(os.path.join(d, "tokens.bin"), dtype=np.uint16, mode="r",
                     shape=(meta["frames"], self.W))
```

`train.py` then windows clips, and `delay_window()` turns a raw
`[L, W]` slice into the delayed 16-wide training sequence (a mono clip
is padded with `-1` for the second speaker; stereo clips are trained in
both speaker orders). The tokenizer does **not** apply that delay
pattern; it only writes the raw store.

Planned stores (names only, not yet built):

- `tokenize/out/mono/` — width 8, from single-speaker Italian
- `tokenize/out/stereo/` — width 16, DataHive real conversations plus
  synthetic two-track dialogues
- `tokenize/out/dev/` — held-out width-16 conversations for `--dev`

Implementation status: not started. Measure tokenization throughput on
the first rented (or local) hour before budgeting the rest of the $50.
