# eval/ — word clarity, timing, and listening

Evaluation for the Italian pilot, once a checkpoint exists:

1. **Held-out token loss** — `train.py --dev` on a held-out Italian
   stereo store (see `tokenize/README.md`).
2. **ASR word clarity** — caramel reports the mean log-probability of a
   speech recogniser's transcript (closer to zero is clearer), and this
   project also reports WER/CER on Italian. The codec test already used
   faster-whisper `small` and `large-v3-turbo` on FLEURS read speech;
   conversation audio is still untested.
3. **Timing / closeness** — run the scorer in `score/` on model
   self-play and on held-out real DataHive conversations.
4. **Listening** — stereo (left = A, right = B) samples from
   `third_party/caramel/generate.py`.

`eval/codec_test/` is the completed, free CPU test of whether Mimi at
8 codebooks preserves Italian. It is a codec-preservation test, not a
trained-model evaluation. Start there; the rest is not implemented.
