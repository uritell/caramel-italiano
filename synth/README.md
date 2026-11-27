# synth/ — synthetic two-track Italian dialogues

caramel-1 needs each speaker on their own track so the model can learn
overlap and backchannels. Open Italian stereo conversation is only the
~3 hour DataHive sample. This folder is where synthetic left/right
dialogues will be built from open single-speaker Italian (FLEURS IT,
MLS IT, Common Voice IT).

## Constraints

- Use only the clean-pilot licences in `DATA_LICENSES.md`.
- Common Voice: no speaker re-identification, no re-hosting of the
  corpus. Treat speaker labels as opaque ids from the official files.
- Do not clone ITALIC voices (out of scope anyway).
- Synthetic turn-taking will miss authentic overlap, barging-in,
  backchannels, and informal Italian. Keep the DataHive sample as the
  only *real* stereo conversations, and hold some of it out for scoring.
- Output layout (planned, not implemented): stereo WAV, 24 kHz, left =
  speaker A, right = speaker B, plus a sidecar of turn timestamps.
  Write to `synth/out/` (gitignored).

## Implementation status

Not started. A concatenative or pause-scheduled pairing of existing
single-speaker clips is the first thing to try; a commercially licensed
TTS is only in scope if its weights and output are compatible with the
$50 clean-pilot rules (VoxPopuli *checkpoints* are CC BY-NC and must
not be used).
