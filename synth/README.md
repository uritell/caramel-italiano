# synth/ — synthetic two-track Italian dialogues

Open Italian stereo conversation is the DataHive sample (about three
hours, six calls). This folder builds extra two-track audio from
Multilingual LibriSpeech Italian, which publishes a speaker id per clip.

FLEURS is not mixed in here. Its release has gender and no speaker id, so
a track made of FLEURS clips would change voice every turn. FLEURS stays
a mono source for `tokenize/`.

## What a synthetic conversation is

- Left and right are two different MLS train speakers. Dev and test are
  left unused.
- MLS `segments.txt` marks the 10–20 s clip only, not pauses inside it.
  Each clip is split with the same energy VAD that scores DataHive.
  Pauses shorter than `vad.bridge_ms` are filled first, so a cut is a
  pause that detector would not glue into one region.
- Pieces are trimmed to the speech mask. On 80 MLS train clips the
  median silence was about 340 ms at the start and 370 ms at the end.
  Leaving that on the clip made the scored gap about twice the sampled
  gap (675 ms against a 350 ms median).
- Pieces are packed toward a duration drawn from
  `empirical.floor_turn_duration_s` (1,566 DataHive floor turns, median
  3.6 s, mean 6.2 s). A further piece is added while the turn is still
  short by at least that piece; a smaller shortfall is filled with that
  probability. A stretch with no pause in it is kept whole.
- Turns alternate. The silence or overlap before the next turn is sampled
  from the empirical DataHive gap list. A negative gap starts the next
  speaker before the previous one has finished.
- Backchannels, when the measurement found any, are short slices of the
  listener's unused audio, placed in a silent part of their channel at
  least `intra_turn_pause_ms` away from their own speech so the scorer
  does not merge them into the neighbouring turn. Rate, duration, and
  position are the measured lists. The sound is still read speech.
- Each track is scaled to the median active-frame RMS measured on DataHive.
- If a sampled gap would stack two clips of the same speaker, the later
  clip is pushed forward. The sidecar counts those clamps.

Output (gitignored): `synth/out/conversations/conv_NNN.wav` at 24 kHz,
left = speaker A, right = speaker B, plus a JSON sidecar with the clip
ids, the licence, and the turn times. `synth/out/manifest.jsonl` is the
list of conversations.

## Timing parameters

They live in `synth/timing_config.json`, written by `score/measure.py`.
Re-running the measurement overwrites it. `synthesize.py` reads that file
and does not invent a second set of gaps.

| Field | Role |
|---|---|
| `vad.frame_ms`, `vad.hop_ms` | Energy frame and hop. |
| `vad.noise_percentile`, `vad.speech_percentile`, `vad.threshold_fraction`, `vad.min_margin_db` | Per-channel threshold between the quiet frames and the loud frames. |
| `vad.bridge_ms` | Gaps shorter than this, inside one channel, are filled before the percentages are computed. |
| `vad.min_region_ms` | Shorter blips are dropped. |
| `vad.intra_turn_pause_ms` | Same-speaker floor regions separated by a shorter pause, with no floor turn of the other speaker in between, count as one turn. |
| `vad.backchannel_max_s` | Regions at or under this length, and overlapping the other speaker by at least `backchannel_overlap_ratio`, are backchannels rather than floor turns. The value is the first valley in the histogram of overlapped-region durations on DataHive. |
| `empirical.gap_ms` | Every measured speaker-change gap, in milliseconds. Negative means overlap. The synthesiser samples this list. |
| `empirical.backchannel_duration_ms` | Measured backchannel lengths. |
| `empirical.backchannel_rel_position` | Where each backchannel started inside the other person's turn, 0 at the start and 1 at the end. |
| `empirical.backchannels_per_min` | Count divided by hours of DataHive audio, times 60. The synthesiser draws a Poisson count at this rate. |
| `empirical.target_active_rms` | Median RMS of active frames, the gain target for each clip. |
| `empirical.floor_turn_duration_s` | Every measured DataHive floor-turn length, in seconds. The synthesiser draws a target duration from this list for each turn. |
| `empirical.conversation_duration_s` | The six DataHive durations. With no `--conversation-seconds`, each synthetic conversation aims at the median of this list, and the last one is shortened so the total matches `--hours`. |

The VAD knobs above the empirical lists are the detector, not the
conversation targets. They were picked by a grid search so the detector's
turns/minute, percent-both-talking and median gap sit near the figures
DataHive already prints on its dataset card. The card figures are a check
on the detector. The lists the synthesiser samples are the detector's own
output. `backchannel_cutoff_rule` in the JSON says how the cutoff was read
off the histogram. If that histogram has no valley, the backchannel lists
are empty and no backchannels are added.

```bash
python score/measure.py          # writes synth/timing_config.json
python synth/synthesize.py --hours 1.0 --seed 0
python synth/synthesize.py --hours 1.0 --conversation-seconds 600 --seed 0
```

`--hours` and `--conversation-seconds` change the length of the files.
They do not replace the measured gap or backchannel distributions.
