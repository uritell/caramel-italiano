# score/ — two-track timing statistics

Energy voice activity on stereo audio, one speaker per channel. The
headline numbers are the timing rows caramel publishes: turns per minute,
percent both talking, percent nobody talking, median gap between turns,
and overlapping turn changes per minute. Backchannels per minute are
counted as well, because synthesis needs a rate.

Upstream did not ship this scorer. The definitions are in `activity.py`.
Short version:

- A frame is speech when its RMS sits above a threshold between the
  channel's quiet frames and its loud frames.
- A turn change is two successive floor turns by different speakers.
- The gap is next start minus previous end. Negative means the next
  person starts first.
- An overlapping turn change is a turn change with a negative gap.
- A backchannel is a short region that sits inside the other person's
  speech. "Short" is the valley in the DataHive duration histogram, not
  a fixed one-second rule.
- Percents are fractions of the timeline. Both-talking, exactly-one, and
  nobody-talking add up to 100%.

`measure.py` runs that detector on every DataHive conversation, chooses
the VAD smoothing by a grid against DataHive's own card labels, and
writes `synth/timing_config.json`.

`score_files.py` scores any stereo wavs with that same detector, so the
synthetic set is not scored with a different definition.

`compare.py` writes `comparison.md`.

```bash
python score/activity.py --self-test
python score/measure.py
python score/score_files.py --audio synth/out/conversations/*.wav --out score/out/synthetic.json
python score/compare.py --synthetic score/out/synthetic.json
```

The target bands for a full closeness score are not in this folder.
Do not invent them. The comparison here is the five timing numbers (plus
backchannels) on real DataHive versus the synthetic set.
