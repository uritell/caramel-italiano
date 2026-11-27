# Real DataHive timing vs synthetic two-track dialogues

Both columns use the same energy voice-activity detector. Its threshold, bridge and intra-turn pause were chosen so the detector lands near the turns/minute, overlap and median gap printed on the DataHive dataset card. The synthesiser draws gaps, backchannel durations, backchannel positions, and now turn lengths from the lists that detector measured on all six DataHive conversations. Read-speech clips supply the sound. They are not real backchannels.

Real audio: 6 conversations, 3.051 h. Each synthetic column is about 1.00 h, seed 0, two MLS speaker pairs.

## Before and after

"Before" is the first smoke: each MLS clip, about 16 s, was one turn, including the silence at both ends of the file. "After" splits those clips at energy pauses, trims that edge silence, and packs the pieces toward a duration drawn from the 1,566 measured DataHive floor turns (median 3.6 s, mean 6.2 s).

| Statistic | DataHive | Before | After |
|---|---:|---:|---:|
| Turns per minute | 8.55 | 4.74 | 9.39 |
| Both talking (%) | 15.4 | 10.2 | 11.2 |
| Nobody talking (%) | 15.2 | 10.2 | 16.1 |
| Median gap between turns (ms) | 350 | 675 | 360 |
| Overlapping turn changes per minute | 1.98 | 0.88 | 2.20 |
| Backchannels per minute | 15.64 | 8.31 | 9.92 |

## Why the gap was twice the sampled value

The sampled DataHive gaps have median 350 ms, and the gaps actually written between turns in the first smoke had median about 210–370 ms. The scorer nevertheless reported 675 ms. On 80 MLS train clips the same VAD finds a median 340 ms of silence at the start of the file and 370 ms at the end. That silence sits inside the scored gap, on top of the gap that was sampled. MLS `segments.txt` only gives those clip boundaries (a start and end in the original chapter). It does not mark pauses inside the clip, so the split is the energy VAD, not an alignment.

After trimming to the speech mask, the scored median gap is 360 ms. The gaps written for that rebuild had medians of 400 ms and 310 ms, so the scorer is now reading the gap that was sampled.

## What is still off

Turns per minute moved from 4.74 to 9.39, against 8.55. Placed turns have median 4.3–4.9 s. The DataHive median is 3.6 s and the mean is 6.2 s, because the tail runs past 13 s. Pieces with no pause inside them cannot be cut, so the match is not exact.

Overlapping turn changes moved from 0.88 to 2.20 per minute, against 1.98. Both-talking moved only from 10.2% to 11.2%, against 15.4%. Nobody-talking, 16.1%, is now next to 15.2%.

Backchannels moved from 8.31 to 9.92 per minute, against 15.64. The median measured backchannel is 390 ms; the histogram valley that sets the maximum is 2.75 s. Snippets are kept at least 1 s (the scorer's intra-turn pause) away from the same speaker's own speech, or the scorer merges them into that turn and they disappear from the count. About a third of the drawn snippets still do not survive as backchannels. The sound is still a slice of read speech.

## How the detector was set

Cutoff rule: valley in the histogram of regions with overlap ratio at least 0.5, bin 0.10 s, cutoff 2.75 s (2,929 regions).

Chosen VAD: threshold fraction 0.5, minimum margin 8 dB, bridge 250 ms, intra-turn pause 1000 ms, backchannel cutoff 2.75 s.

Grid search: 36 settings, all inside a loose sanity window, mean card-fit error 0.498. Per file, the card's own labels against this detector:

| File | Card turns/min | Measured turns/min | Card overlap % | Measured both % | Card gap ms | Measured gap ms |
|---|---:|---:|---:|---:|---:|---:|
| it_000001 | 4.1 | 4.1 | 12.8 | 22.3 | -350 | -420 |
| it_000002 | 11.1 | 11.6 | 15.3 | 15.0 | 309 | 550 |
| it_000003 | 7.6 | 12.6 | 12.6 | 14.1 | 450 | 400 |
| it_000004 | 8.4 | 12.5 | 12.4 | 13.1 | 401 | 480 |
| it_000005 | 7.9 | 9.0 | 12.5 | 12.4 | 141 | 385 |
| it_000006 | 8.3 | 7.7 | 9.6 | 9.1 | 383 | 430 |

A negative gap means the next speaker usually starts before the previous one has finished. Overlapping turn changes count only those speaker changes, not backchannels.

## Mimi encode speed on this CPU

This section is the first smoke. The timing rebuild above did not re-encode tokens.

No GPU was used. This machine has no CUDA device, and this step does not rent one.

Short clips (FLEURS mono smoke, 57 clips, 10.1 min of audio) encoded in one shot at real-time factor **0.0281** (36× realtime): 16.9 s of encode time. A warmup-excluded repeat on one clip was RTF 0.0259.

Long stereo (DataHive dev file plus the synthetic hour and an 8-minute DataHive slice) is encoded in 60 s chunks with 60 s of left context, because a single 30-minute stereo encode asked this CPU build for a 20 GB convolution buffer and was killed. That path measured real-time factor **0.0767** (13× realtime) over 2.52 channel-hours. The extra time versus the short-clip rate is the repeated context. Chunked tokens are not bit-identical to a one-shot encode: on a 150 s test, about 1.2% of tokens differed. On a shorter check with the same 60 s context, the mismatches sat mostly in the higher codebooks.

Upstream `TokenStore` and `delay_window` (imported from `third_party/caramel/train.py`, not copied) loaded all three stores: mono width 8, 7559 frames; dev width 16, 5511 frames; stereo width 16, 51107 frames, 3 clips.

| What you would tokenize | Rate used | CPU wall time |
|---|---|---:|
| 300 h of short mono clips | RTF 0.028 | 8.4 h |
| 600 h of short mono clips | RTF 0.028 | 16.8 h |
| 300 h of long stereo files | RTF 0.077, two channels | 46.0 h |
| 600 h of long stereo files | RTF 0.077, two channels | 92.0 h |
| The 274.9 h already downloaded (mostly short MLS and FLEURS clips, DataHive counted twice for its two channels) | short-clip rate, DataHive at the long-file rate | 8.1 h |

There is no measured GPU figure. A published Candle Mimi encode on an M4 with Metal is about 9× realtime, which is a different implementation and slower than this CPU's short-clip rate, so it is not a quote for one datacentre GPU. Training step counts sized from the published 145.6k frames/s are in `train/README.md`.
