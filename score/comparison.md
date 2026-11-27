# Real DataHive timing vs synthetic two-track dialogues

Both columns use the same energy voice-activity detector. Its threshold, bridge and intra-turn pause were chosen so the detector lands near the turns/minute, overlap and median gap printed on the DataHive dataset card; that fit is a calibration check, not the target the synthesiser copies. The synthesiser draws each gap, each backchannel duration and each backchannel position from the empirical lists measured on all six DataHive conversations. Read-speech clips supply the sound. They are not real backchannels, and the comparison is expected to show that.

Real audio: 6 conversations, 3.051 h. Synthetic audio: 2 conversations, 1.002 h.

| Statistic | DataHive (measured here) | Synthetic |
|---|---:|---:|
| Turns per minute | 8.55 | 4.74 |
| Both talking (%) | 15.4 | 10.2 |
| Nobody talking (%) | 15.2 | 10.2 |
| Median gap between turns (ms) | 350 | 675 |
| Overlapping turn changes per minute | 1.98 | 0.88 |
| Backchannels per minute | 15.64 | 8.31 |

## How the detector was set

Cutoff rule: Valley in the histogram of regions with overlap ratio >= the configured threshold, bin 0.10s, cutoff 2.750s (2929 regions)

Chosen VAD: threshold fraction 0.5, minimum margin 8.0 dB, bridge 250.0 ms, intra-turn pause 1000.0 ms, backchannel cutoff 2.75 s, overlap ratio for a backchannel 0.5.

The cutoff is the long edge of that histogram. The durations actually sampled are shorter: median 390 ms across 2863 measured backchannels.

Grid search: 36 settings, 36 inside a loose sanity window, mean card-fit error 0.498. Per file, the card's own labels against this detector:

| File | Card turns/min | Measured turns/min | Card overlap % | Measured both % | Card gap ms | Measured gap ms |
|---|---:|---:|---:|---:|---:|---:|
| it_000001 | 4.1 | 4.1 | 12.8 | 22.3 | -350 | -420 |
| it_000002 | 11.1 | 11.6 | 15.3 | 15.0 | 309 | 550 |
| it_000003 | 7.6 | 12.6 | 12.6 | 14.1 | 450 | 400 |
| it_000004 | 8.4 | 12.5 | 12.4 | 13.1 | 401 | 480 |
| it_000005 | 7.9 | 9.0 | 12.5 | 12.4 | 141 | 385 |
| it_000006 | 8.3 | 7.7 | 9.6 | 9.1 | 383 | 430 |

## Reading the gap

A negative median gap means the next speaker usually starts before the previous one has finished. Overlapping turn changes per minute count only those speaker changes, not backchannels. Backchannels are short regions that sit inside the other person's turn; their audio in the synthetic set is a slice of read speech, cut at a low-energy point, placed with the measured rate and the measured position along the other turn.

The synthetic column is not expected to copy the DataHive column. MLS utterances are mostly 10–20 seconds and are kept whole, so a conversation built from them has fewer turns per minute than DataHive even when every gap is drawn from the measured list. The median gap in the table is what the detector finds between those longer turns, not a second hand-set pause. Some drawn backchannels are skipped when they would land on speech already placed for that speaker, and some slices are not labelled as backchannels by the detector, so the synthetic backchannel rate comes out lower than the measured rate.

## Mimi encode speed on this CPU

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

There is no measured GPU figure. A published Candle Mimi encode on an M4 with Metal is about 9× realtime, which is a different implementation and slower than this CPU's short-clip rate, so it is not a quote for one datacentre GPU.
