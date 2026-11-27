# score/ — Italian timing / closeness statistics

caramel-1 picked data and later judged the model with a **closeness
score**: 13 statistics, each turned into a target band, mostly about
conversation timing (overlap, gaps between turns, backchannels per
minute, silence) and a few about the recording (cross-talk, noise
floor). A score of 0 means every statistic sits inside its band.

The public caramel repo **does not include** this scorer, the
speaker-splitting tools, or the published English/Dessa band values as
code. Those have to be rebuilt from the statistics and the tables in
the [caramel README](https://github.com/naklecha/caramel/blob/84a9271d265841543287d4044d7ef80991a2b003/README.md).

## What this folder should do

- Re-implement the published statistics on stereo (one speaker per
  channel) Italian audio.
- **Fit the target bands on real Italian conversations**, not on the
  English Dessa bands. The only clean-licence, separate-track Italian
  conversations in the pilot are the DataHive sample (~3 hours, 6
  conversations). Hold some of them out so the bands are not fitted on
  the same recordings used to praise the model.
- Use the score both as a sanity check on synthetic dialogues (expect
  them to look unlike real Italian talk) and as an evaluation of
  generated conversations.

Implementation status: not started. Do not invent band values here.
