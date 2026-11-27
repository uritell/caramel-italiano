# train/ — running upstream `train.py` for the Italian pilot

Training uses **upstream** `third_party/caramel/train.py` at commit
`84a9271d265841543287d4044d7ef80991a2b003`. Do not vendor or fork that
file unless a documented Italian-specific patch is required later.

**Do not rent a GPU until the owner has accepted a quote that fits the
USD 50 cap.**

## Three-stage recipe (as published)

From the [caramel README](https://github.com/naklecha/caramel/blob/84a9271d265841543287d4044d7ef80991a2b003/README.md):

| Stage | Data (full English recipe) | GPUs | Time (published) |
|---|---|---|---|
| 1. Learn to speak | one-person speech, 0.83 passes | 1 | 27 min |
| 2. Learn to converse | 5,743 h of conversations, 2.4 passes, 20% one-person speech mixed in | 2× H100 | 46 min |
| 3. Keep going | same data, up to 5.65 passes | 2× H100 | 61 min |

Stages 1 and 2 share one cosine LR curve with η_max = 3e-4 (after a
short warmup, down to 0.1× peak). Stage 3 warms back up to
η_max = 1.5e-4 (`--sched_start 25000 --warmup 200 --lr 1.5e-4`). The
full published run was about **4 H100 GPU-hours**.

Published commands (full recipe — **not** the Italian pilot sizes):

```bash
torchrun --nproc_per_node 2 train.py --data data/mono --steps 25000 --stop_at 9000 --out runs/s1
torchrun --nproc_per_node 2 train.py --data data/stereo data/mono --weights 0.8 0.2 --steps 25000 \
    --resume runs/s1/ckpt.pt --out runs/s2
torchrun --nproc_per_node 2 train.py --data data/stereo data/mono --weights 0.8 0.2 --steps 47000 \
    --resume runs/s2/ckpt.pt --sched_start 25000 --warmup 200 --lr 1.5e-4 \
    --dev data/dev --eval_every 1000 --patience 2 --save_best --out runs/s3
python model.py runs/s3/best.pt ./caramel-1
```

Run them from `third_party/caramel/`, and point `--data` / `--dev` /
`--out` at this project's token stores and a gitignored `runs/`
directory.

## Pilot-sized step counts

These keep the same passes as the published recipe. They are not the
English step counts scaled by a guess. One training frame is one Mimi
frame, 12.5 per second of timeline, so an hour is

```text
12.5 × 3600 = 45,000 frames
```

Upstream's batch is `--fpb 24000` frames per GPU. The published
throughput used below is 145,600 frames/s on one H100. GPU-hours are
`frames / 145600 / 3600`. Wall-clock on two GPUs assumes that rate on
each GPU. The published English run lands within about 10% of this
arithmetic, so it is a peak-rate estimate, not a measurement on our data.

### Hours the steps are sized to

Measured file hours, with the usual holdouts taken out:

| Pool | Hours | What is in it |
|---|---:|---|
| Mono train | 257.923 | MLS train 247.378 + FLEURS train 8.998 + FLEURS dev 1.547 |
| Held out of mono | 14.0 | MLS dev 5.176, MLS test 5.273, FLEURS test 3.521 |
| Real stereo train | 2.929 | DataHive 3.051 minus `it_000002` (440.8 s), the smoke `--dev` file |
| Synthetic stereo | 235.71 | see below |

Synthetic stereo is not the 1 hour timing smoke. It is the timeline you
get by running every mono training clip through the synthesizer once.
On 80 MLS train clips, the median fraction that is speech after the same
VAD is 0.8686. On the rebuilt 1 hour synthetic set, channel activity
(both-talking + exactly-one) is 0.9504. So

```text
235.71 h = 257.923 × 0.8686 / 0.9504
stereo train = 235.71 + 2.929 = 238.64 h
```

The same recordings are heard as monologue in stage 1 and again inside
the synthetic dialogues. That is the pilot. It is not a second 258 hours
of new speakers.

### Passes, then steps

Stage 1 is 0.83 passes of the mono pool, one GPU, as published:

```text
frames = 0.83 × 257.923 × 45,000 = 9,633,424
steps  = 9,633,424 / 24,000 = 401.4  →  401 steps
check  = 401 × 24,000 / 45,000 / 257.923 = 0.829 passes
```

Stage 2 is 2.4 passes of the stereo pool, and 80% of frames are stereo
(`--weights 0.8 0.2`), on 2 GPUs:

```text
frames = 2.4 × 238.64 × 45,000 / 0.8 = 32,216,400
steps  = 32,216,400 / (24,000 × 2) = 671.2  →  671 steps
check  = 0.8 × 671 × 48,000 / 45,000 / 238.64 = 2.40 passes
```

Stage 3 adds the rest of the published 5.65 stereo passes, so
5.65 − 2.40 = 3.25 more passes, same weights, 2 GPUs:

```text
frames = 3.25 × 238.64 × 45,000 / 0.8 = 43,626,375
steps  = 43,626,375 / 48,000 = 908.9  →  909 steps
check  = 0.8 × 909 × 48,000 / 45,000 / 238.64 = 3.25 passes
cumulative stereo passes = 2.40 + 3.25 = 5.65
```

Mono is also drawn for the 20% of stage 2 and stage 3. That is another
0.55 then 0.75 passes, so mono is seen about 0.83 + 0.55 + 0.75 = 2.14
times. Upstream's mono pool is heard about 2.2 times for the same reason.

The shared stage-1/2 schedule is `401 + 671 = 1072` steps. Stage 3 runs
to `1072 + 909 = 1981`.

Published warmups are 50 steps and, in stage 3, 200 steps. Those are
0.6% of the English stage-1 length and 0.9% of the English stage-3
length. The same fractions of these runs are 2 steps and 8 steps.
Using 50 and 200 here would spend most of stage 1 warming up.

```bash
# from third_party/caramel/, paths pointed at this repo's stores
python train.py --data <mono> --steps 1072 --stop_at 401 --warmup 2 --out runs/s1
torchrun --nproc_per_node 2 train.py --data <stereo> <mono> --weights 0.8 0.2 \
    --steps 1072 --resume runs/s1/ckpt.pt --out runs/s2
torchrun --nproc_per_node 2 train.py --data <stereo> <mono> --weights 0.8 0.2 \
    --steps 1981 --resume runs/s2/ckpt.pt --sched_start 1072 --warmup 8 --lr 1.5e-4 \
    --dev <held-out DataHive store> --eval_every 100 --patience 2 --save_best --out runs/s3
```

`<stereo>` has to be the full synthetic build plus the real calls, not
the 1 hour timing smoke. `<dev>` is a real stereo store held out of that
mix. `it_000002` is only 7.3 minutes; a longer held-out call would be a
sturdier dev set and would shrink the 2.929 h figure a little.

### H100 time and money

```text
frames = 401×24,000 + 671×48,000 + 909×48,000
       = 9,624,000 + 32,208,000 + 43,632,000
       = 85,464,000
GPU-hours = 85,464,000 / 145,600 / 3,600 = 0.163
```

At the peak rate that is 1.1 min on one H100 for stage 1, then 1.8 min
and 2.5 min of wall-clock on two H100s. Prices already written down on
27 Nov 2026, USD before tax, times 0.163 h: about $0.28 (Vast.ai
$1.73/h), $0.72 (RunPod $4.39/h), $0.80 (Lambda $4.89/h). Even at twice
the peak estimate this is a few dollars, inside the $50 cap. It is still
an estimate from the published frame rate, not a timed run, and nothing
here rents a GPU.

Tokenizing the 275 h already downloaded is a separate CPU cost, about
8 hours on the smoke machine (`score/comparison.md`). The pilot stores
from that smoke are a 10 minute mono slice and about an hour of stereo,
not the pools above.

## RunPod launch

`runpod/` runs these same three commands on one Secure Cloud 2× H100 SXM
pod. `<mono>`, `<stereo>`, and `<dev>` are the stores `pod_run.sh` builds:
MLS and FLEURS mono, the synthetic dialogues plus the real DataHive calls
kept for training, and a held-out DataHive stereo store. The pod attempt
has its own **USD 25** ceiling. The project cap stays USD 50.
`python runpod/launch.py --dry-run` checks the plan and creates nothing.
`python runpod/launch.py --launch` is the command that rents the pod. It
terminates at 2.5 hours or about $22, whichever comes first. See
`runpod/README.md`.

The 0.163 GPU-hour figure above is training at the published frame rate.
Download and tokenization are extra, which is why the pod's wall-clock cap
is 2.5 hours. The pod writes measured frames/s to `speed.json`.

Other `train.py` flags that matter, with their defaults: `--fpb 24000`
(frames per batch per GPU), `--lr 3e-4`, `--warmup 50`, `--wd 0.1`,
`--clip 1.0`, `--workers 6`. Single-GPU training is valid (`python
train.py ...` without `torchrun`); the published stage-1 pass used one
GPU.

Export a Hugging Face-style folder with upstream `model.py` only after
a checkpoint is worth keeping.
