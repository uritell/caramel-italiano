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

## Pilot-sized step counts: TODO (do not invent)

`--steps 25000 / 47000` and `--stop_at 9000` belong to the full
~11,000-hour English mix. They are almost certainly too large for a
few-hundred-hour Italian pilot and would blow the $50 cap if copied
blindly.

Before any rental, measure:

- token hours actually written to `tokenize/out/mono` and `.../stereo`
- frames per second on the chosen GPU (upstream quotes up to ~145.6k
  frames/s on one H100 with their kernels)
- tokenization time, which the paper does not publish

Then set `--steps`, `--stop_at`, `--warmup`, and `--sched_start` so
that the three stages stay inside the remaining budget (on-demand
prices checked 27 Nov 2026, USD before tax: roughly $1.73/h Vast.ai
verified H100 offer, $4.39/h RunPod, $4.89/h Lambda). Keep a held-out
Italian stereo store on `--dev`. Write the chosen numbers here when
they exist; do not fill them in by guesswork.

Other `train.py` flags that matter, with their defaults: `--fpb 24000`
(frames per batch per GPU), `--lr 3e-4`, `--warmup 50`, `--wd 0.1`,
`--clip 1.0`, `--workers 6`. Single-GPU training is valid (`python
train.py ...` without `torchrun`); the published stage-1 pass used one
GPU.

Export a Hugging Face-style folder with upstream `model.py` only after
a checkpoint is worth keeping.
