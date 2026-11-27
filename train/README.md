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

## Pilot-sized step counts (one RunPod attempt)

The full-recipe `--steps 25000 / 47000` and `--stop_at 9000` belong to
the ~11,000-hour English mix. This pilot uses a much shorter run, sized
for one 2× H100 SXM pod under a **USD 25** approval (the project cap
stays USD 50). The launch scripts live in `runpod/`. They do not rent
anything until `python runpod/launch.py --launch`.

Stages 1 and 2 share one cosine schedule that ends at step 1072
(η_max = 3e-4, warmup 50). Stage 1 stops early, on one GPU. Stage 3
restarts the schedule at step 1072 with the published stage-3 peak
(η_max = 1.5e-4, warmup 200).

| Stage | GPUs | What is passed to `train.py` | Stop |
|---|---|---|---|
| 1. Learn to speak | 1 | `--steps 1072 --stop_at 401` on mono | step 401 |
| 2. Learn to converse | 2 | `--steps 1072`, resume stage 1, weights 0.8 / 0.2 | step 1072 |
| 3. Keep going | 2 | `--steps 1981 --stop_at 1981 --sched_start 1072 --warmup 200 --lr 1.5e-4 --save_best` | step 1981 |

Stage 3 uses `--eval_every 200`. The published stride of 1000 never
lands between 1072 and 1981, so `--save_best` would not run. Token
hours and GPU frames/s are still measured on the pod and written to
`speed.json`; those measurements are what a later, longer run should
be sized from. Prices checked 27 Nov 2026, USD before tax: about
$4.39/GPU-h for a RunPod Secure Cloud H100.

Other `train.py` flags that matter, with their defaults: `--fpb 24000`
(frames per batch per GPU), `--lr 3e-4`, `--warmup 50`, `--wd 0.1`,
`--clip 1.0`, `--workers 6`. Single-GPU training is valid (`python
train.py ...` without `torchrun`); the published stage-1 pass used one
GPU.

Export a Hugging Face-style folder with upstream `model.py` only after
a checkpoint is worth keeping.
