"""Warm start from the public English caramel-1 weights, then the Option A tokens.

Stage 1 is skipped. Stages 2 and 3 use a lower learning rate and a short
warmup. The first clips of the stereo store are the DataHive training
conversations (prepare_audio writes them before the synthetic dialogues);
those clips are sampled 5 times more often than their length alone would.

The English checkpoint has no optimizer state. train.py --init-weights
loads the safetensors and leaves Adam freshly initialised. Stage 3 then
resumes from the stage-2 ckpt.pt, which does have the optimizer.
"""
from __future__ import annotations

from option_a import (
    DISK_CENTS_PER_GB_MONTH,
    HOURLY_CENTS,
    MONTH_S,
    STEP_S,
    disk_cents,
    training_plan as option_a_plan,
)

NAME = "warmstart"
# $3.00 hard stop. 38 min at $4.39/h is about $2.78, so the clock hits
# first and stays under the dollar cap.
MAX_USD_CENTS = 300
MAX_WALL_S = 38 * 60  # 2280 s
ABORT_CENTS = 250
# Tokens only, plus the 826 MB public weights and one optimizer checkpoint.
DISK_GB = 40
# Allowance for cloning and downloading naklecha/caramel-1. Not a measurement.
WEIGHT_DOWNLOAD_S = 3 * 60
INIT_WEIGHTS = "/workspace/caramel-1/model.safetensors"
# 6 DataHive files: shortest is dev, the next is the score clip, four train.
REAL_CLIPS = 4
REAL_BOOST = 5.0
LR = "1e-4"
STORES_DIR = "runpod/retrieved/mviebvzw3ht7hn/stores"


def training_plan(gpus: int = 1) -> dict:
    base = option_a_plan(gpus)
    s2, s3 = base["s2_steps"], base["s3_steps"]
    return {
        "gpus": gpus,
        "name": NAME,
        "mono_hours": base["mono_hours"],
        "synth_hours": base["synth_hours"],
        "stereo_hours": base["stereo_hours"],
        "s1_stop": 0,
        "s2_end": s2,
        "s3_end": s2 + s3,
        "s2_steps": s2,
        "s3_steps": s3,
        "s1_warmup": 0,
        "s2_warmup": 8,
        "s3_warmup": 8,
        "eval_every": base["eval_every"],
        "skip_stage1": True,
        "init_weights": INIT_WEIGHTS,
        "lr": LR,
        "boost_first": REAL_CLIPS,
        "boost": REAL_BOOST,
        "save_during_stage": True,
    }


def estimate(hourly_cents: int = HOURLY_CENTS) -> dict:
    plan = training_plan()
    steps = plan["s2_steps"] + plan["s3_steps"]
    train_s = steps * STEP_S
    phases = [
        ("download caramel-1 weights", float(WEIGHT_DOWNLOAD_S)),
        ("train stages 2 and 3", train_s),
    ]
    rows = []
    for name, sec in phases:
        rows.append({"name": name, "seconds": sec, "cents": int(round(sec / 3600.0 * hourly_cents))})
    total_s = sum(sec for _, sec in phases)
    gpu_cents = int(round(total_s / 3600.0 * hourly_cents))
    return {
        "phases": rows,
        "seconds": total_s,
        "gpu_cents": gpu_cents,
        "disk_cents": disk_cents(DISK_GB, MAX_WALL_S),
        "steps": steps,
        "plan": plan,
        "over_abort": gpu_cents > ABORT_CENTS,
    }


def explain_estimate(est: dict | None = None) -> str:
    est = est or estimate()
    lines = []
    for row in est["phases"]:
        lines.append(f"  {row['name']}: {row['seconds'] / 60.0:.1f} min, ${row['cents'] / 100:.2f}")
    lines.append(
        f"  GPU total: {est['seconds'] / 60.0:.1f} min, ${est['gpu_cents'] / 100:.2f} at $4.39/h"
    )
    raw_cents = DISK_GB * DISK_CENTS_PER_GB_MONTH * MAX_WALL_S / MONTH_S
    disk_txt = f"${raw_cents / 100:.3f}" if raw_cents < 1 else f"${est['disk_cents'] / 100:.2f}"
    lines.append(
        f"  container disk: {DISK_GB} GB at ${DISK_CENTS_PER_GB_MONTH / 100:.2f}/GB/month "
        f"while the pod runs, about {disk_txt} for {MAX_WALL_S / 60:.0f} min. "
        f"volumeInGb is 0."
    )
    lines.append("  no audio download and no tokenization; the Option A stores are uploaded.")
    if est["over_abort"]:
        lines.append("  REFUSING: the GPU estimate is above $2.50.")
    return "\n".join(lines)
