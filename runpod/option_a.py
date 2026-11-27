"""Option A: one Secure H100, the open Italian pools, hard cap $9.

The retry's best.pt is weights only. train.py --resume also wants the
optimizer and the RNG, so this run starts from the same random
initialisation as the retry. It does not load that checkpoint.

Hours below are the published sizes, not a new measurement:
MLS Italian 257.83 h (train+dev+test), FLEURS Italian about 14 h,
Emozionalmente 6,902 clips times the stated mean 3.81 s (about 7.30 h).
Synthetic stereo is 218 h. DataHive training split stays 2.929 h.
"""
from __future__ import annotations

from pilot_config import (
    DATAHIVE_TRAIN_HOURS,
    FPB,
    FRAMES_PER_HOUR,
    MONO_PASSES,
    S1_WARMUP,
    S3_EVAL_EVERY,
    S3_WARMUP,
    STEREO_PASSES_S2,
    STEREO_PASSES_S3,
    STEREO_WEIGHT,
    _steps,
)

NAME = "optionA"
MLS_HOURS = 257.83
FLEURS_HOURS = 14.0
EMO_CLIPS = 6902
EMO_MEAN_S = 3.81
EMO_HOURS = EMO_CLIPS * EMO_MEAN_S / 3600.0  # 7.305 h
MONO_HOURS = MLS_HOURS + FLEURS_HOURS + EMO_HOURS
SYNTH_HOURS = 218.0
STEREO_HOURS = SYNTH_HOURS + DATAHIVE_TRAIN_HOURS

# $9.00 hard stop. 1.9 h at $4.39/h is about $8.34, so the clock hits
# first and leaves margin under the dollar cap and under the $9.45 balance.
MAX_USD_CENTS = 900
MAX_WALL_S = int(1.9 * 3600)  # 6840 s
HOURLY_CENTS = 439
# Refuse to create the pod when the phase estimate is above this.
ABORT_CENTS = 800

# The 182 h retry used 48 GB after the raw downloads were deleted.
# ~500 h at that rate is about 132 GB of audio. 250 GB of container disk
# leaves room for the raw zip/tar before it is deleted, plus one checkpoint.
# No network volume: container disk is erased on terminate.
REFERENCE_JOB_HOURS = 182.0
REFERENCE_DISK_GB = 48.0
DISK_GB = 250
# RunPod container disk, running pod, from the pricing page (Nov 2026).
DISK_CENTS_PER_GB_MONTH = 10
MONTH_S = 30 * 24 * 3600

# Measured on the one-H100 retry.
REFERENCE_DOWNLOAD_MIN = 2.1
MONO_REALTIME = 1145.0
STEREO_REALTIME = 1412.0
STEP_S = 0.160


def training_plan(gpus: int = 1) -> dict:
    s1 = _steps(MONO_HOURS, MONO_PASSES, gpus)
    s2 = _steps(STEREO_HOURS, STEREO_PASSES_S2, gpus, STEREO_WEIGHT)
    s3 = _steps(STEREO_HOURS, STEREO_PASSES_S3, gpus, STEREO_WEIGHT)
    return {
        "gpus": gpus,
        "name": NAME,
        "mono_hours": round(MONO_HOURS, 3),
        "synth_hours": SYNTH_HOURS,
        "stereo_hours": round(STEREO_HOURS, 3),
        "s1_stop": s1,
        "s2_end": s1 + s2,
        "s3_end": s1 + s2 + s3,
        "s2_steps": s2,
        "s3_steps": s3,
        "s1_warmup": S1_WARMUP,
        "s3_warmup": S3_WARMUP,
        "eval_every": S3_EVAL_EVERY,
        "save_during_stage": True,
    }


def audio_hours() -> float:
    return MONO_HOURS + SYNTH_HOURS + DATAHIVE_TRAIN_HOURS


def disk_cents(gb: int = DISK_GB, seconds: int = MAX_WALL_S) -> int:
    """Container-disk rent for a running pod. Volume disk stays 0."""
    return int(round(gb * DISK_CENTS_PER_GB_MONTH * seconds / MONTH_S))


def estimate(hourly_cents: int = HOURLY_CENTS) -> dict:
    """Wall time and GPU cost from the retry's measured rates."""
    plan = training_plan()
    steps = plan["s1_stop"] + plan["s2_steps"] + plan["s3_steps"]
    download_s = REFERENCE_DOWNLOAD_MIN * 60.0 * (audio_hours() / REFERENCE_JOB_HOURS)
    mono_s = MONO_HOURS * 3600.0 / MONO_REALTIME
    stereo_s = STEREO_HOURS * 3600.0 / STEREO_REALTIME
    train_s = steps * STEP_S
    phases = [
        ("download", download_s),
        ("tokenize mono", mono_s),
        ("tokenize stereo", stereo_s),
        ("train", train_s),
    ]
    total_s = sum(sec for _, sec in phases)
    rows = []
    for name, sec in phases:
        cents = int(round(sec / 3600.0 * hourly_cents))
        rows.append({"name": name, "seconds": sec, "cents": cents})
    gpu_cents = int(round(total_s / 3600.0 * hourly_cents))
    return {
        "phases": rows,
        "seconds": total_s,
        "gpu_cents": gpu_cents,
        "disk_cents": disk_cents(),
        "steps": steps,
        "plan": plan,
        "over_abort": gpu_cents > ABORT_CENTS,
    }


def explain_estimate(est: dict | None = None) -> str:
    est = est or estimate()
    lines = []
    for row in est["phases"]:
        lines.append(
            f"  {row['name']}: {row['seconds'] / 60.0:.1f} min, ${row['cents'] / 100:.2f}"
        )
    lines.append(
        f"  GPU total: {est['seconds'] / 60.0:.1f} min, ${est['gpu_cents'] / 100:.2f} at $4.39/h"
    )
    lines.append(
        f"  container disk: {DISK_GB} GB at $0.10/GB/month while the pod runs, "
        f"about ${est['disk_cents'] / 100:.2f} for {MAX_WALL_S / 3600:.1f} h. "
        f"volumeInGb is 0, so disk billing ends when the pod is terminated."
    )
    lines.append(
        f"  scaled from {REFERENCE_DISK_GB:.0f} GB for {REFERENCE_JOB_HOURS:.0f} h "
        f"after raw was deleted; audio is deleted again after each tokenize."
    )
    if est["over_abort"]:
        lines.append("  REFUSING: the GPU estimate is above $8.")
    return "\n".join(lines)
