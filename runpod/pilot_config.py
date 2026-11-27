"""Size of the one-GPU retry. One place so download, synth, and train agree.

The earlier plan was 258 h of mono and 239 h of stereo on 2 GPUs, which
landed at steps 401 / 1072 / 1981. This retry is smaller and uses one GPU,
so those step counts do not keep the same passes. The numbers below are
the same pass targets (0.83, then 2.4, then 3.25 more) on the smaller pools,
with one GPU's 24,000 frames per step.
"""
from __future__ import annotations

GPU_COUNT = 1
MONO_HOURS = 100.0
SYNTH_STEREO_HOURS = 80.0
# DataHive sample is 3 h 3 m 4 s. The shortest call (it_000002, 440.8 s)
# stays held out for dev, as in the measured plan.
DATAHIVE_TRAIN_HOURS = 2.929
STEREO_HOURS = SYNTH_STEREO_HOURS + DATAHIVE_TRAIN_HOURS

FRAMES_PER_HOUR = 12.5 * 3600  # 45_000
FPB = 24000
MONO_PASSES = 0.83
STEREO_PASSES_S2 = 2.4
STEREO_PASSES_S3 = 3.25
STEREO_WEIGHT = 0.8
S1_WARMUP = 2
S3_WARMUP = 8
S3_EVAL_EVERY = 100
S3_LR = "1.5e-4"
S3_PATIENCE = 2


def _steps(hours: float, passes: float, gpus: int, weight: float = 1.0) -> int:
    frames = passes * hours * FRAMES_PER_HOUR / weight
    return int(round(frames / (FPB * gpus)))


def plan(gpus: int = GPU_COUNT) -> dict:
    s1 = _steps(MONO_HOURS, MONO_PASSES, gpus)
    s2 = _steps(STEREO_HOURS, STEREO_PASSES_S2, gpus, STEREO_WEIGHT)
    s3 = _steps(STEREO_HOURS, STEREO_PASSES_S3, gpus, STEREO_WEIGHT)
    return {
        "gpus": gpus,
        "mono_hours": MONO_HOURS,
        "synth_hours": SYNTH_STEREO_HOURS,
        "stereo_hours": round(STEREO_HOURS, 3),
        "s1_stop": s1,
        "s2_end": s1 + s2,
        "s3_end": s1 + s2 + s3,
        "s2_steps": s2,
        "s3_steps": s3,
        "s1_warmup": S1_WARMUP,
        "s3_warmup": S3_WARMUP,
        "eval_every": S3_EVAL_EVERY,
    }


def explain() -> str:
    p = plan()
    return (
        f"1 GPU, {MONO_HOURS:.0f} h mono, {SYNTH_STEREO_HOURS:.0f} h synthetic stereo "
        f"+ {DATAHIVE_TRAIN_HOURS:.3f} h DataHive train "
        f"({p['stereo_hours']} h stereo). "
        f"Steps: stop {p['s1_stop']}, stage-2 end {p['s2_end']}, "
        f"stage-3 end {p['s3_end']}."
    )
