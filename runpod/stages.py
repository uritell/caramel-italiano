"""The three upstream train.py invocations for this pilot.

Step ends are the ones set for the pilot: stage 1 stops at 401 on 1 GPU,
stage 2 runs on to 1072 and stage 3 to 1981, both on 2 GPUs. Stages 1 and 2
share one cosine schedule that ends at step 1072. Stage 3 restarts it there
with the published stage-3 peak and warmup.

`--eval_every 200` is not the published 1000. A stride of 1000 never lands
between steps 1072 and 1981, so `--save_best` would not run. 200 still
evaluates a handful of times on this short stage.
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

S1_STOP = 401
S2_END = 1072
S3_END = 1981
S3_WARMUP = 200
S3_LR = "1.5e-4"
S3_EVAL_EVERY = 200
S3_PATIENCE = 2

# Tiny network for the CPU smoke test only. The rented run does not pass --cfg.
CPU_CFG = {
    "dim": 64,
    "num_layers": 1,
    "num_heads": 4,
    "dep_dim": 32,
    "dep_layers": 1,
    "dep_heads": 4,
    "hidden_scale": 4.0,
    "dep_hidden_scale": 4.0,
}


def _torchrun(py: str, nproc: int) -> list[str]:
    exe = shutil.which("torchrun")
    if exe:
        return [exe, "--nproc_per_node", str(nproc)]
    return [py, "-m", "torch.distributed.run", "--nproc_per_node", str(nproc)]


def full_commands(py: str, mono: Path, stereo: Path, dev: Path, runs: Path) -> list[tuple[str, list[str], dict[str, str]]]:
    """(name, argv, extra env) for the three GPU stages. cwd is caramel/."""
    s1, s2, s3 = runs / "s1", runs / "s2", runs / "s3"
    common = ["--workers", "6"]
    stage1 = [
        py, "train.py",
        "--data", str(mono),
        "--steps", str(S2_END),
        "--stop_at", str(S1_STOP),
        *common,
        "--out", str(s1),
    ]
    stage2 = [
        *_torchrun(py, 2), "train.py",
        "--data", str(stereo), str(mono),
        "--weights", "0.8", "0.2",
        "--steps", str(S2_END),
        "--resume", str(s1 / "ckpt.pt"),
        *common,
        "--out", str(s2),
    ]
    stage3 = [
        *_torchrun(py, 2), "train.py",
        "--data", str(stereo), str(mono),
        "--weights", "0.8", "0.2",
        "--steps", str(S3_END),
        "--stop_at", str(S3_END),
        "--resume", str(s2 / "ckpt.pt"),
        "--sched_start", str(S2_END),
        "--warmup", str(S3_WARMUP),
        "--lr", S3_LR,
        "--dev", str(dev),
        "--eval_every", str(S3_EVAL_EVERY),
        "--patience", str(S3_PATIENCE),
        "--save_best",
        *common,
        "--out", str(s3),
    ]
    # Stage 1 is one GPU. The pod still has two; the idle one is billed.
    return [
        ("s1", stage1, {"CUDA_VISIBLE_DEVICES": "0"}),
        ("s2", stage2, {}),
        ("s3", stage3, {}),
    ]


def cpu_smoke_commands(py: str, mono: Path, stereo: Path, dev: Path, runs: Path) -> list[tuple[str, list[str], dict[str, str]]]:
    """A few CPU steps that exercise stop, resume, and --save_best.

    Upstream train.py runs on CPU when CUDA is absent. Window lengths are
    hardcoded in train.py (60s and up), so this uses a tiny --cfg. It is a
    wiring test, not a small version of the real model.
    """
    cfg = json.dumps(CPU_CFG, separators=(",", ":"))
    s1, s2, s3 = runs / "s1", runs / "s2", runs / "s3"
    base = ["--fpb", "750", "--workers", "0", "--cfg", cfg]
    stage1 = [
        py, "train.py",
        "--data", str(mono),
        "--steps", "4", "--stop_at", "2",
        *base, "--out", str(s1),
    ]
    stage2 = [
        py, "train.py",
        "--data", str(stereo), str(mono),
        "--weights", "0.8", "0.2",
        "--steps", "4",
        "--resume", str(s1 / "ckpt.pt"),
        *base, "--out", str(s2),
    ]
    stage3 = [
        py, "train.py",
        "--data", str(stereo), str(mono),
        "--weights", "0.8", "0.2",
        "--steps", "6", "--stop_at", "6",
        "--resume", str(s2 / "ckpt.pt"),
        "--sched_start", "4", "--warmup", "1", "--lr", "1.5e-4",
        "--dev", str(dev),
        "--eval_every", "1", "--patience", "0", "--save_best",
        *base, "--out", str(s3),
    ]
    return [("s1", stage1, {}), ("s2", stage2, {}), ("s3", stage3, {})]


def format_commands(commands) -> str:
    lines = []
    for name, argv, env in commands:
        prefix = " ".join(f"{k}={v}" for k, v in env.items())
        rendered = " ".join(shlex.quote(a) for a in argv)
        lines.append(f"# {name}\n{prefix + ' ' if prefix else ''}{rendered}")
    return "\n\n".join(lines) + "\n"


def assert_full_numbers(commands) -> None:
    blob = " ".join(" ".join(argv) for _, argv, _ in commands)
    for needle in (str(S1_STOP), str(S2_END), str(S3_END), "0.8", "0.2", "--save_best", "1.5e-4"):
        if needle not in blob:
            raise SystemExit(f"pilot command is missing {needle}")
    if commands[0][2].get("CUDA_VISIBLE_DEVICES") != "0":
        raise SystemExit("stage 1 must be pinned to one GPU")
    if "--nproc_per_node" not in " ".join(commands[1][1][:4]):
        raise SystemExit("stage 2 must launch two processes")
    if "--nproc_per_node" not in " ".join(commands[2][1][:4]):
        raise SystemExit("stage 3 must launch two processes")


def run_commands(commands, cwd: Path, log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    for name, argv, extra in commands:
        log_path = log_dir / f"{name}.stdout"
        print(f"=== {name} ===", flush=True)
        print(" ".join(shlex.quote(a) for a in argv), flush=True)
        env = os.environ.copy()
        env.update(extra)
        env["PYTHONUNBUFFERED"] = "1"
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(argv, cwd=str(cwd), env=env, stdout=log, stderr=subprocess.STDOUT)
        text = log_path.read_text(encoding="utf-8", errors="replace")
        tail = "\n".join(text.splitlines()[-30:])
        if tail:
            print(tail, flush=True)
        if proc.returncode != 0:
            raise SystemExit(f"{name} exited {proc.returncode}; see {log_path}")


def main(argv: list[str]) -> int:
    mode = "print-full"
    if "--cpu-smoke" in argv:
        mode = "print-cpu"
    if "--check" in argv:
        mono = stereo = dev = Path("/data")
        runs = Path("/runs")
        cmds = full_commands(sys.executable, mono, stereo, dev, runs)
        assert_full_numbers(cmds)
        cpu = cpu_smoke_commands(sys.executable, mono, stereo, dev, runs)
        if "CUDA_VISIBLE_DEVICES" in cpu[0][1]:
            raise SystemExit("CPU smoke must not pin a GPU")
        print("stages ok")
        return 0
    if mode == "print-full":
        print(format_commands(full_commands(
            sys.executable, Path("MONO"), Path("STEREO"), Path("DEV"), Path("runs"))))
    else:
        print(format_commands(cpu_smoke_commands(
            sys.executable, Path("MONO"), Path("STEREO"), Path("DEV"), Path("runs"))))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
