"""The three upstream train.py invocations for the one-GPU retry.

All three stages use one GPU. Step counts come from pilot_config.py: the
same pass targets as the 2-GPU plan, on 100 h of mono and about 83 h of
stereo, so they are not 401 / 1072 / 1981.
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from pilot_config import plan as _plan

_P = _plan()
S1_STOP = _P["s1_stop"]
S1_WARMUP = _P["s1_warmup"]
S2_END = _P["s2_end"]
S3_END = _P["s3_end"]
S3_WARMUP = _P["s3_warmup"]
S3_LR = "1.5e-4"
S3_EVAL_EVERY = _P["eval_every"]
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


def _one_gpu(py: str, extra: list[str], out: Path) -> list[str]:
    return [py, "train.py", *extra, "--workers", "4", "--out", str(out)]


def full_commands(py: str, mono: Path, stereo: Path, dev: Path, runs: Path) -> list[tuple[str, list[str], dict[str, str]]]:
    """(name, argv, extra env). Every stage is one process on GPU 0."""
    s1, s2, s3 = runs / "s1", runs / "s2", runs / "s3"
    pin = {"CUDA_VISIBLE_DEVICES": "0"}
    stage1 = _one_gpu(py, [
        "--data", str(mono),
        "--steps", str(S2_END),
        "--stop_at", str(S1_STOP),
        "--warmup", str(S1_WARMUP),
    ], s1)
    stage2 = _one_gpu(py, [
        "--data", str(stereo), str(mono),
        "--weights", "0.8", "0.2",
        "--steps", str(S2_END),
        "--resume", str(s1 / "ckpt.pt"),
    ], s2)
    stage3 = _one_gpu(py, [
        "--data", str(stereo), str(mono),
        "--weights", "0.8", "0.2",
        "--steps", str(S3_END),
        "--resume", str(s2 / "ckpt.pt"),
        "--sched_start", str(S2_END),
        "--warmup", str(S3_WARMUP),
        "--lr", S3_LR,
        "--dev", str(dev),
        "--eval_every", str(S3_EVAL_EVERY),
        "--patience", str(S3_PATIENCE),
        "--save_best",
    ], s3)
    return [("s1", stage1, pin), ("s2", stage2, pin), ("s3", stage3, pin)]


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
    for needle in (
        str(S1_STOP), str(S2_END), str(S3_END), "0.8", "0.2", "--save_best", "1.5e-4",
        "--warmup 2", "--warmup 8", "--eval_every 100",
    ):
        if needle not in blob:
            raise SystemExit(f"pilot command is missing {needle}")
    for name, argv, env in commands:
        if env.get("CUDA_VISIBLE_DEVICES") != "0":
            raise SystemExit(f"{name} must be pinned to one GPU")
        if "--nproc_per_node" in argv or "torch.distributed.run" in argv:
            raise SystemExit(f"{name} must not launch a second GPU")


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
        publish = os.environ.get("PILOT_OUT")
        if publish:
            out_dir = Path(argv[argv.index("--out") + 1])
            dest = Path(publish) / "checkpoints" / name
            dest.mkdir(parents=True, exist_ok=True)
            for fname in ("ckpt.pt", "best.pt"):
                src = out_dir / fname
                if src.is_file():
                    shutil.copy2(src, dest / fname)
                    print(f"published {name}/{fname}", flush=True)


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
