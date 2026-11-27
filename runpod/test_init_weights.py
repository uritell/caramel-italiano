#!/usr/bin/env python3
"""CPU check for --init-weights. Tiny random weights, no download and no pod."""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "runpod" / "patches" / "train-init-weights.patch"


def main() -> int:
    subprocess.run(
        ["patch", "-d", str(ROOT / "third_party" / "caramel"), "-p1", "--dry-run", "-i", str(PATCH)],
        check=True,
    )
    work = Path(tempfile.mkdtemp(prefix="init-weights-"))
    try:
        shutil.copytree(ROOT / "third_party" / "caramel", work / "caramel")
        subprocess.run(["patch", "-d", str(work / "caramel"), "-p1", "-i", str(PATCH)], check=True)
        sys.path.insert(0, str(work / "caramel"))
        import torch
        from safetensors.torch import save_file
        import model
        import train

        cfg = model.Config(
            dim=32, num_layers=1, num_heads=4, dep_dim=16, dep_layers=1, dep_heads=4,
            hidden_scale=4.0, dep_hidden_scale=4.0,
        )
        src = model.Caramel(cfg)
        with torch.no_grad():
            for p in src.parameters():
                p.normal_()
        path = work / "model.safetensors"
        save_file({k: v.detach().cpu().contiguous() for k, v in src.state_dict().items()}, str(path))
        fresh = model.Caramel(cfg)
        opt = torch.optim.AdamW(fresh.parameters(), lr=1e-4)
        state_before = opt.state_dict()
        train._load_init_weights(fresh, str(path))
        for k, v in src.state_dict().items():
            if not torch.equal(fresh.state_dict()[k].cpu(), v.cpu()):
                raise SystemExit(f"weight {k} was not copied")
        if opt.state_dict()["state"] != state_before["state"]:
            raise SystemExit("init-weights touched the optimizer")
        bad = model.Caramel(model.Config(
            dim=64, num_layers=1, num_heads=4, dep_dim=16, dep_layers=1, dep_heads=4,
            hidden_scale=4.0, dep_hidden_scale=4.0,
        ))
        try:
            train._load_init_weights(bad, str(path))
        except SystemExit as exc:
            if "shape_mismatch" not in str(exc):
                raise SystemExit(f"shape mismatch was not loud: {exc}")
        else:
            raise SystemExit("a shape mismatch was accepted")
        text = (work / "caramel" / "train.py").read_text(encoding="utf-8")
        if "fresh optimizer" not in text or "--init-weights" not in text:
            raise SystemExit("patched train.py lost the flag")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print("init-weights ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
