#!/usr/bin/env python3
"""CPU check: one-frame Mimi decode clicks, a code window does not.

The warmstart export, when it is on disk at
runpod/retrieved/h5sfw2sl3oudv5/caramel-it, is loaded strictly and driven
for two seconds with the gate off. This checkout often does not have that
folder (it is gitignored). The codec check still runs: it is the tick.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "chat"))
sys.path.insert(0, str(ROOT / "third_party" / "caramel"))

from live import DECODE_CONTEXT, FRAME, SR, LiveLoop, decode_window  # noqa: E402
from server import check_export  # noqa: E402

EXPORT = ROOT / "runpod" / "retrieved" / "h5sfw2sl3oudv5" / "caramel-it"


def _voiced(n_frames: int) -> np.ndarray:
    """A steady vowel, so a click at each frame boundary is obvious."""
    n = FRAME * n_frames
    t = np.arange(n) / SR
    wave = (
        0.30 * np.sin(2 * np.pi * 140 * t)
        + 0.15 * np.sin(2 * np.pi * 280 * t)
        + 0.08 * np.sin(2 * np.pi * 420 * t)
    )
    return wave.astype(np.float32)


def _boundary(y: np.ndarray) -> tuple[float, float]:
    y = np.asarray(y, dtype=np.float32)
    y = y[: (len(y) // FRAME) * FRAME]
    blocks = y.reshape(-1, FRAME)
    jump = np.abs(blocks[1:, 0] - blocks[:-1, -1])
    step = np.mean(np.abs(np.diff(y)))
    return float(jump.mean()), float(step)


def check_clicks() -> None:
    from transformers import MimiModel

    mimi = MimiModel.from_pretrained("kyutai/mimi").eval()
    wave = _voiced(16)
    with torch.no_grad():
        codes = mimi.encode(torch.from_numpy(wave)[None, None, :], num_quantizers=8).audio_codes
        full = mimi.decode(codes).audio_values[0, 0].numpy()
    per = []
    for i in range(codes.shape[-1]):
        with torch.no_grad():
            piece = mimi.decode(codes[:, :, i : i + 1]).audio_values[0, 0].numpy()
        per.append(piece[-FRAME:])
    per_jump, per_step = _boundary(np.concatenate(per))
    window = []
    hist = []
    for i in range(codes.shape[-1]):
        hist.append(codes[0, :, i].to(dtype=torch.long))
        if len(hist) > DECODE_CONTEXT:
            del hist[0]
        if len(hist) < DECODE_CONTEXT:
            continue
        with torch.no_grad():
            tail = decode_window(mimi, torch.stack(hist, dim=0)).detach().cpu().numpy()
        window.append(tail)
    win_jump, win_step = _boundary(np.concatenate(window))
    full_jump, full_step = _boundary(full)
    if per_jump < per_step * 8:
        raise SystemExit(f"per-frame decode was not clicky: jump {per_jump:.4f} step {per_step:.4f}")
    if win_jump > win_step * 3:
        raise SystemExit(f"windowed decode still clicks: jump {win_jump:.4f} step {win_step:.4f}")
    if full_jump > full_step * 3:
        raise SystemExit(f"full decode clicks: jump {full_jump:.4f} step {full_step:.4f}")
    rms = float(np.sqrt(np.mean(np.concatenate(window) ** 2)))
    if rms < 0.05:
        raise SystemExit(f"windowed vowel collapsed: rms {rms:.4f}")
    print(
        f"codec ok: per-frame jump/step {per_jump / per_step:.1f}x, "
        f"window {win_jump / win_step:.2f}x, rms {rms:.3f}"
    )


def _tiny_export(dest: Path, *, extra_config=None, drop_config=None, extra_tensor=False) -> None:
    from model import Caramel, Config
    from safetensors.torch import save_file

    cfg = Config(dim=32, num_layers=1, num_heads=4, dep_dim=16, dep_layers=1, dep_heads=4, hidden_scale=2.0, dep_hidden_scale=2.0)
    raw = {k: getattr(cfg, k) for k in Config.__dataclass_fields__}
    if extra_config:
        raw.update(extra_config)
    if drop_config:
        raw.pop(drop_config, None)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    with torch.device("meta"):
        keys = list(Caramel(cfg).state_dict())
    blobs = {k: torch.zeros(1) for k in keys}
    if extra_tensor:
        blobs["not.a.real.weight"] = torch.zeros(1)
    # Real shapes are required for a passing check. Build on CPU for the good case.
    if not extra_tensor and not extra_config and not drop_config:
        model = Caramel(cfg)
        save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(dest / "model.safetensors"))
    else:
        save_file(blobs, str(dest / "model.safetensors"))


def check_strict() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        good = Path(tmp) / "good"
        _tiny_export(good)
        check_export(good)
        bad_cfg = Path(tmp) / "cfg"
        _tiny_export(bad_cfg, extra_config={"optimizer": "adam"})
        try:
            check_export(bad_cfg)
        except SystemExit as exc:
            if "unexpected" not in str(exc):
                raise
        else:
            raise SystemExit("an extra config key was accepted")
        missing = Path(tmp) / "missing"
        _tiny_export(missing, drop_config="delays")
        try:
            check_export(missing)
        except SystemExit as exc:
            if "missing" not in str(exc):
                raise
        else:
            raise SystemExit("a config without delays was accepted")
        extra = Path(tmp) / "extra"
        _tiny_export(extra, extra_tensor=True)
        try:
            check_export(extra)
        except SystemExit as exc:
            if "unexpected" not in str(exc):
                raise
        else:
            raise SystemExit("an unexpected tensor was accepted")
    print("strict export check ok")


def _token_line(rows: list[list[int]]) -> str:
    arr = np.asarray(rows, dtype=np.int64)
    parts = []
    for k in range(arr.shape[1]):
        col = arr[:, k]
        parts.append(f"cb{k}:unique={len(set(col.tolist()))} zero={(col == 0).mean():.2f} mean={col.mean():.0f}")
    return " ".join(parts)


def check_export_loop() -> None:
    if not (EXPORT / "config.json").is_file() or not (EXPORT / "model.safetensors").is_file():
        print(f"warmstart export not in this checkout ({EXPORT}); codec check covers the tick")
        return
    from transformers import MimiModel
    from model import load

    check_export(EXPORT)
    dev = "cpu"
    model = load(str(EXPORT), dev)
    mimi = MimiModel.from_pretrained("kyutai/mimi").to(dev).eval()
    loop = LiveLoop(model, mimi, gate=False)
    audio = []
    # Two seconds plus the eight-frame warmup. Gate off, so this is the model's voice.
    for frame in _voiced(DECODE_CONTEXT + 25).reshape(-1, FRAME):
        result = loop.push(frame)
        if result.pcm.size:
            audio.append(result.pcm)
    if not audio:
        raise SystemExit("the warmstart export produced no audio")
    cat = np.concatenate(audio)
    rms = float(np.sqrt(np.mean(cat ** 2)))
    jump, step = _boundary(cat)
    print(f"export audio rms={rms:.4f} jump/step={jump / max(step, 1e-8):.2f}")
    print("tokens", _token_line(loop.heard_codes[-len(audio):]))
    if rms < 0.01:
        raise SystemExit(f"export audio is silence (rms {rms:.4f})")
    if jump > step * 4:
        raise SystemExit(f"export audio still clicks: jump {jump:.4f} step {step:.4f}")
    print("warmstart export listen ok")


def main() -> int:
    check_clicks()
    check_strict()
    check_export_loop()
    print("listen test ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
