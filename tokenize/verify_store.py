#!/usr/bin/env python3
"""Load a TokenStore with upstream train.py and run delay_window on it.

Imports third_party/caramel/train.py. Does not copy TokenStore.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CARAMEL = ROOT / "third_party" / "caramel"


def _import_train():
    sys.path.insert(0, str(CARAMEL))
    from train import TokenStore, delay_window

    return TokenStore, delay_window


def verify(store: Path) -> dict:
    TokenStore, delay_window = _import_train()
    loaded = TokenStore(str(store))
    meta = json.loads((store / "meta.json").read_text(encoding="utf-8"))
    if int(loaded.W) != int(meta["width"]):
        raise RuntimeError("width mismatch")
    if tuple(loaded.tok.shape) != (int(meta["frames"]), int(meta["width"])):
        raise RuntimeError(f"memmap shape {loaded.tok.shape} != meta")
    if loaded.index.ndim != 2 or loaded.index.shape[1] != 2:
        raise RuntimeError(f"index shape {loaded.index.shape}")
    if loaded.index.dtype != np.int64:
        raise RuntimeError(f"index dtype {loaded.index.dtype}")
    total = 0
    for i in range(loaded.index.shape[0]):
        offset, length = (int(loaded.index[i, 0]), int(loaded.index[i, 1]))
        if offset < 0 or length <= 0 or offset + length > loaded.tok.shape[0]:
            raise RuntimeError(f"clip {i} offset {offset} length {length} out of range")
        if i > 0 and offset < int(loaded.index[i - 1, 0]) + int(loaded.index[i - 1, 1]):
            raise RuntimeError(f"clip {i} overlaps the previous clip")
        total += length
    if total != loaded.tok.shape[0]:
        raise RuntimeError(f"index covers {total} frames, store has {loaded.tok.shape[0]}")
    length0 = int(loaded.index[0, 1])
    # Upstream windows can read past a short clip into the next one. The check
    # stays inside this clip so it is testing the store, not that quirk.
    window = min(length0, 750)
    raw = np.asarray(loaded.window(0, 0, window))
    if raw.shape != (window, loaded.W):
        raise RuntimeError(f"window shape {raw.shape}")
    seq = delay_window(raw, window, False)
    seq_swap = delay_window(raw, window, True)
    if seq.shape != (window + 1, 16) or seq_swap.shape != (window + 1, 16):
        raise RuntimeError(f"delay_window shapes {seq.shape} {seq_swap.shape}")
    if loaded.W == 8:
        # Mono: the second speaker is masked, both orders.
        if not np.all(seq[:, 8:] == -1) or not np.all(seq_swap[:, 8:] == -1):
            raise RuntimeError("mono delay_window did not mask the second speaker")
    report = {
        "store": str(store),
        "width": int(loaded.W),
        "frames": int(loaded.tok.shape[0]),
        "clips": int(loaded.index.shape[0]),
        "delay_window": list(seq.shape),
        "token_min": int(loaded.tok.min()),
        "token_max": int(loaded.tok.max()),
        "imported_from": str(CARAMEL / "train.py"),
    }
    print(json.dumps(report, indent=2), flush=True)
    return report


def _self_test() -> None:
    import tempfile

    TokenStore, delay_window = _import_train()
    with tempfile.TemporaryDirectory() as tmp:
        store = Path(tmp)
        frames, width = 20, 8
        tokens = np.arange(frames * width, dtype=np.uint16).reshape(frames, width) % 2048
        mm = np.memmap(store / "tokens.bin", dtype=np.uint16, mode="w+", shape=(frames, width))
        mm[:] = tokens
        mm.flush()
        del mm
        np.save(store / "index.npy", np.array([[0, frames]], dtype=np.int64))
        (store / "meta.json").write_text(json.dumps({"width": width, "frames": frames}), encoding="utf-8")
        verify(store)
    print("verify_store self-test ok")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", type=Path, action="append")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        _self_test()
    if not args.store and not args.self_test:
        p.error("pass --store and/or --self-test")
    for store in args.store or []:
        verify(store)
    return 0


if __name__ == "__main__":
    sys.exit(main())
