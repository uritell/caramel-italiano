#!/usr/bin/env python3
"""Placeholder: encode 24 kHz speaker tracks with kyutai/mimi (8 codebooks)
into a caramel TokenStore folder (tokens.bin, index.npy, meta.json).

See tokenize/README.md for the exact dtypes and shapes read by
third_party/caramel/train.py. This script does not load Mimi or write
tokens yet.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, help="data/work/manifest.jsonl (or synth manifest)")
    p.add_argument("--out", type=Path, required=False, help="store folder to create")
    p.add_argument("--width", type=int, choices=(8, 16), default=None)
    args = p.parse_args()
    print("tokenize.py is a placeholder. It will not encode audio.")
    print(f"planned manifest: {args.manifest}")
    print(f"planned out:      {args.out}")
    print(f"planned width:    {args.width}")
    print("TODO: resample to 24 kHz, MimiModel.from_pretrained('kyutai/mimi'),")
    print("TODO: encode with num_quantizers=8, write uint16 tokens.bin + index.npy + meta.json.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
