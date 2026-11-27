#!/usr/bin/env python3
"""Placeholder: build a clip-level manifest from data/raw/.

Each JSONL record should eventually include: source id, licence, attribution
tag, split, path, duration_s, sample_rate, channels, speaker_id if the
source provides one (never inferred for Common Voice), and whether the
clip is mono or already stereo-separated.

This script does not read or write audio yet.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "work" / "manifest.jsonl"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, default=ROOT / "raw")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    print("manifest.py is a placeholder. It will not scan or write a manifest.")
    print(f"planned raw dir: {args.raw}")
    print(f"planned out:     {args.out}")
    print("TODO: walk data/raw/, record licence tags from DATA_LICENSES.md,")
    print("TODO: and keep Common Voice clips flagged as non-redistributable.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
