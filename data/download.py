#!/usr/bin/env python3
"""Placeholder: download the four clean-pilot Italian speech sources.

This script is intentionally a no-op. It does not create accounts, does
not hit the network, and does not write audio. Implement the fetchers
only after the owner has approved any signup (Common Voice / MDC) and
any machine or disk use.

Pilot sources (see data/README.md and DATA_LICENSES.md):

- datahiveai/italian-conversations-sample (CC BY 4.0)
- google/fleurs, config it_it (CC BY 4.0)
- Multilingual LibriSpeech Italian, OpenSLR 94 (CC BY 4.0)
- Common Voice Scripted Speech 27.0 Italian via Mozilla Data Collective
  (CC0 + no re-host / no speaker-id; MDC account required)

Outputs should land in data/raw/ (gitignored).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"

SOURCES = (
    "datahive_it_conversations_sample",
    "fleurs_it_it",
    "mls_it",
    "common_voice_it_27",
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=RAW, help="raw download directory")
    p.add_argument(
        "--source",
        choices=SOURCES,
        action="append",
        help="subset to fetch (repeatable)",
    )
    args = p.parse_args()
    wanted = args.source or list(SOURCES)
    print("download.py is a placeholder. It will not download anything.")
    print(f"planned out dir: {args.out}")
    print("planned sources:")
    for s in wanted:
        print(f"  - {s}")
    print("TODO: implement licensed fetches. Common Voice must go through MDC;")
    print("TODO: do not re-host Common Voice; do not commit audio.")
    return 2  # explicit "not implemented"


if __name__ == "__main__":
    sys.exit(main())
