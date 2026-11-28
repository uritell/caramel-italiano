"""Plan the lean Italian audio add-on. No network and no pod.

VoxPopuli Italian unlabelled audio for 2015 only, from the official tars
``it_2015.tar`` and ``it_2015_2.tar`` (CC0), capped at 1,000 h. Plus the
MLS Italian hours that are not already in the mono store: the archive is
257.83 h and the mono store already took the first 100 h in archive order,
so about 158 h remain.

Segments are 12 s from the start of each file, the same turn cap
``prepare_audio.TURN_CAP_S`` uses, and files stay in archive order.
"""
from __future__ import annotations

import math
from pathlib import Path

import cost

VOX_BASE = "https://dl.fbaipublicfiles.com/voxpopuli/audios"
VOX_TARS = ("it_2015.tar", "it_2015_2.tar")
VOX_YEAR = "2015"
VOX_CAP_H = 1000.0
MLS_URL = "https://dl.fbaipublicfiles.com/mls/mls_italian.tar.gz"
MLS_TOTAL_H = 257.83
MLS_ALREADY_H = 100.0
MLS_KEEP_H = 158.0  # 257.83 - 100 = 157.83; 158 h holds the remainder
SEGMENT_S = 12.0  # prepare_audio.TURN_CAP_S, the mono turn length
MIN_SEGMENT_S = 0.2
SPEED_TEST_S = 300.0  # five minutes
# Allowance for the tars, not a measurement. The speed test replaces the encode term.
DOWNLOAD_ALLOWANCE_S = 30 * 60
# H100 mono factor from option A. Dry-run labels it as that measurement.
# The pod's five-minute encode is what the launcher trusts.
REFERENCE_RTF = 1145.0
A40_HOURLY_CENTS = 59
# Data prep only. The approved $4–5.50 total also has to cover the later train.
MAX_USD_CENTS = 300  # $3.00
MAX_WALL_S = 6 * 3600
# Launcher gives up if the pull server is still down this long after RUNNING.
PROXY_WAIT_S = 10 * 60
DISK_GB = 200
STORE_VOX = "mono_voxpopuli"
STORE_MLS = "mono_mls2"
STORE_SYNTH = "synth_stereo"
WIDTH = 8
AUDIO_EXT = {".ogg", ".wav", ".flac", ".mp3", ".opus"}

ATTRIBUTION = {
    "voxpopuli_it_2015": (
        "VoxPopuli Italian unlabelled parliamentary audio, 2015 "
        "(Wang et al., 2021), CC0, https://github.com/facebookresearch/voxpopuli, "
        "tars it_2015.tar and it_2015_2.tar. European Parliament source acknowledged: "
        "https://www.europarl.europa.eu/legal-notice/en/. "
        "VoxPopuli pretrained models and code are CC BY-NC 4.0 and are not used."
    ),
    "mls_it_remainder": (
        "Multilingual LibriSpeech Italian (Pratap et al., 2020), CC BY 4.0, "
        "https://www.openslr.org/94/. This slice is the archive order after the "
        "first 100 h already in the mono store (about 158 h). "
        "Derived from public-domain LibriVox and Project Gutenberg."
    ),
}


def vox_urls() -> list[str]:
    return [f"{VOX_BASE}/{name}" for name in VOX_TARS]


def event_year(member: str) -> str | None:
    """Year of a VoxPopuli audio member. Directory ``2015`` or an id starting YYYYMMDD."""
    for part in Path(member).parts:
        if len(part) == 4 and part.isdigit():
            return part
    stem = Path(member).stem
    head = stem.split("_", 1)[0]
    if len(head) >= 4 and head[:4].isdigit():
        return head[:4]
    if len(stem) >= 4 and stem[:4].isdigit():
        return stem[:4]
    return None


def verify_2015_members(names: list[str]) -> list[str]:
    """Return the audio members. Refuse the tar unless every one is a 2015 event."""
    audio = [name for name in names if Path(name).suffix.lower() in AUDIO_EXT]
    if not audio:
        raise SystemExit("tar has no audio members")
    bad = [name for name in audio if event_year(name) != VOX_YEAR]
    if bad:
        raise SystemExit(f"tar is not only {VOX_YEAR} events, for example {bad[0]}")
    return audio


def segment_duration(duration_s: float, segment_s: float = SEGMENT_S, min_s: float = MIN_SEGMENT_S) -> list[tuple[float, float]]:
    """(start_s, dur_s) cuts from t=0. Same 12 s cap as the mono dialogue turns."""
    if duration_s < min_s or segment_s <= 0:
        return []
    cuts = []
    start = 0.0
    while start + min_s <= duration_s + 1e-9:
        dur = min(segment_s, duration_s - start)
        if dur + 1e-6 < min_s:
            break
        cuts.append((round(start, 3), round(dur, 3)))
        start += segment_s
    return cuts


def take_hours(durations_s: list[float], max_hours: float) -> list[float]:
    """Keep a deterministic prefix whose sum stays within max_hours."""
    if max_hours <= 0:
        return []
    cap = max_hours * 3600.0
    kept = []
    total = 0.0
    for dur in durations_s:
        if dur <= 0:
            continue
        if total >= cap:
            break
        if total + dur > cap and total > 0:
            break
        kept.append(dur)
        total += dur
    return kept


def mls_remainder(durations_s: list[float], skip_hours: float = MLS_ALREADY_H, keep_hours: float = MLS_KEEP_H) -> list[float]:
    """Drop the first ``skip_hours`` (already in the mono store) and keep the next slice."""
    skipped = take_hours(durations_s, skip_hours)
    rest = durations_s[len(skipped):]
    return take_hours(rest, keep_hours)


def project_cost(audio_hours: float, rtf: float, hourly_cents: int, download_s: float = DOWNLOAD_ALLOWANCE_S) -> dict:
    """Cents for download allowance plus encode. rtf is audio-seconds per encode-second."""
    if rtf <= 0:
        raise ValueError("realtime factor must be positive")
    encode_s = audio_hours * 3600.0 / rtf
    total_s = encode_s + download_s
    cents = cost.cost_cents(int(math.ceil(total_s)), hourly_cents)
    return {
        "audio_hours": audio_hours,
        "rtf": rtf,
        "encode_s": encode_s,
        "download_s": download_s,
        "total_s": total_s,
        "cents": cents,
    }


def planned_hours() -> float:
    return VOX_CAP_H + MLS_KEEP_H


def over_cap(cents: int, cap_cents: int = MAX_USD_CENTS) -> bool:
    return int(cents) > int(cap_cents)


def reference_projection(hourly_cents: int = A40_HOURLY_CENTS) -> dict:
    """Dry-run figure from the H100 mono factor. The pod speed test replaces it."""
    proj = project_cost(planned_hours(), REFERENCE_RTF, hourly_cents)
    proj["measured"] = False
    proj["note"] = (
        f"encode rate {REFERENCE_RTF:.0f}x is the H100 mono measurement, not an A40 measurement"
    )
    return proj


def speed_report(audio_s: float, encode_s: float, hourly_cents: int, cap_cents: int = MAX_USD_CENTS) -> dict:
    if audio_s <= 0 or encode_s <= 0:
        raise ValueError("speed test needs positive audio and encode time")
    rtf = audio_s / encode_s
    proj = project_cost(planned_hours(), rtf, hourly_cents)
    proj["speed_audio_s"] = audio_s
    proj["speed_encode_s"] = encode_s
    proj["measured"] = True
    proj["over_cap"] = over_cap(proj["cents"], cap_cents)
    return proj


def proxy_missed(running_s: float, proxy_up: bool, limit_s: float = PROXY_WAIT_S) -> bool:
    """True when 8080 has not answered within the wait after the pod is RUNNING."""
    return (not proxy_up) and running_s >= limit_s


def attribution_text() -> str:
    return "\n".join(ATTRIBUTION[key] for key in ("voxpopuli_it_2015", "mls_it_remainder")) + "\n"


def explain_plan(cap_cents: int = MAX_USD_CENTS) -> str:
    ref = reference_projection()
    lines = [
        f"voxpopuli: {', '.join(VOX_TARS)} from {VOX_BASE}/",
        f"  CC0, 2015 events only, cap {VOX_CAP_H:.0f} h",
        f"mls: {MLS_URL}",
        f"  CC BY 4.0, skip the first {MLS_ALREADY_H:.0f} h already in the mono store, "
        f"keep up to {MLS_KEEP_H:.0f} h",
        f"segments: {SEGMENT_S:.0f} s from t=0, same cap as prepare_audio turns, width {WIDTH}",
        f"stores: {STORE_VOX}, {STORE_MLS}",
        f"tokens only on the way back (about 0.8 GB); raw audio is deleted on the pod",
        f"cap: ${cap_cents / 100:.2f} or {MAX_WALL_S / 3600:.0f} h, A40 ${A40_HOURLY_CENTS / 100:.2f}/h, "
        f"disk {DISK_GB} GB, terminateAfter set",
        f"reference projection: ${ref['cents'] / 100:.2f} "
        f"({ref['total_s'] / 3600:.2f} h). {ref['note']}.",
        "the pod encodes 5 min of the first tar members before the bulk download",
        "pull server listens on 8080 before that download; "
        "the launcher terminates if it is not up within 10 min of RUNNING",
    ]
    if over_cap(ref["cents"], cap_cents):
        lines.append("REFUSING: the reference projection is already over the cap.")
    return "\n".join(lines)
