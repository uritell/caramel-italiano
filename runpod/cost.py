"""Dollar and wall-clock caps for the one-GPU retry.

Integer cents, rounded up, so the launcher and the in-pod watchdog stop on
the same second. One H100 at $4.39/h for 2.5 h is $10.98, so the clock hits
before the $12 cap. A higher live price makes the $12 cap hit first.
"""
from __future__ import annotations

import math

# 2.5 hours. Billing is treated as starting when the create call returns.
MAX_WALL_S = 9000
# Hard stop for this retry. The account had about $12.36 left after the
# failed 2-GPU pod; $12 leaves a few cents of margin under that.
MAX_USD_CENTS = 1200
USER_BUDGET_CENTS = 2500
GPU_COUNT = 1
# Secure Cloud H100 SXM, per GPU-hour, checked 27 Nov 2026.
QUOTED_CENTS_PER_GPU_HOUR = 439
# Refuse to launch if the catalog price is above this. One GPU at $4.90
# for 2.5 h is $12.25, just over this retry's $12 cap.
MAX_ACCEPT_CENTS_PER_GPU_HOUR = 490
CONTAINER_DISK_GB = 160
PULL_GRACE_S = 600

QUOTED_HOURLY_CENTS = QUOTED_CENTS_PER_GPU_HOUR * GPU_COUNT  # 439 = $4.39/h


def dollars_to_cents_up(amount: float) -> int:
    """Smallest whole cent that is not below `amount`."""
    return int(math.ceil(float(amount) * 100 - 1e-9))


def hourly_cents(cents_per_gpu_hour: int, gpus: int = GPU_COUNT) -> int:
    if cents_per_gpu_hour <= 0 or gpus <= 0:
        raise ValueError("price and GPU count must be positive")
    return cents_per_gpu_hour * gpus


def cost_cents(elapsed_s: int, hourly: int) -> int:
    """Ceil to the next cent so the cap arrives slightly early, not late.

    Same expression as the shell watchdog:
    (elapsed * hourly_cents + 3599) / 3600
    """
    if elapsed_s <= 0 or hourly <= 0:
        return 0
    return (int(elapsed_s) * int(hourly) + 3599) // 3600


def stop_reason(
    elapsed_s: int,
    hourly: int,
    max_wall_s: int = MAX_WALL_S,
    max_usd_cents: int = MAX_USD_CENTS,
) -> str | None:
    """Return 'wall', 'dollars', or None. Wall is checked first."""
    if elapsed_s >= max_wall_s:
        return "wall"
    if cost_cents(elapsed_s, hourly) >= max_usd_cents:
        return "dollars"
    return None


def first_stop(hourly: int, search_s: int = MAX_WALL_S + 7200) -> tuple[int, str]:
    for t in range(0, search_s + 1):
        reason = stop_reason(t, hourly)
        if reason:
            return t, reason
    raise RuntimeError("cap never fired")


def format_usd(cents: int) -> str:
    sign = "-" if cents < 0 else ""
    cents = abs(int(cents))
    return f"{sign}${cents // 100}.{cents % 100:02d}"


def projected_wall_cents(hourly: int) -> int:
    return cost_cents(MAX_WALL_S, hourly)


def price_ok(cents_per_gpu_hour: int) -> str | None:
    """None when the catalog price is acceptable, otherwise a reason string."""
    if cents_per_gpu_hour <= 0:
        return "catalog price is missing"
    if cents_per_gpu_hour > MAX_ACCEPT_CENTS_PER_GPU_HOUR:
        return (
            f"Secure Cloud price {format_usd(cents_per_gpu_hour)}/GPU-h is above "
            f"the accepted ceiling {format_usd(MAX_ACCEPT_CENTS_PER_GPU_HOUR)}/GPU-h "
            f"(quote was {format_usd(QUOTED_CENTS_PER_GPU_HOUR)}/GPU-h)"
        )
    wall = projected_wall_cents(hourly_cents(cents_per_gpu_hour))
    if wall > USER_BUDGET_CENTS:
        return (
            f"2.5 h at this price is {format_usd(wall)}, above the "
            f"{format_usd(USER_BUDGET_CENTS)} approval"
        )
    return None
