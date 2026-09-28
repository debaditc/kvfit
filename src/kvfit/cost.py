"""Very rough cloud GPU cost estimates.

These are order-of-magnitude on-demand hourly rates and will drift. They exist so
the report can say "roughly $X/hr" rather than leaving cost as an exercise for
the reader. Always confirm against your provider's current pricing.
"""

from __future__ import annotations

# Approximate on-demand USD/hour, single GPU: rough medians across GPU clouds
# (neoclouds are often cheaper, hyperscalers 1.5-3x more). The fastest-drifting
# numbers in this package: pass ``price_per_hour`` with what you actually pay.
PRICES_AS_OF = "2026-09"

_HOURLY_USD: dict[str, float] = {
    "b200": 6.50,
    "h200": 3.80,
    "h100": 3.20,
    "h100-80gb": 3.20,
    "mi300x": 2.90,
    "a100-80gb": 2.00,
    "a100": 1.30,
    "a100-40gb": 1.30,
    "l40s": 1.50,
    "l4": 0.70,
    "a10g": 0.75,
    "v100": 0.50,
    "t4": 0.35,
    "rtx-a6000": 0.50,
    "rtx-5090": 0.75,
    "rtx-4090": 0.45,
    "rtx-3090": 0.25,
}


def hourly_cost(
    gpu_name: str, num_gpus: int = 1, *, price_per_hour: float | None = None,
) -> float | None:
    """Estimated USD/hour for ``num_gpus`` of a GPU, or ``None`` if unknown.

    ``price_per_hour`` (per GPU) overrides the built-in table.
    """
    rate = price_per_hour if price_per_hour is not None else _HOURLY_USD.get(
        gpu_name.strip().lower())
    if rate is None:
        return None
    return rate * max(1, num_gpus)


def monthly_cost(gpu_name: str, num_gpus: int = 1, hours: float = 730.0) -> float | None:
    """Estimated USD/month assuming continuous use (default 730 hrs)."""
    hr = hourly_cost(gpu_name, num_gpus)
    return None if hr is None else hr * hours
