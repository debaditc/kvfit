"""Very rough cloud GPU cost estimates.

These are order-of-magnitude on-demand hourly rates and will drift. They exist so
the report can say "roughly $X/hr" rather than leaving cost as an exercise for
the reader. Always confirm against your provider's current pricing.
"""

from __future__ import annotations

# Approximate on-demand USD/hour, single GPU. Deliberately conservative.
_HOURLY_USD: dict[str, float] = {
    "h200": 4.50,
    "h100": 3.00,
    "h100-80gb": 3.00,
    "a100-80gb": 1.80,
    "a100": 1.20,
    "a100-40gb": 1.20,
    "l40s": 1.10,
    "l4": 0.75,
    "a10g": 0.75,
    "v100": 0.60,
    "t4": 0.35,
    "rtx-4090": 0.50,
    "rtx-3090": 0.35,
}


def hourly_cost(gpu_name: str, num_gpus: int = 1) -> float | None:
    """Estimated USD/hour for ``num_gpus`` of a GPU, or ``None`` if unknown."""
    rate = _HOURLY_USD.get(gpu_name.strip().lower())
    if rate is None:
        return None
    return rate * max(1, num_gpus)


def monthly_cost(gpu_name: str, num_gpus: int = 1, hours: float = 730.0) -> float | None:
    """Estimated USD/month assuming continuous use (default 730 hrs)."""
    hr = hourly_cost(gpu_name, num_gpus)
    return None if hr is None else hr * hours
