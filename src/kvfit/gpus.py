"""A small registry of common GPUs, keyed by friendly aliases.

Memory figures are the nominal on-board VRAM. Actual usable memory is computed
in :class:`~kvfit.models.GPUSpec` after subtracting a CUDA-context reserve and
applying a usable fraction.
"""

from __future__ import annotations

from .models import GPUSpec

# name -> total VRAM in GiB
_GPU_MEMORY: dict[str, float] = {
    # Data-center
    "h200": 141,
    "h100": 80,
    "h100-80gb": 80,
    "a100-80gb": 80,
    "a100": 40,
    "a100-40gb": 40,
    "l40s": 48,
    "l40": 48,
    "l4": 24,
    "a10g": 24,
    "a10": 24,
    "v100": 16,
    "v100-16gb": 16,
    "v100-32gb": 32,
    "t4": 16,
    # Prosumer / desktop
    "rtx-5090": 32,
    "rtx-4090": 24,
    "rtx-4080": 16,
    "rtx-3090": 24,
    "rtx-3080": 10,
    "rtx-a6000": 48,
    # Apple unified memory (approximate usable share)
    "m3-max-64gb": 48,
    "m2-ultra-192gb": 147,
}


def list_gpus() -> list[str]:
    """Sorted list of known GPU aliases."""
    return sorted(_GPU_MEMORY)


def resolve_gpu(name: str) -> GPUSpec:
    """Resolve a GPU by alias, or by ``"<name>:<gib>"`` for a custom size.

    Examples:
        ``resolve_gpu("a100-40gb")``
        ``resolve_gpu("my-card:48")``  -> a 48 GiB custom GPU
    """
    raw = name.strip().lower()
    if ":" in raw:
        label, _, gib = raw.partition(":")
        try:
            return GPUSpec(name=label or "custom", memory_gib=float(gib))
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Could not parse custom GPU size in {name!r}") from exc

    if raw not in _GPU_MEMORY:
        hint = ", ".join(list_gpus()[:8])
        raise ValueError(
            f"Unknown GPU {name!r}. Try one of: {hint}, ... "
            f"or pass a custom size like 'mycard:48'."
        )
    return GPUSpec(name=raw, memory_gib=_GPU_MEMORY[raw])
