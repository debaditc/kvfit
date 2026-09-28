"""A small registry of common GPUs, keyed by friendly aliases.

Memory figures are the nominal on-board VRAM. Actual usable memory is computed
in :class:`~kvfit.models.GPUSpec` after subtracting a CUDA-context reserve and
applying a usable fraction. Bandwidth and dense bf16 TFLOPS feed the rough
speed / cost-per-token estimate only.
"""

from __future__ import annotations

from typing import NamedTuple

from .models import GPUSpec


class _GPU(NamedTuple):
    memory_gib: float  # nominal; reserve + usable fraction absorb the GB/GiB gap
    bandwidth_gbs: float  # peak memory bandwidth
    tflops: float  # dense bf16 / fp16 tensor throughput (no sparsity)


_GPUS: dict[str, _GPU] = {
    # NVIDIA data-center — Blackwell
    "gb300": _GPU(288, 8000, 2500),
    "b300": _GPU(288, 8000, 2250),
    "gb200": _GPU(186, 8000, 2500),
    "b200": _GPU(180, 8000, 2250),
    # NVIDIA data-center — Hopper
    "h200": _GPU(141, 4800, 989),
    "h100-nvl": _GPU(94, 3900, 835),
    "h100": _GPU(80, 3350, 989),
    "h20": _GPU(96, 4000, 148),
    # NVIDIA data-center — Ampere / Ada / older
    "a100-80gb": _GPU(80, 2039, 312),
    "a100-40gb": _GPU(40, 1555, 312),
    "l40s": _GPU(48, 864, 362),
    "l40": _GPU(48, 864, 181),
    "l4": _GPU(24, 300, 121),
    "a10g": _GPU(24, 600, 70),
    "a10": _GPU(24, 600, 125),
    "v100-16gb": _GPU(16, 900, 125),
    "v100-32gb": _GPU(32, 900, 125),
    "t4": _GPU(16, 320, 65),
    # AMD Instinct
    "mi355x": _GPU(288, 8000, 2500),
    "mi325x": _GPU(256, 6000, 1307),
    "mi300x": _GPU(192, 5300, 1307),
    # Workstation / prosumer
    "rtx-pro-6000": _GPU(96, 1792, 250),
    "rtx-a6000": _GPU(48, 768, 155),
    "rtx-6000-ada": _GPU(48, 960, 364),
    "rtx-5090": _GPU(32, 1792, 209),
    "rtx-5080": _GPU(16, 960, 113),
    "rtx-5070-ti": _GPU(16, 896, 88),
    "rtx-5060-ti-16gb": _GPU(16, 448, 47),
    "rtx-4090": _GPU(24, 1008, 165),
    "rtx-4080": _GPU(16, 717, 97),
    "rtx-3090": _GPU(24, 936, 71),
    "rtx-3080": _GPU(10, 760, 60),
    # Unified-memory boxes (approximate share the GPU can use)
    "dgx-spark": _GPU(119, 273, 125),  # 128 GB LPDDR5X
    "ryzen-ai-max-395-128gb": _GPU(96, 256, 59),  # Strix Halo, max VRAM carve-out
    # Apple unified memory (~75% default GPU wired limit)
    "m3-max-64gb": _GPU(48, 400, 28),
    "m4-max-128gb": _GPU(96, 546, 34),
    "m2-ultra-192gb": _GPU(147, 800, 54),
    "m3-ultra-512gb": _GPU(384, 819, 57),
}

# Extra spellings for the entries above.
_ALIASES: dict[str, str] = {
    "h100-80gb": "h100",
    "h100-sxm": "h100",
    "a100": "a100-40gb",
    "v100": "v100-16gb",
}


def list_gpus() -> list[str]:
    """Sorted list of known GPU aliases."""
    return sorted([*_GPUS, *_ALIASES])


def canonical_gpus() -> list[str]:
    """One name per distinct GPU (no alias duplicates), largest memory first."""
    return sorted(_GPUS, key=lambda n: (-_GPUS[n].memory_gib, n))


def _spec(name: str, memory_gib: float | None = None) -> GPUSpec:
    info = _GPUS[_ALIASES.get(name, name)]
    return GPUSpec(
        name=name,
        memory_gib=info.memory_gib if memory_gib is None else memory_gib,
        bandwidth_gbs=info.bandwidth_gbs,
        tflops=info.tflops,
    )


def resolve_gpu(name: str) -> GPUSpec:
    """Resolve a GPU by alias, or by ``"<name>:<gib>"`` for a custom size.

    Examples:
        ``resolve_gpu("a100-40gb")``
        ``resolve_gpu("my-card:48")``  -> a 48 GiB custom GPU
        ``resolve_gpu("h100:94")``     -> a known GPU's speed specs, 94 GiB
    """
    raw = name.strip().lower()
    if ":" in raw:
        label, _, gib = raw.partition(":")
        try:
            size = float(gib)
        except ValueError as exc:
            raise ValueError(f"Could not parse custom GPU size in {name!r}") from exc
        if label in _GPUS or label in _ALIASES:
            return _spec(label, size)
        return GPUSpec(name=label or "custom", memory_gib=size)

    if raw not in _GPUS and raw not in _ALIASES:
        hint = ", ".join(list_gpus()[:8])
        raise ValueError(
            f"Unknown GPU {name!r}. Try one of: {hint}, ... "
            f"or pass a custom size like 'mycard:48'."
        )
    return _spec(raw)
