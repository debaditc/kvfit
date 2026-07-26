"""What-if sweeps: how does memory change as you vary the KV cache precision?"""

from __future__ import annotations

from dataclasses import replace

from .math_engine import estimate_memory
from .models import (
    KV_DTYPE_SWEEP,
    GPUSpec,
    ModelConfig,
    SweepRow,
    Workload,
)


def sweep_kv_dtype(
    model: ModelConfig,
    workload: Workload,
    gpu: GPUSpec | None = None,
    *,
    dtypes: tuple[str, ...] = KV_DTYPE_SWEEP,
) -> list[SweepRow]:
    """Vary KV cache dtype and report cache + total memory (and fit, if a GPU
    is given)."""
    rows: list[SweepRow] = []
    usable = gpu.usable_bytes if gpu is not None else float("inf")
    for dt in dtypes:
        wl = replace(workload, kv_dtype=dt)
        mem = estimate_memory(model, wl)
        rows.append(
            SweepRow(
                kv_dtype=dt,
                kv_cache_bytes=mem.kv_cache_bytes,
                total_bytes=mem.total_bytes,
                fits=mem.total_bytes <= usable,
            )
        )
    return rows
