"""kvfit — will your LLM fit? A tiny, dependency-free KV cache & memory planner.

Answer "will this model + context + batch fit on this GPU?" before you spend the
money finding out. Pure math, no GPU or model download required.

Quick start::

    import kvfit

    fit = kvfit.check("llama-3-8b", gpu="a100-40gb", context=8192, batch=32)
    print(fit.fits, fit.headroom_gib)
    print(kvfit.report_text("llama-3-8b", gpu="a100-40gb", context=8192, batch=32))
"""

from __future__ import annotations

from .cost import hourly_cost, monthly_cost
from .cpu import detect_cpu_ram_gib, list_cpu_presets, resolve_cpu
from .fit import check_fit, max_batch_for, max_context_for
from .gpus import list_gpus, resolve_gpu
from .math_engine import (
    activation_bytes,
    estimate_memory,
    kv_bytes_per_token,
    kv_cache_bytes,
    kv_cache_gib,
    weight_bytes,
)
from .models import (
    DeviceSpec,
    FitResult,
    GPUSpec,
    MemoryBreakdown,
    ModelConfig,
    SweepRow,
    Workload,
)
from .report import render_fit, render_sweep, report
from .resolver import list_models, resolve_model
from .sweep import sweep_kv_dtype

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # high-level helpers
    "check",
    "report_text",
    "sweep",
    # resolution
    "resolve_model",
    "resolve_gpu",
    "resolve_cpu",
    "list_models",
    "list_gpus",
    "list_cpu_presets",
    "detect_cpu_ram_gib",
    # core math
    "estimate_memory",
    "kv_cache_bytes",
    "kv_cache_gib",
    "kv_bytes_per_token",
    "weight_bytes",
    "activation_bytes",
    # fit
    "check_fit",
    "max_context_for",
    "max_batch_for",
    # cost
    "hourly_cost",
    "monthly_cost",
    # reporting
    "report",
    "render_fit",
    "render_sweep",
    # data types
    "ModelConfig",
    "DeviceSpec",
    "GPUSpec",
    "Workload",
    "MemoryBreakdown",
    "FitResult",
    "SweepRow",
    "sweep_kv_dtype",
]


def _device(
    gpu: str | DeviceSpec | None,
    cpu: str | float | None,
) -> DeviceSpec:
    """Resolve exactly one of ``gpu`` / ``cpu`` into a DeviceSpec."""
    if (gpu is None) == (cpu is None):
        raise ValueError("Provide exactly one of gpu=... or cpu=...")
    if cpu is not None:
        return resolve_cpu(cpu)
    return gpu if isinstance(gpu, DeviceSpec) else resolve_gpu(gpu)  # type: ignore[arg-type]


def check(
    model: str | ModelConfig,
    *,
    gpu: str | DeviceSpec | None = None,
    cpu: str | float | None = None,
    context: int,
    batch: int = 1,
    kv_dtype: str = "fp16",
    weight_dtype: str = "fp16",
) -> FitResult:
    """One-call fit check against a GPU or CPU/RAM target.

    Examples::

        kvfit.check("mistral-7b", gpu="rtx-4090", context=32768)
        kvfit.check("llama-3-8b", cpu=32, context=8192, weight_dtype="int4")
        kvfit.check("phi-3-mini", cpu="auto", context=4096)
    """
    m = model if isinstance(model, ModelConfig) else resolve_model(model)
    g = _device(gpu, cpu)
    wl = Workload(context_length=context, batch_size=batch,
                  kv_dtype=kv_dtype, weight_dtype=weight_dtype)
    return check_fit(m, wl, g)


def report_text(
    model: str | ModelConfig,
    *,
    gpu: str | DeviceSpec | None = None,
    cpu: str | float | None = None,
    context: int,
    batch: int = 1,
    kv_dtype: str = "fp16",
    weight_dtype: str = "fp16",
    show_sweep: bool = True,
    color: bool | None = None,
) -> str:
    """Return the full formatted report as a string (GPU or CPU target)."""
    m = model if isinstance(model, ModelConfig) else resolve_model(model)
    g = _device(gpu, cpu)
    wl = Workload(context_length=context, batch_size=batch,
                  kv_dtype=kv_dtype, weight_dtype=weight_dtype)
    return report(m, wl, g, show_sweep=show_sweep, color=color)


def sweep(
    model: str | ModelConfig,
    *,
    context: int,
    batch: int = 1,
    gpu: str | DeviceSpec | None = None,
    cpu: str | float | None = None,
) -> list[SweepRow]:
    """What-if sweep of KV cache dtype for a model + workload.

    A GPU or CPU target is optional; if given, a fit column is included.
    """
    m = model if isinstance(model, ModelConfig) else resolve_model(model)
    g: DeviceSpec | None = None
    if gpu is not None or cpu is not None:
        g = _device(gpu, cpu)
    wl = Workload(context_length=context, batch_size=batch)
    return sweep_kv_dtype(m, wl, g)
