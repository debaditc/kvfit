"""Decide whether a workload fits a device, and if not, why and what to change."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from .math_engine import estimate_memory
from .models import (
    CHUNKED,
    FULL,
    SLIDING,
    FitResult,
    GPUSpec,
    ModelConfig,
    Workload,
    dtype_bytes,
)

# Returned by :func:`max_context_for` when the cache stops growing (every
# attention layer is windowed), so no context length runs out of memory.
UNLIMITED_CONTEXT = 1_000_000_000

_MAX_CONTEXT_SEARCH = 1 << 24  # 16.7M tokens
_MAX_BATCH_SEARCH = 1 << 20


def _largest_fitting(hi: int, fits: Callable[[int], bool]) -> int:
    """Largest n in [1, hi] with fits(n), given fits is monotone. 0 if none."""
    if not fits(1):
        return 0
    if fits(hi):
        return hi
    lo = 1  # fits
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if fits(mid):
            lo = mid
        else:
            hi = mid
    return lo


def _cache_is_bounded(model: ModelConfig) -> bool:
    """True if no layer caches the full context (all windowed / linear)."""
    kinds = set(model.layer_kinds)
    return FULL not in kinds and bool(kinds & {SLIDING, CHUNKED}) and not model.indexer_head_dim


def max_context_for(
    model: ModelConfig,
    gpu: GPUSpec,
    *,
    batch_size: int = 1,
    kv_dtype: str = "fp16",
    weight_dtype: str = "fp16",
    prefill_chunk: int = 2048,
    overhead_fraction: float = 0.05,
    tp: int = 1,
    pp: int = 1,
) -> int:
    """Largest context length that fits, at a fixed batch size. 0 if none fits.

    Solved against the same :func:`estimate_memory` used by :func:`check_fit`,
    so ``check_fit`` at the returned context always fits. Returns
    :data:`UNLIMITED_CONTEXT` when the cache stops growing (all layers windowed).
    """
    usable = gpu.usable_bytes

    def fits(ctx: int) -> bool:
        wl = Workload(ctx, batch_size, kv_dtype, weight_dtype, prefill_chunk, tp, pp)
        return estimate_memory(model, wl, overhead_fraction=overhead_fraction).total_bytes <= usable

    best = _largest_fitting(_MAX_CONTEXT_SEARCH, fits)
    if best == _MAX_CONTEXT_SEARCH and _cache_is_bounded(model):
        return UNLIMITED_CONTEXT
    return best


def max_batch_for(
    model: ModelConfig,
    gpu: GPUSpec,
    *,
    context_length: int,
    kv_dtype: str = "fp16",
    weight_dtype: str = "fp16",
    prefill_chunk: int = 2048,
    overhead_fraction: float = 0.05,
    tp: int = 1,
    pp: int = 1,
) -> int:
    """Largest batch size that fits at a fixed context length. 0 if none fits."""
    usable = gpu.usable_bytes

    def fits(batch: int) -> bool:
        wl = Workload(context_length, batch, kv_dtype, weight_dtype, prefill_chunk, tp, pp)
        return estimate_memory(model, wl, overhead_fraction=overhead_fraction).total_bytes <= usable

    return _largest_fitting(_MAX_BATCH_SEARCH, fits)


# Tensor-parallel sizes tried, largest last; pipeline stages extend beyond one node.
_TP_SIZES: tuple[int, ...] = (1, 2, 4, 8)
_PP_SIZES: tuple[int, ...] = (1, 2, 4, 8)


def valid_tp_sizes(model: ModelConfig, sizes: tuple[int, ...] = _TP_SIZES) -> list[int]:
    """Tensor-parallel sizes that divide the model's attention heads."""
    return [t for t in sizes if model.num_attention_heads % t == 0]


def min_parallelism(
    model: ModelConfig,
    workload: Workload,
    gpu: GPUSpec,
    *,
    max_devices: int = 64,
    overhead_fraction: float = 0.05,
) -> tuple[int, int] | None:
    """Smallest (tp, pp) with tp*pp <= max_devices at which the workload fits.

    Prefers fewer GPUs, then more tensor parallelism (tp within a node is
    faster than extra pipeline stages). ``None`` if nothing fits.
    """
    options = sorted(
        ((t, p) for t in valid_tp_sizes(model) for p in _PP_SIZES
         if t * p <= max_devices and p <= model.num_layers),
        key=lambda tp_pp: (tp_pp[0] * tp_pp[1], tp_pp[1]),
    )
    usable = gpu.usable_bytes
    for t, p in options:
        wl = replace(workload, tp=t, pp=p)
        if estimate_memory(model, wl, overhead_fraction=overhead_fraction).total_bytes <= usable:
            return t, p
    return None


def _suggestions(
    model: ModelConfig,
    workload: Workload,
    gpu: GPUSpec,
    fits: bool,
) -> list[str]:
    tips: list[str] = []
    if fits:
        return tips

    mem = estimate_memory(model, workload)
    usable = gpu.usable_bytes

    # 0) Weights alone don't fit: no context / batch / KV setting can help.
    if mem.weights_bytes >= usable:
        gib = mem.weights_bytes / (1024**3)
        if gpu.is_gpu:
            where = (f"per GPU at tp={workload.tp} x pp={workload.pp}"
                     if workload.num_devices > 1 else "")
            found = min_parallelism(model, workload, gpu)
            if found is not None:
                t, p = found
                how = f"tp={t}" + (f" x pp={p}" if p > 1 else "")
                need = (f"It fits on {t * p} of these GPUs with {how} "
                        f"(--tp {t}" + (f" --pp {p}" if p > 1 else "") + ")")
            else:
                need = (f"Even {max(_TP_SIZES) * max(_PP_SIZES)} of these GPUs aren't enough; "
                        "use bigger GPUs")
            tips.append(
                f"The weights alone ({gib:,.1f} GiB{' ' + where if where else ''}) exceed "
                f"this {gpu.memory_label}. {need}, or use a much smaller weight format "
                "(e.g. weight_dtype=nvfp4 / awq)."
            )
        else:
            tips.append(
                f"The weights alone ({gib:,.1f} GiB) exceed this RAM. Use a smaller "
                "quantized build (e.g. weight_dtype=q4_k_m) or a smaller model."
            )
        if model.is_moe:
            tips.append(
                "This is a mixture-of-experts model: all experts must be in memory even "
                "though only a few run per token. Offloading experts to CPU RAM "
                "(llama.cpp --n-cpu-moe, ktransformers) trades speed for fit."
            )
        return tips

    # 1) Quantize the KV cache.
    if dtype_bytes(workload.kv_dtype) > 1.0:
        smaller = replace(workload, kv_dtype="int4")
        if estimate_memory(model, smaller).total_bytes <= usable:
            tips.append(
                "Quantize the KV cache to int4 (kv_dtype=int4) — this alone gets "
                "you under budget, at a small quality cost."
            )
        else:
            tips.append(
                "Quantize the KV cache (try fp8 or int4) to cut cache memory 2-4x."
            )

    # 2) Shorter context.
    mc = max_context_for(model, gpu, batch_size=workload.batch_size,
                         kv_dtype=workload.kv_dtype, weight_dtype=workload.weight_dtype,
                         prefill_chunk=workload.prefill_chunk,
                         tp=workload.tp, pp=workload.pp)
    if 0 < mc < workload.context_length:
        tips.append(
            f"Reduce max context to ~{mc:,} tokens at this batch size to fit."
        )

    # 3) Smaller batch.
    mb = max_batch_for(model, gpu, context_length=workload.context_length,
                       kv_dtype=workload.kv_dtype, weight_dtype=workload.weight_dtype,
                       prefill_chunk=workload.prefill_chunk,
                       tp=workload.tp, pp=workload.pp)
    if 0 < mb < workload.batch_size:
        tips.append(f"Reduce batch size to {mb} at this context length to fit.")

    # 4) Weight quantization if weights dominate.
    if mem.weights_bytes > gpu.usable_bytes * 0.6 and dtype_bytes(workload.weight_dtype) > 1.0:
        if gpu.is_gpu:
            tips.append(
                "Weights dominate memory here — load them in fp8 / int4 (e.g. "
                "weight_dtype=fp8, awq, nvfp4) or shard across GPUs with tensor parallelism."
            )
        else:
            tips.append(
                "Weights dominate memory here — run a quantized build (e.g. GGUF "
                "weight_dtype=q4_k_m or q8_0) to cut weight memory 2-3x."
            )

    # 5) MoE note: every expert must be resident even though few are active.
    if model.is_moe and mem.weights_bytes > gpu.usable_bytes * 0.6:
        tips.append(
            "This is a mixture-of-experts model: all experts must be in memory even "
            "though only a few run per token. Offloading experts to CPU RAM "
            "(llama.cpp --n-cpu-moe, ktransformers) trades speed for fit."
        )

    # 6) GQA note.
    if model.attention_kind == "MHA":
        tips.append(
            "This model uses full multi-head attention; a GQA model of similar "
            "size would use several times less KV cache."
        )

    if not tips:
        if gpu.is_gpu:
            found = min_parallelism(model, workload, gpu)
            if found is not None and found[0] * found[1] > workload.num_devices:
                t, p = found
                tips.append(f"Shard across {t * p} GPUs (--tp {t}"
                            + (f" --pp {p}" if p > 1 else "") + ").")
            else:
                tips.append("Use a larger GPU or shard the model across multiple GPUs.")
        else:
            tips.append(
                "Add more system RAM, or choose a smaller / more-quantized model."
            )
    return tips


def check_fit(
    model: ModelConfig,
    workload: Workload,
    gpu: GPUSpec,
    *,
    overhead_fraction: float = 0.05,
) -> FitResult:
    """Full fit verdict for a model + workload + GPU."""
    breakdown = estimate_memory(model, workload, overhead_fraction=overhead_fraction)
    usable = gpu.usable_bytes
    headroom = usable - breakdown.total_bytes
    fits = headroom >= 0

    return FitResult(
        model=model,
        workload=workload,
        gpu=gpu,
        breakdown=breakdown,
        fits=fits,
        headroom_bytes=headroom,
        max_context=max_context_for(
            model, gpu, batch_size=workload.batch_size,
            kv_dtype=workload.kv_dtype, weight_dtype=workload.weight_dtype,
            prefill_chunk=workload.prefill_chunk, overhead_fraction=overhead_fraction,
            tp=workload.tp, pp=workload.pp),
        max_batch=max_batch_for(
            model, gpu, context_length=workload.context_length,
            kv_dtype=workload.kv_dtype, weight_dtype=workload.weight_dtype,
            prefill_chunk=workload.prefill_chunk, overhead_fraction=overhead_fraction,
            tp=workload.tp, pp=workload.pp),
        suggestions=_suggestions(model, workload, gpu, fits),
    )
