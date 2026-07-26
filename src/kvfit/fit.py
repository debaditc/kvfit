"""Decide whether a workload fits a GPU, and if not, why and what to change."""

from __future__ import annotations

from dataclasses import replace

from .math_engine import (
    estimate_memory,
    kv_bytes_per_token,
    weight_bytes,
)
from .models import (
    FitResult,
    GPUSpec,
    ModelConfig,
    Workload,
    dtype_bytes,
)

_GIB = 1024**3


def max_context_for(
    model: ModelConfig,
    gpu: GPUSpec,
    *,
    batch_size: int = 1,
    kv_dtype: str = "fp16",
    weight_dtype: str = "fp16",
) -> int:
    """Largest context length that fits, at a fixed batch size. 0 if none fits."""
    usable = gpu.usable_bytes
    w = weight_bytes(model, weight_dtype)
    if w >= usable:
        return 0
    # Solve budget = weights + per_token*tokens*batch + activations + overhead.
    # activations + overhead are small and context-independent-ish; fold a 5%
    # overhead in and ignore activations for the closed-form bound, then it is a
    # conservative estimate that we don't over-promise on.
    budget = usable / 1.05 - w
    per_token = kv_bytes_per_token(model, kv_dtype) * batch_size
    if per_token <= 0:
        return 0
    tokens = int(budget // per_token)
    # For sliding-window models, beyond the window more context is free, so
    # there is effectively no cache-driven cap; report a large practical ceiling.
    if model.sliding_window is not None and tokens >= model.sliding_window:
        return 1_000_000
    return max(0, tokens)


def max_batch_for(
    model: ModelConfig,
    gpu: GPUSpec,
    *,
    context_length: int,
    kv_dtype: str = "fp16",
    weight_dtype: str = "fp16",
) -> int:
    """Largest batch size that fits at a fixed context length. 0 if none fits."""
    single = replace(
        Workload(context_length=context_length, batch_size=1,
                 kv_dtype=kv_dtype, weight_dtype=weight_dtype),
    )
    mem1 = estimate_memory(model, single)
    usable = gpu.usable_bytes
    # weights are shared across the batch; only kv + activations scale.
    fixed = mem1.weights_bytes
    per_seq = mem1.kv_cache_bytes + mem1.activation_bytes
    per_seq *= 1.05  # fold in overhead
    if per_seq <= 0 or fixed >= usable:
        return 0
    return max(0, int((usable - fixed) // per_seq))


def _suggestions(
    model: ModelConfig,
    workload: Workload,
    gpu: GPUSpec,
    fits: bool,
) -> list[str]:
    tips: list[str] = []
    if fits:
        return tips

    # 1) Quantize the KV cache.
    if dtype_bytes(workload.kv_dtype) > 1.0:
        smaller = replace(workload, kv_dtype="int4")
        if estimate_memory(model, smaller).total_bytes <= gpu.usable_bytes:
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
                         kv_dtype=workload.kv_dtype, weight_dtype=workload.weight_dtype)
    if 0 < mc < workload.context_length:
        tips.append(
            f"Reduce max context to ~{mc:,} tokens at this batch size to fit."
        )

    # 3) Smaller batch.
    mb = max_batch_for(model, gpu, context_length=workload.context_length,
                       kv_dtype=workload.kv_dtype, weight_dtype=workload.weight_dtype)
    if 0 < mb < workload.batch_size:
        tips.append(f"Reduce batch size to {mb} at this context length to fit.")

    # 4) Weight quantization if weights dominate.
    mem = estimate_memory(model, workload)
    if mem.weights_bytes > gpu.usable_bytes * 0.6 and dtype_bytes(workload.weight_dtype) > 1.0:
        if gpu.is_gpu:
            tips.append(
                "Weights dominate memory here — load them in int8/int4 (weight_dtype) "
                "or shard across GPUs with tensor parallelism."
            )
        else:
            tips.append(
                "Weights dominate memory here — run a quantized build (int4/int8, "
                "e.g. a GGUF Q4 model) via weight_dtype to cut weight memory 2-4x."
            )

    # 5) GQA note.
    if model.attention_kind == "MHA":
        tips.append(
            "This model uses full multi-head attention; a GQA model of similar "
            "size would use several times less KV cache."
        )

    if not tips:
        if gpu.is_gpu:
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
            kv_dtype=workload.kv_dtype, weight_dtype=workload.weight_dtype),
        max_batch=max_batch_for(
            model, gpu, context_length=workload.context_length,
            kv_dtype=workload.kv_dtype, weight_dtype=workload.weight_dtype),
        suggestions=_suggestions(model, workload, gpu, fits),
    )
