"""The math. This is the whole point of the package, so it is deliberately
small, explicit, and heavily commented.

Everything is a pure function of a :class:`ModelConfig` and a :class:`Workload`.
No GPU, no model download, no side effects.
"""

from __future__ import annotations

from .models import (
    MemoryBreakdown,
    ModelConfig,
    Workload,
    dtype_bytes,
)

_GIB = 1024**3


def kv_bytes_per_token(model: ModelConfig, kv_dtype: str = "fp16") -> float:
    """Bytes of KV cache produced by a single token, across all layers.

    The formula:

        2  x  num_layers  x  num_kv_heads  x  head_dim  x  bytes_per_element

    The leading ``2`` is because we cache both Keys and Values. Note it uses
    ``num_kv_heads`` (not ``num_attention_heads``): a GQA model with 8 KV heads
    on 32 query heads caches 4x less than an MHA model of the same width.
    """
    b = dtype_bytes(kv_dtype)
    return (
        2
        * model.num_layers
        * model.num_kv_heads
        * model.effective_head_dim
        * b
    )


def cached_tokens(model: ModelConfig, context_length: int) -> int:
    """Number of tokens actually held in cache.

    For full attention this is the whole context. For sliding-window models the
    cache is capped at the window size, so a longer context costs no more.
    """
    if model.sliding_window is not None:
        return min(context_length, model.sliding_window)
    return context_length


def kv_cache_bytes(model: ModelConfig, workload: Workload) -> float:
    """Total KV cache bytes for a workload (context x batch)."""
    per_token = kv_bytes_per_token(model, workload.kv_dtype)
    tokens = cached_tokens(model, workload.context_length)
    return per_token * tokens * workload.batch_size


def weight_bytes(model: ModelConfig, weight_dtype: str = "fp16") -> float:
    """Approximate bytes to hold the model weights."""
    return model.num_params * dtype_bytes(weight_dtype)


def activation_bytes(model: ModelConfig, workload: Workload) -> float:
    """Rough estimate of peak activation memory during a decode step.

    Activations during autoregressive decode are small relative to weights and
    cache (you process one token per sequence at a time). We approximate peak
    activations as a few multiples of the hidden state for the active batch.
    This is intentionally conservative rather than exact.
    """
    b = dtype_bytes(workload.weight_dtype)
    # ~ a small constant number of hidden-sized buffers per active sequence.
    per_seq = 18 * model.hidden_size * b
    return per_seq * workload.batch_size


def framework_overhead_bytes(subtotal_bytes: float, fraction: float = 0.05) -> float:
    """Allocator slack, fragmentation, and framework bookkeeping.

    Modeled as a small fraction of everything else. Defaults to 5%.
    """
    return subtotal_bytes * fraction


def estimate_memory(
    model: ModelConfig,
    workload: Workload,
    *,
    overhead_fraction: float = 0.05,
) -> MemoryBreakdown:
    """Full memory breakdown for a model + workload.

    This is the primary entry point most callers want.
    """
    w = weight_bytes(model, workload.weight_dtype)
    kv = kv_cache_bytes(model, workload)
    act = activation_bytes(model, workload)
    overhead = framework_overhead_bytes(w + kv + act, overhead_fraction)
    return MemoryBreakdown(
        weights_bytes=w,
        kv_cache_bytes=kv,
        activation_bytes=act,
        framework_overhead_bytes=overhead,
    )


def kv_cache_gib(model: ModelConfig, workload: Workload) -> float:
    """Convenience: KV cache size in GiB."""
    return kv_cache_bytes(model, workload) / _GIB
