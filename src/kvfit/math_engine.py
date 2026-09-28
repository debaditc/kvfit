"""The math. This is the whole point of the package, so it is deliberately
small, explicit, and heavily commented.

Everything is a pure function of a :class:`ModelConfig` and a :class:`Workload`.
No GPU, no model download, no side effects.
"""

from __future__ import annotations

from .models import (
    CHUNKED,
    FULL,
    LINEAR,
    SLIDING,
    MemoryBreakdown,
    ModelConfig,
    Workload,
    dtype_bytes,
)

_GIB = 1024**3

# Activations are computed in 16-bit even when weights are quantized.
_ACT_BYTES = 2.0
# Logits are materialised in fp32 for sampling.
_LOGIT_BYTES = 4.0
_DEFAULT_VOCAB = 32000


def kv_elems_per_token_per_layer(model: ModelConfig) -> int:
    """Values cached per token in one attention layer.

    Standard attention caches a Key and a Value vector per KV head::

        num_kv_heads  x  (head_dim + v_head_dim)      # = 2 x kv_heads x head_dim

    It uses ``num_kv_heads`` (not ``num_attention_heads``): a GQA model with 8 KV
    heads on 32 query heads caches 4x less than an MHA model of the same width.

    Multi-head latent attention (MLA, DeepSeek-V3 and friends) instead caches a
    single compressed latent plus a small decoupled RoPE key, shared by all
    heads::

        kv_lora_rank + qk_rope_head_dim               # no factor of 2, no heads
    """
    if model.kv_lora_rank is not None:
        return model.kv_lora_rank + (model.qk_rope_head_dim or 0)
    k = model.effective_head_dim
    v = model.v_head_dim if model.v_head_dim is not None else k
    return model.num_kv_heads * (k + v)


def _attention_layers(model: ModelConfig) -> int:
    return sum(1 for k in model.layer_kinds if k in (FULL, SLIDING, CHUNKED))


def _indexer_layers(model: ModelConfig) -> int:
    if not model.indexer_head_dim:
        return 0
    if model.indexer_layers is not None:
        return model.indexer_layers
    return _attention_layers(model)


def kv_bytes_per_token(model: ModelConfig, kv_dtype: str = "fp16") -> float:
    """Bytes of KV cache produced by a single token, across all layers.

    For a plain full-attention model this is the classic formula::

        2  x  num_layers  x  num_kv_heads  x  head_dim  x  bytes_per_element

    Linear-attention / SSM layers add nothing per token (their state is fixed),
    and sliding-window layers stop growing once the window is full; see
    :func:`kv_cache_bytes` for the context-dependent total.
    """
    b = dtype_bytes(kv_dtype)
    per_layer = kv_elems_per_token_per_layer(model)
    elems = per_layer * _attention_layers(model)
    elems += (model.indexer_head_dim or 0) * _indexer_layers(model)
    return elems * b


def cached_tokens(model: ModelConfig, context_length: int) -> int:
    """Tokens held per layer for a model whose layers are all alike.

    For full attention this is the whole context. For sliding-window models the
    cache is capped at the window size, so a longer context costs no more.
    Hybrid models cache different amounts per layer; use :func:`kv_cache_bytes`.
    """
    if model.sliding_window is not None and model.layer_types is None:
        return min(context_length, model.sliding_window)
    return context_length


def _layer_tokens(model: ModelConfig, kind: str, context_length: int) -> int:
    if kind == FULL:
        return context_length
    if kind in (SLIDING, CHUNKED):
        assert model.sliding_window is not None
        return min(context_length, model.sliding_window)
    return 0


def _kv_parts(model: ModelConfig, workload: Workload) -> tuple[float, float, float]:
    """(attention KV, indexer keys, linear state) bytes for the whole model."""
    b = dtype_bytes(workload.kv_dtype)
    ctx = workload.context_length
    per_layer = kv_elems_per_token_per_layer(model)
    tokens = sum(_layer_tokens(model, kind, ctx) for kind in model.layer_kinds)
    attn = per_layer * tokens * b
    # Sparse-attention indexer keys live on full-attention layers.
    indexer = (model.indexer_head_dim or 0) * _indexer_layers(model) * ctx * b
    state = model.linear_state_bytes * model.layer_kinds.count(LINEAR)
    n = workload.batch_size
    return attn * n, indexer * n, state * n


def kv_cache_bytes(model: ModelConfig, workload: Workload) -> float:
    """Total KV cache (plus fixed recurrent state) for a workload, all devices.

    Summed layer by layer so hybrid models are counted correctly: full-attention
    layers cache the whole context, sliding / chunked layers at most the window,
    and linear-attention / SSM layers a fixed state per sequence.
    """
    return sum(_kv_parts(model, workload))


def _check_parallelism(model: ModelConfig, workload: Workload) -> None:
    if model.num_attention_heads % workload.tp:
        raise ValueError(
            f"tp={workload.tp} must divide the number of attention heads "
            f"({model.num_attention_heads}) of {model.name}"
        )
    if workload.pp > model.num_layers:
        raise ValueError(f"pp={workload.pp} exceeds the {model.num_layers} layers")


def kv_cache_bytes_per_device(model: ModelConfig, workload: Workload) -> float:
    """KV cache held by each GPU under tensor / pipeline parallelism.

    Pipeline stages split the layers. Tensor parallelism splits standard K/V by
    head, but each rank needs at least one KV head, so heads are *replicated*
    when ``num_kv_heads < tp`` (e.g. 4 KV heads on tp=8 -> each rank holds 1,
    i.e. 2x the ideal). MLA latents and indexer keys aren't per-head, so every
    tensor rank keeps a full copy of its stage's share.
    """
    attn, indexer, state = _kv_parts(model, workload)
    tp, pp = workload.tp, workload.pp
    if model.is_mla:
        attn_share = 1.0
    else:
        kv = model.num_kv_heads
        attn_share = -(-kv // tp) / kv  # ceil(kv / tp) heads of kv per rank
    return (attn * attn_share + indexer + state / tp) / pp


def weight_bytes(model: ModelConfig, weight_dtype: str = "fp16") -> float:
    """Approximate bytes to hold the model weights."""
    return model.num_params * dtype_bytes(weight_dtype)


def activation_bytes(model: ModelConfig, workload: Workload) -> float:
    """Rough estimate of peak activation memory.

    Two phases compete for the peak:

    * **decode** processes one token per sequence: a handful of hidden-sized
      buffers per active sequence;
    * **prefill** processes up to ``prefill_chunk`` prompt tokens at once, and
      each needs its hidden state plus the (much wider) MLP intermediate.

    On top of the larger of the two sits the fp32 logits buffer, one
    vocabulary-sized row per sequence. Modern vocabularies are 150k-260k
    entries, so this is no longer negligible at large batch sizes.
    This is intentionally conservative rather than exact.

    Under tensor parallelism the MLP intermediate and the logits are split
    across ranks; the hidden state is replicated. The result is per device.
    """
    h = model.hidden_size
    tp = workload.tp
    decode = 18 * h * _ACT_BYTES * workload.batch_size
    tokens = min(workload.prefill_chunk, workload.context_length * workload.batch_size)
    inter = model.intermediate_size or 4 * h
    prefill = tokens * (4 * h + 2 * inter / tp) * _ACT_BYTES
    vocab = model.vocab_size or _DEFAULT_VOCAB
    logits = workload.batch_size * vocab * _LOGIT_BYTES / tp
    return max(decode, prefill) + logits


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
    """Full memory breakdown for a model + workload, per device.

    This is the primary entry point most callers want. With ``workload.tp`` /
    ``workload.pp`` above 1 it returns what *each* GPU must hold.
    """
    _check_parallelism(model, workload)
    w = weight_bytes(model, workload.weight_dtype) / workload.num_devices
    kv = kv_cache_bytes_per_device(model, workload)
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
