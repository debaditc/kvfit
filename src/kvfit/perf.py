"""Rough speed and cost-per-token estimates from a roofline model.

Autoregressive decode is almost always limited by memory bandwidth: every step
streams the (active) weights and the whole KV cache through the chip to emit
one token per sequence. Prefill is limited by compute. So, per step::

    time = max(bytes_read / bandwidth, flops / tflops)

with realistic efficiency factors applied to the peak numbers, plus a fixed
per-layer latency (kernel launches, MoE routing, tensor-parallel all-reduces)
that dominates at small batch sizes. Treat the result
as an *order-of-magnitude* planning figure, not a benchmark: real engines add
kernel launch, scheduling and communication costs, and speculative decoding or
fp8 tensor cores can beat it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .cost import hourly_cost
from .math_engine import kv_cache_bytes
from .models import CHUNKED, FULL, SLIDING, DeviceSpec, ModelConfig, Workload, dtype_bytes

# Fraction of peak that decode / prefill typically achieve in practice.
DEFAULT_MBU = 0.8  # model bandwidth utilisation
DEFAULT_MFU = 0.5  # model FLOPs utilisation
# Tensor-parallel scaling efficiency (all-reduce per layer is not free).
_TP_EFFICIENCY = 0.9
# Fixed latency per layer per decode step, on top of the roofline. At small
# batch these dominate for MoE and multi-GPU runs, where little data moves:
_LAYER_OVERHEAD_S = 20e-6  # kernel launches / norms / small ops
_MOE_LAYER_OVERHEAD_S = 30e-6  # routing, token dispatch, grouped GEMM setup
_TP_LAYER_OVERHEAD_S = 50e-6  # two all-reduces per layer across tensor ranks
# Assumed when a CPU target has no known specs (dual-channel DDR5 desktop).
_CPU_DEFAULT_BW = 60.0
_CPU_DEFAULT_TFLOPS = 1.0


@dataclass(frozen=True)
class PerfEstimate:
    """Roofline estimate for one model + workload + device (+ parallelism)."""

    decode_tok_s_per_seq: float
    decode_tok_s_total: float
    decode_bound: str  # "memory" or "compute"
    prefill_seconds: float  # time to prefill one full-context prompt
    usd_per_hour: float | None
    usd_per_million_output: float | None
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "decode_tok_s_per_seq": round(self.decode_tok_s_per_seq, 1),
            "decode_tok_s_total": round(self.decode_tok_s_total, 1),
            "decode_bound": self.decode_bound,
            "prefill_seconds": round(self.prefill_seconds, 3),
            "usd_per_hour": self.usd_per_hour,
            "usd_per_million_output": (None if self.usd_per_million_output is None
                                       else round(self.usd_per_million_output, 4)),
            "notes": list(self.notes),
        }


def _weight_bytes_read(model: ModelConfig, workload: Workload) -> float:
    """Weight bytes streamed per decode step for the whole batch.

    Dense models read everything. MoE models read the shared part plus only the
    experts the batch's tokens are routed to: with k of E experts per token, a
    batch of B tokens touches about ``1 - (1 - k/E)^B`` of them.
    """
    wb = dtype_bytes(workload.weight_dtype)
    n = model.num_params
    a, e, k = model.num_active_params, model.num_experts, model.experts_per_token
    if not model.is_moe or a is None or not e or not k or k >= e:
        return n * wb
    expert_total = (n - a) / (1 - k / e)
    shared = max(0.0, n - expert_total)
    touched = 1 - (1 - k / e) ** workload.batch_size
    return (shared + expert_total * touched) * wb


def _attn_dim(model: ModelConfig) -> int:
    if model.is_mla:
        return 2 * (model.kv_lora_rank or 0) + (model.qk_rope_head_dim or 0)
    return 2 * model.effective_head_dim  # QK^T and AV


def _attended(model: ModelConfig, context: int, *, prefill: bool) -> float:
    """Sum over layers of tokens attended: per step (decode) or in total (prefill)."""
    total = 0.0
    w = model.sliding_window or context
    for kind in model.layer_kinds:
        if kind == FULL:
            total += context * context / 2 if prefill else context
        elif kind in (SLIDING, CHUNKED):
            span = min(context, w)
            total += context * span if prefill else span
    return total


def estimate_performance(
    model: ModelConfig,
    workload: Workload,
    device: DeviceSpec,
    *,
    price_per_hour: float | None = None,
    mbu: float = DEFAULT_MBU,
    mfu: float = DEFAULT_MFU,
) -> PerfEstimate | None:
    """Estimate decode speed, prefill time and $/1M output tokens.

    Returns ``None`` for a GPU whose bandwidth isn't known (e.g. a custom
    ``name:GiB`` target). ``price_per_hour`` is per device and overrides the
    built-in rate table.
    """
    notes: list[str] = []
    bw, tf = device.bandwidth_gbs, device.tflops
    if bw is None or tf is None:
        if device.is_gpu:
            return None
        bw, tf = bw or _CPU_DEFAULT_BW, tf or _CPU_DEFAULT_TFLOPS
        notes.append(f"Assumes ~{bw:.0f} GB/s RAM bandwidth; use a preset for your machine.")

    tp, n_dev = workload.tp, workload.num_devices
    scale = tp * (_TP_EFFICIENCY if tp > 1 else 1.0)
    bw_eff = bw * 1e9 * mbu * scale
    flops_eff = tf * 1e12 * mfu * scale
    active = model.num_active_params or model.num_params
    heads = model.num_attention_heads
    ctx, batch = workload.context_length, workload.batch_size

    # Decode at full context (worst case: the cache is as large as it gets).
    step_bytes = _weight_bytes_read(model, workload) + kv_cache_bytes(model, workload)
    step_flops = batch * (2 * active + 2 * heads * _attn_dim(model)
                          * _attended(model, ctx, prefill=False))
    t_mem, t_comp = step_bytes / bw_eff, step_flops / flops_eff
    per_layer = (_LAYER_OVERHEAD_S + (_MOE_LAYER_OVERHEAD_S if model.is_moe else 0.0)
                 + (_TP_LAYER_OVERHEAD_S if tp > 1 else 0.0))
    step = max(t_mem, t_comp) + per_layer * model.num_layers
    per_seq = 1.0 / step
    total = batch / step
    if workload.pp > 1:
        notes.append("Pipeline stages add capacity, not per-sequence speed.")

    # Prefill one full-context prompt: compute-bound, but weights are re-read
    # once per prefill chunk, which dominates on low-FLOP devices.
    pf_flops = 2 * active * ctx + 2 * heads * _attn_dim(model) * _attended(model, ctx,
                                                                         prefill=True)
    chunks = math.ceil(ctx / workload.prefill_chunk)
    pf_bytes = chunks * model.num_params * dtype_bytes(workload.weight_dtype)
    prefill = max(pf_flops / flops_eff, pf_bytes / bw_eff)

    price = hourly_cost(device.name, n_dev, price_per_hour=price_per_hour)
    per_m = None if price is None else price / (total * 3600) * 1e6
    return PerfEstimate(
        decode_tok_s_per_seq=per_seq,
        decode_tok_s_total=total,
        decode_bound="memory" if t_mem >= t_comp else "compute",
        prefill_seconds=prefill,
        usd_per_hour=price,
        usd_per_million_output=per_m,
        notes=tuple(notes),
    )
