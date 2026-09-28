"""Core data structures for kvfit.

Everything here is a plain, typed dataclass. No heavy dependencies, no I/O.
The whole point of keeping these pure is that the estimation math is easy to
reason about and easy to test.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Dtypes
# ---------------------------------------------------------------------------
# Bytes-per-element for the dtypes people actually store weights / KV cache in.
# int4 is 0.5 because two values are packed per byte.
BYTES_PER_DTYPE: dict[str, float] = {
    "fp32": 4.0,
    "f32": 4.0,
    "fp16": 2.0,
    "f16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "f8": 1.0,
    "int8": 1.0,
    "i8": 1.0,
    "int4": 0.5,
    "i4": 0.5,
    # Newer low-precision formats (Hopper / Blackwell / gpt-oss).
    "fp8_e4m3": 1.0,
    "fp8_e5m2": 1.0,
    "fp4": 0.5,
    "nvfp4": 0.5625,  # 4 bits + one fp8 scale per 16 values
    "mxfp4": 0.53125,  # 4 bits + one 8-bit scale per 32 values
    # Effective bytes/param of common quantized *weight* formats, including their
    # group scales. Plain "int4" above is the raw 4 bits and slightly optimistic.
    "awq": 0.53125,  # 4-bit, group 128
    "gptq": 0.53125,  # 4-bit, group 128
    "nf4": 0.516,  # bitsandbytes NF4 with double quantization
    "q8_0": 1.0625,  # GGUF, ~8.5 bits/weight
    "q6_k": 0.82,  # GGUF, ~6.56 bits/weight
    "q5_k_m": 0.7125,  # GGUF, ~5.7 bits/weight
    "q4_k_m": 0.6063,  # GGUF, ~4.85 bits/weight
    "q3_k_m": 0.4875,  # GGUF, ~3.9 bits/weight
}

# A sensible order to sweep from "biggest / highest quality" to "smallest".
KV_DTYPE_SWEEP: tuple[str, ...] = ("fp16", "fp8", "int4")

_GIB = 1024**3

# Per-layer attention kinds (see ``ModelConfig.layer_types``).
FULL = "full"  # caches every token of the context
SLIDING = "sliding"  # caches at most ``sliding_window`` tokens
CHUNKED = "chunked"  # chunked local attention (Llama 4); same cap as sliding
LINEAR = "linear"  # linear attention / SSM (Mamba, Gated DeltaNet): fixed-size state
NONE = "none"  # no attention in this layer (e.g. MLP-only blocks in Nemotron-H)
LAYER_KINDS: tuple[str, ...] = (FULL, SLIDING, CHUNKED, LINEAR, NONE)


def dtype_bytes(dtype: str) -> float:
    """Return bytes-per-element for a dtype string, case-insensitively."""
    key = dtype.strip().lower()
    if key not in BYTES_PER_DTYPE:
        valid = ", ".join(sorted(set(BYTES_PER_DTYPE)))
        raise ValueError(f"Unknown dtype {dtype!r}. Valid options: {valid}")
    return BYTES_PER_DTYPE[key]


@dataclass(frozen=True)
class ModelConfig:
    """The handful of architecture numbers that determine KV cache size.

    Attributes:
        name: Human-readable identifier (e.g. ``"meta-llama/Llama-3-8B"``).
        num_layers: Number of transformer decoder layers.
        hidden_size: Model hidden dimension (a.k.a. ``d_model``).
        num_attention_heads: Number of query heads.
        num_kv_heads: Number of key/value heads. Equals ``num_attention_heads``
            for classic multi-head attention (MHA); is smaller for grouped-query
            attention (GQA) and equals 1 for multi-query attention (MQA). This is
            the single most important field for cache size and the one naive
            calculators get wrong.
        num_params: Total parameter count, used only to estimate weight memory.
        head_dim: Dimension per head. Defaults to ``hidden_size // heads``.
        sliding_window: Window (or chunk) size for sliding / chunked layers. With
            no ``layer_types`` every layer is treated as sliding (Mistral-7B v0.1
            style). ``None`` means full attention.
        layer_types: Optional per-layer kinds (``full``, ``sliding``, ``chunked``,
            ``linear``, ``none``) for hybrid models such as Gemma 3 (5 sliding : 1
            full), gpt-oss, Llama 4, Qwen3-Next / Qwen3.8 (linear + full) and
            Mamba hybrids. ``None`` means every layer is the same.
        v_head_dim: Value head size when it differs from the key head size.
        kv_lora_rank: Multi-head latent attention (MLA, DeepSeek-V3 / Kimi K2 /
            GLM-5). When set, each token caches one compressed latent of
            ``kv_lora_rank + qk_rope_head_dim`` values per layer, independent of
            the number of heads.
        qk_rope_head_dim: Decoupled RoPE key size cached alongside the MLA latent.
        indexer_head_dim: Sparse-attention indexer key size (DeepSeek-V3.2 / GLM-5
            DSA), cached per token on ``indexer_layers`` layers.
        indexer_layers: Number of layers that hold their own indexer cache.
            Defaults to every attention layer.
        linear_state_bytes: Fixed recurrent state per ``linear`` layer per
            sequence, in bytes. It does not grow with context.
        num_active_params: Parameters used per token (MoE). ``None`` = dense.
        num_experts: Routed experts per MoE layer (``None`` = dense).
        experts_per_token: Routed experts each token activates.
        intermediate_size: MLP width active per token, used for activations.
        vocab_size: Vocabulary size, used for the logits buffer.
        max_position_embeddings: Longest context the model supports.
        hf_repo: Hugging Face repo id, used when generating serving commands.
        notes: Caveats found while resolving the model (shown in reports).
    """

    name: str
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_kv_heads: int
    num_params: float
    head_dim: int | None = None
    sliding_window: int | None = None
    layer_types: tuple[str, ...] | None = None
    v_head_dim: int | None = None
    kv_lora_rank: int | None = None
    qk_rope_head_dim: int | None = None
    indexer_head_dim: int | None = None
    indexer_layers: int | None = None
    linear_state_bytes: float = 0.0
    num_active_params: float | None = None
    num_experts: int | None = None
    experts_per_token: int | None = None
    intermediate_size: int | None = None
    vocab_size: int | None = None
    max_position_embeddings: int | None = None
    hf_repo: str | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for f in ("num_layers", "hidden_size", "num_attention_heads", "num_kv_heads"):
            if getattr(self, f) <= 0:
                raise ValueError(f"{f} must be positive, got {getattr(self, f)}")
        if self.num_kv_heads > self.num_attention_heads:
            raise ValueError(
                "num_kv_heads cannot exceed num_attention_heads "
                f"({self.num_kv_heads} > {self.num_attention_heads})"
            )
        if self.layer_types is not None:
            if len(self.layer_types) != self.num_layers:
                raise ValueError(
                    f"layer_types has {len(self.layer_types)} entries but the model "
                    f"has {self.num_layers} layers"
                )
            bad = sorted(set(self.layer_types) - set(LAYER_KINDS))
            if bad:
                raise ValueError(f"Unknown layer types {bad}; valid: {LAYER_KINDS}")
            windowed = {SLIDING, CHUNKED} & set(self.layer_types)
            if windowed and not self.sliding_window:
                raise ValueError("sliding / chunked layers need a sliding_window size")

    @property
    def effective_head_dim(self) -> int:
        if self.head_dim is not None:
            return self.head_dim
        return self.hidden_size // self.num_attention_heads

    @property
    def layer_kinds(self) -> tuple[str, ...]:
        """Per-layer attention kind, expanding the uniform default."""
        if self.layer_types is not None:
            return self.layer_types
        return ((SLIDING if self.sliding_window else FULL),) * self.num_layers

    @property
    def is_mla(self) -> bool:
        return self.kv_lora_rank is not None

    @property
    def is_moe(self) -> bool:
        return self.num_active_params is not None and self.num_active_params < self.num_params

    @property
    def attention_kind(self) -> str:
        if self.is_mla:
            return "MLA"
        if self.num_kv_heads == 1:
            return "MQA"
        if self.num_kv_heads < self.num_attention_heads:
            return "GQA"
        return "MHA"

    @property
    def gqa_savings(self) -> float:
        """Cache reduction vs. full multi-head attention, as a fraction (0..1)."""
        return 1.0 - (self.num_kv_heads / self.num_attention_heads)

    @property
    def attention_label(self) -> str:
        """Short human description, e.g. ``GQA (8/32 kv heads), 52/62 sliding@1024``."""
        if self.is_mla:
            label = f"MLA (latent {self.kv_lora_rank}+{self.qk_rope_head_dim or 0})"
        else:
            label = (f"{self.attention_kind} "
                     f"({self.num_kv_heads}/{self.num_attention_heads} kv heads)")
        kinds = self.layer_kinds
        n = len(kinds)
        for kind, word in ((SLIDING, "sliding"), (CHUNKED, "chunked"), (LINEAR, "linear")):
            k = kinds.count(kind)
            if k == 0:
                continue
            window = f"@{self.sliding_window:,}" if kind != LINEAR else ""
            label += f", {'all' if k == n else f'{k}/{n}'} {word}{window}"
        return label


@dataclass(frozen=True)
class DeviceSpec:
    """A memory target to check against — a GPU (VRAM) or a CPU (system RAM).

    The sizing math is device-agnostic; ``kind`` only affects budgeting defaults
    and how the result is described.
    """

    name: str
    memory_gib: float
    # Reserved memory that is never available for weights/cache. On a GPU this is
    # the CUDA context and kernels; on a CPU it's the OS and other applications.
    reserved_gib: float = 0.75
    # Fraction of the *remaining* memory you'll actually plan to use, before
    # fragmentation / allocator slack / thrashing. vLLM defaults to ~0.90.
    usable_fraction: float = 0.90
    kind: str = "gpu"  # "gpu" or "cpu"
    # Performance specs, used only for the speed / cost-per-token estimate.
    bandwidth_gbs: float | None = None  # peak memory bandwidth, GB/s
    tflops: float | None = None  # dense bf16/fp16 tensor TFLOPS

    def __post_init__(self) -> None:
        if self.memory_gib <= 0:
            raise ValueError(f"memory_gib must be positive, got {self.memory_gib}")
        if not 0 < self.usable_fraction <= 1:
            raise ValueError("usable_fraction must be in (0, 1]")
        if self.reserved_gib < 0:
            raise ValueError("reserved_gib cannot be negative")
        if self.kind not in ("gpu", "cpu"):
            raise ValueError(f"kind must be 'gpu' or 'cpu', got {self.kind!r}")

    @property
    def usable_bytes(self) -> float:
        return max(0.0, (self.memory_gib - self.reserved_gib)) * self.usable_fraction * _GIB

    @property
    def usable_gib(self) -> float:
        return self.usable_bytes / _GIB

    @property
    def is_gpu(self) -> bool:
        return self.kind == "gpu"

    @property
    def memory_label(self) -> str:
        """'VRAM' for a GPU, 'RAM' for a CPU."""
        return "VRAM" if self.is_gpu else "RAM"

    @property
    def hardware_label(self) -> str:
        """'card' for a GPU, 'system' for a CPU — used in reports."""
        return "card" if self.is_gpu else "system"


# Backwards-compatible alias: kvfit started GPU-only.
GPUSpec = DeviceSpec


@dataclass(frozen=True)
class Workload:
    """What you intend to run: how long, how wide, in what precision."""

    context_length: int
    batch_size: int = 1
    kv_dtype: str = "fp16"
    weight_dtype: str = "fp16"
    # Tokens processed per prefill step (vLLM ``max_num_batched_tokens``,
    # llama.cpp ``--ubatch-size``). Sets the activation peak during prefill.
    prefill_chunk: int = 2048
    # Parallelism: tensor-parallel ranks x pipeline stages = GPUs used.
    tp: int = 1
    pp: int = 1

    @property
    def num_devices(self) -> int:
        return self.tp * self.pp

    def __post_init__(self) -> None:
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.prefill_chunk <= 0:
            raise ValueError("prefill_chunk must be positive")
        if self.tp <= 0 or self.pp <= 0:
            raise ValueError("tp and pp must be positive")
        dtype_bytes(self.kv_dtype)  # validate
        dtype_bytes(self.weight_dtype)  # validate


@dataclass(frozen=True)
class MemoryBreakdown:
    """Bytes required *per device*, split by what needs the memory.

    With ``tp = pp = 1`` this is the whole model; when sharded it is what each
    GPU must hold.
    """

    weights_bytes: float
    kv_cache_bytes: float
    activation_bytes: float
    framework_overhead_bytes: float

    @property
    def total_bytes(self) -> float:
        return (
            self.weights_bytes
            + self.kv_cache_bytes
            + self.activation_bytes
            + self.framework_overhead_bytes
        )

    def as_gib(self) -> dict[str, float]:
        return {
            "weights": self.weights_bytes / _GIB,
            "kv_cache": self.kv_cache_bytes / _GIB,
            "activations": self.activation_bytes / _GIB,
            "framework_overhead": self.framework_overhead_bytes / _GIB,
            "total": self.total_bytes / _GIB,
        }


@dataclass(frozen=True)
class FitResult:
    """The verdict for a single model + workload + device combination."""

    model: ModelConfig
    workload: Workload
    gpu: GPUSpec
    breakdown: MemoryBreakdown
    fits: bool
    headroom_bytes: float
    max_context: int | None = None
    max_batch: int | None = None
    suggestions: list[str] = field(default_factory=list)

    @property
    def device(self) -> DeviceSpec:
        """The target checked against (``gpu`` is kept for backwards compatibility)."""
        return self.gpu

    @property
    def headroom_gib(self) -> float:
        return self.headroom_bytes / _GIB

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable summary, for CI tooling and agents."""
        m, wl, d = self.model, self.workload, self.gpu
        return {
            "model": m.name,
            "attention": m.attention_label,
            "num_params": m.num_params,
            "num_active_params": m.num_active_params,
            "device": {"name": d.name, "kind": d.kind, "memory_gib": d.memory_gib,
                       "usable_gib": round(d.usable_gib, 3)},
            "workload": {"context_length": wl.context_length, "batch_size": wl.batch_size,
                         "kv_dtype": wl.kv_dtype, "weight_dtype": wl.weight_dtype,
                         "prefill_chunk": wl.prefill_chunk, "tp": wl.tp, "pp": wl.pp},
            "num_devices": wl.num_devices,
            "memory_gib_per_device": {k: round(v, 3)
                                      for k, v in self.breakdown.as_gib().items()},
            "fits": self.fits,
            "headroom_gib": round(self.headroom_gib, 3),
            "utilization": round(self.utilization, 4),
            "max_context": self.max_context,
            "max_batch": self.max_batch,
            "suggestions": list(self.suggestions),
            "notes": list(m.notes),
        }

    @property
    def utilization(self) -> float:
        """Fraction of usable device memory this workload consumes."""
        usable = self.gpu.usable_bytes
        return self.breakdown.total_bytes / usable if usable > 0 else float("inf")


@dataclass(frozen=True)
class SweepRow:
    """One row of a what-if sweep across KV cache dtypes."""

    kv_dtype: str
    kv_cache_bytes: float
    total_bytes: float
    fits: bool

    @property
    def kv_cache_gib(self) -> float:
        return self.kv_cache_bytes / _GIB

    @property
    def total_gib(self) -> float:
        return self.total_bytes / _GIB
