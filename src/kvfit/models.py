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
}

# A sensible order to sweep from "biggest / highest quality" to "smallest".
KV_DTYPE_SWEEP: tuple[str, ...] = ("fp16", "fp8", "int4")

_GIB = 1024**3


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
        sliding_window: If the model uses sliding-window attention, the maximum
            number of tokens ever kept in cache. ``None`` means full attention.
    """

    name: str
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_kv_heads: int
    num_params: float
    head_dim: int | None = None
    sliding_window: int | None = None

    def __post_init__(self) -> None:
        for f in ("num_layers", "hidden_size", "num_attention_heads", "num_kv_heads"):
            if getattr(self, f) <= 0:
                raise ValueError(f"{f} must be positive, got {getattr(self, f)}")
        if self.num_kv_heads > self.num_attention_heads:
            raise ValueError(
                "num_kv_heads cannot exceed num_attention_heads "
                f"({self.num_kv_heads} > {self.num_attention_heads})"
            )

    @property
    def effective_head_dim(self) -> int:
        if self.head_dim is not None:
            return self.head_dim
        return self.hidden_size // self.num_attention_heads

    @property
    def attention_kind(self) -> str:
        if self.num_kv_heads == 1:
            return "MQA"
        if self.num_kv_heads < self.num_attention_heads:
            return "GQA"
        return "MHA"

    @property
    def gqa_savings(self) -> float:
        """Cache reduction vs. full multi-head attention, as a fraction (0..1)."""
        return 1.0 - (self.num_kv_heads / self.num_attention_heads)


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

    def __post_init__(self) -> None:
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        dtype_bytes(self.kv_dtype)  # validate
        dtype_bytes(self.weight_dtype)  # validate


@dataclass(frozen=True)
class MemoryBreakdown:
    """Bytes required, split by what needs the memory."""

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
    """The verdict for a single model + workload + GPU combination."""

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
    def headroom_gib(self) -> float:
        return self.headroom_bytes / _GIB

    @property
    def utilization(self) -> float:
        """Fraction of usable GPU memory this workload consumes."""
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
