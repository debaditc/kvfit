"""Which hardware should I run this on? Rank every GPU that fits a workload.

For each GPU in the catalog, find the smallest tensor/pipeline layout that
fits, then estimate speed and cost. Priced options are ranked by cost per
million output tokens (the number that matters for serving), the rest by how
few GPUs they need.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .fit import check_fit, min_parallelism
from .gpus import canonical_gpus, resolve_gpu
from .models import FitResult, ModelConfig, Workload
from .perf import PerfEstimate, estimate_performance


@dataclass(frozen=True)
class Recommendation:
    """One hardware option that fits: a GPU type, how many, and the layout."""

    gpu: str
    tp: int
    pp: int
    fit: FitResult
    perf: PerfEstimate | None

    @property
    def num_devices(self) -> int:
        return self.tp * self.pp

    @property
    def usd_per_hour(self) -> float | None:
        return self.perf.usd_per_hour if self.perf else None

    @property
    def usd_per_million_output(self) -> float | None:
        return self.perf.usd_per_million_output if self.perf else None

    def to_dict(self) -> dict[str, object]:
        return {
            "gpu": self.gpu,
            "num_devices": self.num_devices,
            "tp": self.tp,
            "pp": self.pp,
            "memory_gib_per_device": round(self.fit.breakdown.total_bytes / 1024**3, 3),
            "headroom_gib_per_device": round(self.fit.headroom_gib, 3),
            "max_context": self.fit.max_context,
            "max_batch": self.fit.max_batch,
            "performance": self.perf.to_dict() if self.perf else None,
        }


def _sort_key(r: Recommendation) -> tuple[int, float, int, float]:
    tok = r.perf.decode_tok_s_total if r.perf else 0.0
    if r.usd_per_million_output is not None:
        return (0, r.usd_per_million_output, r.num_devices, -tok)
    return (1, float(r.num_devices), 0, -tok)


def recommend(
    model: ModelConfig,
    workload: Workload,
    *,
    gpus: list[str] | None = None,
    max_gpus: int = 8,
    priced_only: bool = False,
) -> list[Recommendation]:
    """All GPU options that fit ``workload``, best first.

    Args:
        model: The model to serve.
        workload: Context, batch and dtypes. Its ``tp`` / ``pp`` are ignored;
            the smallest layout that fits is chosen per GPU.
        gpus: Restrict to these GPU names (default: the whole catalog).
        max_gpus: Largest number of GPUs to consider per option.
        priced_only: Drop GPUs with no known hourly price.
    """
    names = gpus if gpus is not None else canonical_gpus()
    base = replace(workload, tp=1, pp=1)
    out: list[Recommendation] = []
    for name in names:
        device = resolve_gpu(name)
        layout = min_parallelism(model, base, device, max_devices=max_gpus)
        if layout is None:
            continue
        tp, pp = layout
        wl = replace(base, tp=tp, pp=pp)
        perf = estimate_performance(model, wl, device)
        if priced_only and (perf is None or perf.usd_per_hour is None):
            continue
        out.append(Recommendation(device.name, tp, pp, check_fit(model, wl, device), perf))
    return sorted(out, key=_sort_key)
