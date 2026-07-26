"""Human-readable terminal output. Pure stdlib, with optional ANSI color.

No ``rich`` dependency: a plain, well-aligned report reads fine in any terminal
and keeps ``pip install kvfit`` dependency-free.
"""

from __future__ import annotations

import os
import sys

from .cost import hourly_cost
from .fit import check_fit
from .models import FitResult, GPUSpec, ModelConfig, SweepRow, Workload, dtype_bytes
from .sweep import sweep_kv_dtype


def _supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return sys.stdout.isatty()


class _C:
    """ANSI helpers that no-op when color is unsupported."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def bold(self, t: str) -> str:
        return self._wrap("1", t)

    def green(self, t: str) -> str:
        return self._wrap("32", t)

    def red(self, t: str) -> str:
        return self._wrap("31", t)

    def yellow(self, t: str) -> str:
        return self._wrap("33", t)

    def dim(self, t: str) -> str:
        return self._wrap("2", t)


def _bar(fraction: float, width: int = 24) -> str:
    fraction = max(0.0, min(1.5, fraction))
    filled = int(round(min(fraction, 1.0) * width))
    over = fraction > 1.0
    bar = "█" * filled + "░" * (width - filled)
    return bar + (" !" if over else "")


def _fmt_gib(x: float) -> str:
    return f"{x:6.2f} GiB"


def render_fit(result: FitResult, *, color: bool | None = None) -> str:
    c = _C(_supports_color() if color is None else color)
    m, wl, gpu = result.model, result.workload, result.gpu
    b = result.breakdown.as_gib()

    lines: list[str] = []
    lines.append("")
    lines.append(c.bold(f"  kvfit  •  {m.name} on {gpu.name}"))
    lines.append(c.dim(
        f"  context {wl.context_length:,} · batch {wl.batch_size} · "
        f"kv {wl.kv_dtype} · weights {wl.weight_dtype} · "
        f"attention {m.attention_kind} ({m.num_kv_heads}/{m.num_attention_heads} kv heads)"
    ))
    lines.append("")

    # Memory breakdown
    rows = [
        ("Weights", b["weights"]),
        ("KV cache", b["kv_cache"]),
        ("Activations", b["activations"]),
        ("Overhead", b["framework_overhead"]),
    ]
    total = b["total"]
    for label, val in rows:
        frac = val / total if total else 0
        lines.append(f"  {label:<12} {_fmt_gib(val)}  {c.dim(_bar(frac))}")
    lines.append(c.bold(f"  {'Total':<12} {_fmt_gib(total)}"))
    lines.append(f"  {'Usable ' + gpu.memory_label:<12} {_fmt_gib(gpu.usable_gib)}  "
                 f"{c.dim(f'({round(gpu.memory_gib, 1):g} GiB {gpu.hardware_label})')}")
    lines.append("")

    # Verdict
    util = result.utilization
    if result.fits:
        lines.append("  " + c.green(c.bold("✓ FITS"))
                     + f"  {c.green(f'{result.headroom_gib:.2f} GiB headroom')}"
                     + c.dim(f"  ({util*100:.0f}% of usable memory)"))
    else:
        need = -result.headroom_gib
        lines.append("  " + c.red(c.bold("✗ DOES NOT FIT"))
                     + f"  {c.red(f'over by {need:.2f} GiB')}"
                     + c.dim(f"  ({util*100:.0f}% of usable memory)"))

    # Practical limits
    limits = []
    if result.max_context is not None:
        mc = result.max_context
        limits.append("no cache limit" if mc >= 1_000_000 else f"max context ~{mc:,}")
    if result.max_batch is not None:
        limits.append(f"max batch {result.max_batch}")
    if limits:
        lines.append(c.dim("  " + "  ·  ".join(limits) + "  at these settings"))

    # Cost (GPU) or a plain-spoken CPU caveat.
    if gpu.is_gpu:
        cph = hourly_cost(gpu.name)
        if cph is not None:
            lines.append(c.dim(f"  ~${cph:.2f}/hr on-demand (rough)"))
    else:
        lines.append(c.dim(
            "  note: CPU inference is much slower than GPU; this checks whether "
            "it fits in RAM, not how fast it runs."
        ))
        if dtype_bytes(wl.weight_dtype) > 1.0:
            lines.append(c.dim(
                "  tip: on CPU, models are usually run quantized — try "
                "--weight-dtype int4 for a realistic estimate."
            ))

    # Suggestions
    if result.suggestions:
        lines.append("")
        lines.append(c.yellow("  What would make it fit:"))
        for tip in result.suggestions:
            lines.append(c.yellow(f"    → {tip}"))

    lines.append("")
    return "\n".join(lines)


def render_sweep(
    rows: list[SweepRow],
    *,
    color: bool | None = None,
    have_gpu: bool = True,
) -> str:
    c = _C(_supports_color() if color is None else color)
    lines = ["", c.bold("  KV cache what-if sweep")]
    header = f"  {'dtype':<8}{'kv cache':>12}{'total':>12}"
    if have_gpu:
        header += f"{'fits':>8}"
    lines.append(c.dim(header))
    for r in rows:
        line = f"  {r.kv_dtype:<8}{r.kv_cache_gib:>9.2f} GiB{r.total_gib:>9.2f} GiB"
        if have_gpu:
            mark = c.green("  yes") if r.fits else c.red("   no")
            line += f"{mark:>8}"
        lines.append(line)
    lines.append("")
    return "\n".join(lines)


def report(
    model: ModelConfig,
    workload: Workload,
    gpu: GPUSpec,
    *,
    show_sweep: bool = True,
    color: bool | None = None,
) -> str:
    """Convenience: full fit report plus an optional sweep."""
    out = render_fit(check_fit(model, workload, gpu), color=color)
    if show_sweep:
        out += render_sweep(sweep_kv_dtype(model, workload, gpu), color=color)
    return out
