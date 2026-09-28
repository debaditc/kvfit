"""Human-readable terminal output. Pure stdlib, with optional ANSI color.

No ``rich`` dependency: a plain, well-aligned report reads fine in any terminal
and keeps ``pip install kvfit`` dependency-free.
"""

from __future__ import annotations

import os
import sys

from .cost import PRICES_AS_OF, hourly_cost
from .fit import UNLIMITED_CONTEXT, check_fit
from .models import FitResult, GPUSpec, ModelConfig, SweepRow, Workload, dtype_bytes
from .perf import estimate_performance
from .recommend import Recommendation
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


def _fmt_secs(x: float) -> str:
    if x < 1:
        return f"{x * 1000:.0f} ms"
    if x < 120:
        return f"{x:.1f} s"
    return f"{x / 60:.1f} min"


def render_recommendations(
    recs: list[Recommendation],
    *,
    top: int | None = 12,
    color: bool | None = None,
) -> str:
    """Table of hardware options, best first."""
    c = _C(_supports_color() if color is None else color)
    if not recs:
        return "\n  No GPU in the catalog fits this workload within the GPU limit.\n"
    shown = recs[:top] if top else recs
    header = (f"  {'gpu':<24}{'gpus':>5}  {'layout':<11}{'GiB/GPU':>8}{'max ctx':>11}"
              f"{'tok/s/seq':>11}{'tok/s all':>11}{'$/hr':>9}{'$/1M out':>10}")
    lines = ["", c.bold("  Hardware that fits, cheapest per token first"), c.dim(header)]
    for r in shown:
        layout = f"tp{r.tp}" + (f"xpp{r.pp}" if r.pp > 1 else "")
        mc = r.fit.max_context or 0
        ctx = "unlimited" if mc >= UNLIMITED_CONTEXT else f"{mc:,}"
        gib = r.fit.breakdown.total_bytes / 1024**3
        p = r.perf
        seq = f"{p.decode_tok_s_per_seq:,.0f}" if p else "-"
        tot = f"{p.decode_tok_s_total:,.0f}" if p else "-"
        hr = f"{r.usd_per_hour:,.2f}" if r.usd_per_hour is not None else "-"
        pm = f"{r.usd_per_million_output:,.2f}" if r.usd_per_million_output is not None else "-"
        lines.append(f"  {r.gpu:<24}{r.num_devices:>5}  {layout:<11}{gib:>8.1f}{ctx:>11}"
                     f"{seq:>11}{tot:>11}{hr:>9}{pm:>10}")
    if top and len(recs) > top:
        lines.append(c.dim(f"  ... {len(recs) - top} more (use --top 0 to show all)"))
    lines.append(c.dim(f"  Speeds are rough roofline estimates; prices are on-demand medians "
                       f"({PRICES_AS_OF}), '-' = unknown."))
    lines.append("")
    return "\n".join(lines)


def render_fit(
    result: FitResult,
    *,
    color: bool | None = None,
    price_per_hour: float | None = None,
) -> str:
    c = _C(_supports_color() if color is None else color)
    m, wl, gpu = result.model, result.workload, result.gpu
    b = result.breakdown.as_gib()

    lines: list[str] = []
    lines.append("")
    n = wl.num_devices
    if n > 1:
        layout = f"tp={wl.tp}" + (f" x pp={wl.pp}" if wl.pp > 1 else "")
        lines.append(c.bold(f"  kvfit  •  {m.name} on {n}x {gpu.name} ({layout})"))
    else:
        lines.append(c.bold(f"  kvfit  •  {m.name} on {gpu.name}"))
    lines.append(c.dim(
        f"  context {wl.context_length:,} · batch {wl.batch_size} · "
        f"kv {wl.kv_dtype} · weights {wl.weight_dtype}"
    ))
    arch = f"  attention {m.attention_label}"
    if m.is_moe and m.num_active_params is not None:
        arch += (f" · MoE {m.num_params / 1e9:,.0f}B total / "
                 f"~{m.num_active_params / 1e9:,.1f}B active")
    lines.append(c.dim(arch))
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
    per = "  per GPU" if n > 1 else ""
    lines.append(c.bold(f"  {'Total':<12} {_fmt_gib(total)}") + c.dim(per))
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
    if result.max_context == 0:
        limits.append("won't fit at any context length")
    elif result.max_context is not None:
        mc = result.max_context
        limits.append("no cache limit" if mc >= UNLIMITED_CONTEXT else f"max context ~{mc:,}")
    if result.max_batch is not None and result.max_context != 0:
        limits.append(f"max batch {result.max_batch}")
    if limits:
        lines.append(c.dim("  " + "  ·  ".join(limits) + "  at these settings"))
    if m.max_position_embeddings and wl.context_length > m.max_position_embeddings:
        lines.append(c.yellow(
            f"  ! {m.name} supports at most {m.max_position_embeddings:,} tokens of context."
        ))

    # Speed and cost (roofline estimate; rough by design).
    perf = estimate_performance(m, wl, gpu, price_per_hour=price_per_hour) if result.fits else None
    if perf is not None:
        speed = f"  speed ~{perf.decode_tok_s_per_seq:,.0f} tok/s per sequence"
        if wl.batch_size > 1:
            speed += f" · ~{perf.decode_tok_s_total:,.0f} tok/s total"
        speed += f" · {wl.context_length:,}-token prompt in ~{_fmt_secs(perf.prefill_seconds)}"
        lines.append(c.dim(speed + "  (roofline, rough)"))
    cph = hourly_cost(gpu.name, n, price_per_hour=price_per_hour)
    if cph is not None:
        cost = f"  ~${cph:,.2f}/hr" + (f" for {n} GPUs" if n > 1 else "")
        if perf is not None and perf.usd_per_million_output is not None:
            cost += f" · ~${perf.usd_per_million_output:,.2f} per 1M output tokens"
        source = "your price" if price_per_hour is not None else f"on-demand, {PRICES_AS_OF}"
        lines.append(c.dim(cost + f"  ({source})"))
    if not gpu.is_gpu and perf is None:
        lines.append(c.dim(
            "  note: CPU inference is much slower than GPU; this checks whether "
            "it fits in RAM, not how fast it runs."
        ))
    if not gpu.is_gpu and dtype_bytes(wl.weight_dtype) > 1.0:
        lines.append(c.dim(
            "  tip: on CPU, models are usually run quantized — try "
            "--weight-dtype q4_k_m for a realistic estimate."
        ))

    notes = list(m.notes) + (list(perf.notes) if perf else [])
    for note in notes:
        if f"weight_dtype={wl.weight_dtype}" in note:
            continue  # already following the hint
        lines.append(c.dim(f"  note: {note}"))

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
    price_per_hour: float | None = None,
) -> str:
    """Convenience: full fit report plus an optional sweep."""
    out = render_fit(check_fit(model, workload, gpu), color=color,
                     price_per_hour=price_per_hour)
    if show_sweep:
        out += render_sweep(sweep_kv_dtype(model, workload, gpu), color=color)
    return out
