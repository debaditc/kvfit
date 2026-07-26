"""Treat system RAM as a target, so people without a GPU can check whether a
model will fit and run on CPU (llama.cpp / Ollama / GGUF / transformers-on-CPU).

The sizing math is identical to the GPU case — the KV cache and weights are the
same tensors, they just live in system RAM instead of VRAM. What differs is the
memory budget: instead of a CUDA-context reserve we subtract what the OS and
other apps need, and we leave headroom so the machine doesn't thrash.
"""

from __future__ import annotations

from .models import DeviceSpec

# Friendly presets: alias -> total system RAM in GiB.
_CPU_PRESETS: dict[str, float] = {
    "laptop-8gb": 8,
    "laptop-16gb": 16,
    "laptop-32gb": 32,
    "desktop-32gb": 32,
    "desktop-64gb": 64,
    "workstation-128gb": 128,
    "server-256gb": 256,
    "mac-m3-16gb": 16,
    "mac-m3-24gb": 24,
    "mac-m3-max-64gb": 64,
}

# On CPU, memory shared with the OS and other apps. Reserve some, and don't plan
# to use every last byte of what's left.
_DEFAULT_OS_RESERVE_GIB = 2.0
_DEFAULT_USABLE_FRACTION = 0.90


def list_cpu_presets() -> list[str]:
    """Sorted list of known CPU/RAM presets."""
    return sorted(_CPU_PRESETS)


def detect_cpu_ram_gib() -> float | None:
    """Best-effort total system RAM in GiB, or ``None`` if it can't be read.

    Uses ``/proc/meminfo`` on Linux and ``os.sysconf`` as a fallback. Stays in
    the standard library so the core remains dependency-free.
    """
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb * 1024 / (1024**3)
    except (OSError, ValueError):
        pass
    try:
        import os

        page = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        return page * pages / (1024**3)
    except (ValueError, OSError, AttributeError):
        return None
    return None


def resolve_cpu(
    spec: str | float | int,
    *,
    reserve_gib: float = _DEFAULT_OS_RESERVE_GIB,
    usable_fraction: float = _DEFAULT_USABLE_FRACTION,
) -> DeviceSpec:
    """Resolve a CPU/RAM target.

    Accepts:
        * ``"auto"`` — detect this machine's RAM.
        * a number (``32`` or ``"32"``) — that many GiB of RAM.
        * ``"cpu:32"`` / ``"ram:32"`` — that many GiB of RAM.
        * a preset alias (e.g. ``"laptop-16gb"``).
    """
    if isinstance(spec, (int, float)):
        return _make(float(spec), "cpu", reserve_gib, usable_fraction)

    raw = spec.strip().lower()

    if raw == "auto":
        ram = detect_cpu_ram_gib()
        if ram is None:
            raise ValueError(
                "Could not auto-detect system RAM. Pass an explicit size, "
                "e.g. --cpu 32."
            )
        return _make(ram, "cpu (auto-detected)", reserve_gib, usable_fraction)

    if ":" in raw:
        label, _, gib = raw.partition(":")
        try:
            return _make(float(gib), label or "cpu", reserve_gib, usable_fraction)
        except ValueError as exc:
            raise ValueError(f"Could not parse RAM size in {spec!r}") from exc

    if raw in _CPU_PRESETS:
        return _make(_CPU_PRESETS[raw], raw, reserve_gib, usable_fraction)

    # Bare number as string.
    try:
        return _make(float(raw), "cpu", reserve_gib, usable_fraction)
    except ValueError as exc:
        hint = ", ".join(list_cpu_presets()[:5])
        raise ValueError(
            f"Unknown CPU target {spec!r}. Use a number of GiB (e.g. 32), "
            f"'auto', or a preset like: {hint}."
        ) from exc


def _make(gib: float, name: str, reserve: float, frac: float) -> DeviceSpec:
    return DeviceSpec(
        name=name if name.startswith("cpu") else f"cpu-{name}",
        memory_gib=gib,
        reserved_gib=reserve,
        usable_fraction=frac,
        kind="cpu",
    )
