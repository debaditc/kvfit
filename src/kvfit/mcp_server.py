"""Model Context Protocol server: let coding agents ask kvfit "will it fit?".

Run it with ``kvfit mcp`` (stdio) after ``pip install "kvfit[mcp]"``, and
register it with your agent, e.g. for Claude Code::

    claude mcp add kvfit -- kvfit mcp

The tool logic lives in plain functions (``tool_*``) so it is testable, and
usable, without the MCP SDK installed; only :func:`build_server` needs it.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from .cpu import list_cpu_presets, resolve_cpu
from .fit import check_fit
from .gpus import canonical_gpus, resolve_gpu
from .models import DeviceSpec, Workload
from .perf import estimate_performance
from .recommend import recommend
from .resolver import list_models, resolve_model
from .serving import ENGINES, serving_command
from .sweep import sweep_kv_dtype

INSTRUCTIONS = (
    "kvfit estimates LLM inference memory (weights, KV cache, activations) and whether "
    "a model + context length + batch size fits a GPU or CPU/RAM target, with rough "
    "speed and cost per token. Models can be built-in aliases (see list_models), "
    "Hugging Face repo ids, or local config.json paths. Memory is per GPU when tp/pp > 1. "
    "All estimates are planning figures, not benchmarks."
)


def _device(gpu: str | None, cpu: str | None) -> DeviceSpec:
    if (gpu is None) == (cpu is None):
        raise ValueError("Provide exactly one of gpu or cpu.")
    return resolve_gpu(gpu) if gpu is not None else resolve_cpu(cpu or "auto")


def tool_check(
    model: str,
    context: int,
    gpu: str | None = None,
    cpu: str | None = None,
    batch: int = 1,
    kv_dtype: str = "fp16",
    weight_dtype: str = "fp16",
    tp: int = 1,
    pp: int = 1,
    engine: str | None = None,
) -> dict[str, Any]:
    """Check whether a model + workload fits a GPU (e.g. "h100", "rtx-5090", "name:GiB")
    or a CPU/RAM target (GiB number, "auto", or a preset like "mac-m4-max-128gb").

    Returns the per-GPU memory breakdown, fit verdict, headroom, max context and
    batch, suggestions when it doesn't fit, rough speed and $/1M tokens, and
    optionally a serving command for engine = vllm | sglang | llamacpp | ollama.
    """
    m = resolve_model(model)
    wl = Workload(context, batch, kv_dtype, weight_dtype, tp=tp, pp=pp)
    dev = _device(gpu, cpu)
    result = check_fit(m, wl, dev)
    out = result.to_dict()
    perf = estimate_performance(m, wl, dev)
    out["performance"] = perf.to_dict() if perf else None
    if engine:
        out["serving_command"] = serving_command(result, engine)
    return out


def tool_recommend(
    model: str,
    context: int,
    batch: int = 1,
    kv_dtype: str = "fp16",
    weight_dtype: str = "fp16",
    max_gpus: int = 8,
    top: int = 10,
) -> dict[str, Any]:
    """Rank every GPU type (with the smallest tp/pp layout that fits) for serving a
    model at this context and batch size, cheapest per million output tokens first."""
    m = resolve_model(model)
    recs = recommend(m, Workload(context, batch, kv_dtype, weight_dtype), max_gpus=max_gpus)
    return {"model": m.name, "options": [r.to_dict() for r in recs[:top]]}


def tool_sweep(
    model: str,
    context: int,
    batch: int = 1,
    gpu: str | None = None,
    weight_dtype: str = "fp16",
) -> dict[str, Any]:
    """Compare KV cache size and total memory across KV dtypes (fp16 / fp8 / int4)."""
    m = resolve_model(model)
    dev = resolve_gpu(gpu) if gpu else None
    rows = sweep_kv_dtype(m, Workload(context, batch, weight_dtype=weight_dtype), dev)
    return {"model": m.name, "sweep": [
        {"kv_dtype": r.kv_dtype, "kv_cache_gib": round(r.kv_cache_gib, 3),
         "total_gib": round(r.total_gib, 3), "fits": r.fits if dev else None}
        for r in rows]}


def tool_list_models() -> dict[str, Any]:
    """Built-in models with size, active params (MoE), max context and attention layout."""
    out = []
    for name in list_models():
        m = resolve_model(name)
        out.append({"name": name, "hf_repo": m.hf_repo, "num_params": m.num_params,
                    "num_active_params": m.num_active_params,
                    "max_context": m.max_position_embeddings,
                    "attention": m.attention_label})
    return {"models": out}


def tool_list_hardware() -> dict[str, Any]:
    """Built-in GPUs (memory, bandwidth, TFLOPS) and CPU/RAM presets."""
    gpus = []
    for name in canonical_gpus():
        g = resolve_gpu(name)
        gpus.append({"name": name, "memory_gib": g.memory_gib,
                     "bandwidth_gbs": g.bandwidth_gbs, "tflops_bf16": g.tflops})
    return {"gpus": gpus, "cpu_presets": list_cpu_presets(), "engines": list(ENGINES)}


def build_server() -> Any:
    """Create the MCP server (requires ``pip install "kvfit[mcp]"``)."""
    try:
        from mcp.server.mcpserver import MCPServer  # type: ignore[import-not-found,unused-ignore]
        from mcp.server.mcpserver.exceptions import (  # type: ignore[import-not-found,unused-ignore]
            ToolError,
        )
    except ImportError as exc:
        raise ImportError(
            "The MCP server needs the MCP SDK (v2+): pip install 'kvfit[mcp]'"
        ) from exc
    from . import __version__

    def anticipated(fn: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
        # Bad model names, unknown GPUs etc. raise ValueError with a helpful
        # message; as ToolError the agent sees that message instead of a crash.
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
            try:
                return fn(*args, **kwargs)
            except ValueError as exc:
                raise ToolError(str(exc)) from exc
        return wrapper

    server = MCPServer("kvfit", instructions=INSTRUCTIONS, version=__version__)
    for fn, name in ((tool_check, "check_fit"), (tool_recommend, "recommend_hardware"),
                     (tool_sweep, "kv_dtype_sweep"), (tool_list_models, "list_models"),
                     (tool_list_hardware, "list_hardware")):
        server.tool(name=name)(anticipated(fn))
    return server


def main() -> None:  # pragma: no cover - exercised manually / by MCP clients
    build_server().run("stdio")
