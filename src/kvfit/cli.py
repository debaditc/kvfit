"""kvfit command-line interface. Uses only the standard library.

Subcommands:
    check      Full fit report for a model + workload + GPU/CPU target.
    recommend  Rank every GPU (and multi-GPU layout) that fits, by cost per token.
    sweep      KV cache dtype what-if sweep.
    models   List built-in models.
    gpus     List built-in GPUs.
    cpus     List built-in CPU/RAM presets.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .cpu import list_cpu_presets, resolve_cpu
from .fit import check_fit
from .gpus import list_gpus, resolve_gpu
from .models import DeviceSpec, SweepRow, Workload
from .perf import estimate_performance
from .recommend import recommend
from .report import render_fit, render_recommendations, render_sweep
from .resolver import list_models, resolve_model
from .serving import ENGINES, serving_command
from .sweep import sweep_kv_dtype

_DTYPE_HELP = (
    "fp16/bf16, fp8, int8, int4, nvfp4, mxfp4, or a weight format: "
    "awq, gptq, nf4, q8_0, q6_k, q5_k_m, q4_k_m, q3_k_m"
)


def _add_workload(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", "-m", required=True,
                   help="Model alias, local config.json path, or HF repo id.")
    p.add_argument("--context", "-c", type=int, required=True,
                   help="Maximum context length in tokens.")
    p.add_argument("--batch", "-b", type=int, default=1,
                   help="Batch size / concurrent sequences (default 1).")
    p.add_argument("--kv-dtype", default="fp16",
                   help="KV cache dtype: fp16, fp8, int8, int4, nvfp4 (default fp16).")
    p.add_argument("--weight-dtype", default="fp16",
                   help=f"Weight dtype (default fp16): {_DTYPE_HELP}.")
    p.add_argument("--prefill-chunk", type=int, default=2048,
                   help="Tokens per prefill step, e.g. vLLM max_num_batched_tokens "
                        "(default 2048).")
    p.add_argument("--json", action="store_true",
                   help="Print machine-readable JSON instead of the report.")
    p.add_argument("--no-color", action="store_true",
                   help="Disable colored output.")


def _add_common(p: argparse.ArgumentParser) -> None:
    _add_workload(p)
    p.add_argument("--tp", type=int, default=1,
                   help="Tensor-parallel GPUs (default 1). Memory is reported per GPU.")
    p.add_argument("--pp", type=int, default=1,
                   help="Pipeline-parallel stages (default 1).")


def _color_flag(args: argparse.Namespace) -> bool | None:
    if getattr(args, "no_color", False):
        return False
    return None


def _add_device(p: argparse.ArgumentParser, *, required: bool) -> None:
    grp = p.add_argument_group("target (choose one)")
    grp.add_argument("--gpu", "-g", default=None,
                     help="GPU alias (e.g. h100, rtx-5090) or 'name:GiB'.")
    grp.add_argument("--cpu", default=None,
                     help="CPU/RAM target: GiB number, 'auto', or a preset "
                          "(e.g. laptop-16gb).")
    p.set_defaults(_needs_device=required)


def _resolve_device(args: argparse.Namespace) -> DeviceSpec | None:
    if args.gpu and args.cpu:
        raise ValueError("Provide only one of --gpu or --cpu, not both.")
    if args.gpu:
        return resolve_gpu(args.gpu)
    if args.cpu:
        return resolve_cpu(args.cpu)
    if getattr(args, "_needs_device", False):
        raise ValueError("Provide a target: --gpu <alias> or --cpu <GiB|auto|preset>.")
    return None


def _sweep_dicts(rows: list[SweepRow]) -> list[dict[str, object]]:
    return [{"kv_dtype": r.kv_dtype, "kv_cache_gib": round(r.kv_cache_gib, 3),
             "total_gib": round(r.total_gib, 3), "fits": r.fits} for r in rows]


def _models_table() -> str:
    rows = []
    for name in list_models():
        m = resolve_model(name)
        size = f"{m.num_params / 1e9:,.1f}B"
        if m.is_moe and m.num_active_params is not None:
            size += f" (~{m.num_active_params / 1e9:,.1f}B active)"
        ctx = f"{m.max_position_embeddings:,}" if m.max_position_embeddings else "?"
        rows.append((name, size, ctx, m.attention_label))
    w = [max(len(r[i]) for r in rows) for i in range(3)]
    return "\n".join(f"{a:<{w[0]}}  {b:<{w[1]}}  {c:>{w[2]}}  {d}" for a, b, c, d in rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kvfit",
        description="Will your LLM fit? A KV cache & GPU/CPU memory planner.",
    )
    parser.add_argument("--version", action="version",
                        version=f"kvfit {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_check = sub.add_parser("check", help="Full fit report.")
    _add_common(p_check)
    _add_device(p_check, required=True)
    p_check.add_argument("--no-sweep", action="store_true",
                         help="Skip the KV dtype what-if sweep.")
    p_check.add_argument("--price", type=float, default=None,
                         help="Your USD/hour per GPU, overriding the built-in estimate.")
    p_check.add_argument("--emit", choices=ENGINES, default=None,
                         help="Also print a ready-to-run serving command for this engine.")

    p_rec = sub.add_parser("recommend",
                           help="Rank GPUs that fit, cheapest per token first.")
    _add_workload(p_rec)
    p_rec.add_argument("--max-gpus", type=int, default=8,
                       help="Largest GPU count to consider per option (default 8).")
    p_rec.add_argument("--gpus", default=None,
                       help="Comma-separated GPU names to consider (default: all).")
    p_rec.add_argument("--priced-only", action="store_true",
                       help="Only show GPUs with a known hourly price.")
    p_rec.add_argument("--top", type=int, default=12,
                       help="Rows to show (0 = all, default 12).")

    p_sweep = sub.add_parser("sweep", help="KV cache dtype what-if sweep.")
    _add_common(p_sweep)
    _add_device(p_sweep, required=False)

    p_models = sub.add_parser("models", help="List built-in models.")
    p_models.add_argument("--details", "-d", action="store_true",
                          help="Show size, max context and attention layout.")
    sub.add_parser("gpus", help="List built-in GPUs.")
    sub.add_parser("mcp", help="Run the MCP server on stdio (needs kvfit[mcp]).")
    sub.add_parser("cpus", help="List built-in CPU/RAM presets.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    color = _color_flag(args)

    try:
        if args.command == "models":
            print(_models_table() if args.details else "\n".join(list_models()))
            return 0

        if args.command == "mcp":  # pragma: no cover - long-running server
            from .mcp_server import main as mcp_main

            try:
                mcp_main()
            except ImportError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
            return 0

        if args.command == "gpus":
            print("\n".join(list_gpus()))
            return 0

        if args.command == "cpus":
            print("\n".join(list_cpu_presets()))
            return 0

        model = resolve_model(args.model)
        workload = Workload(
            context_length=args.context,
            batch_size=args.batch,
            kv_dtype=args.kv_dtype,
            weight_dtype=args.weight_dtype,
            prefill_chunk=args.prefill_chunk,
            tp=getattr(args, "tp", 1),
            pp=getattr(args, "pp", 1),
        )

        if args.command == "recommend":
            names = [g.strip() for g in args.gpus.split(",")] if args.gpus else None
            recs = recommend(model, workload, gpus=names, max_gpus=args.max_gpus,
                             priced_only=args.priced_only)
            if args.json:
                shown = recs[:args.top] if args.top else recs
                print(json.dumps({"model": model.name,
                                  "options": [r.to_dict() for r in shown]}, indent=2))
            else:
                print(render_recommendations(recs, top=args.top, color=color))
            return 0 if recs else 1

        device = _resolve_device(args)

        if args.command == "check":
            assert device is not None
            result = check_fit(model, workload, device)
            rows = [] if args.no_sweep else sweep_kv_dtype(model, workload, device)
            if args.json:
                out = result.to_dict()
                perf = estimate_performance(model, workload, device,
                                            price_per_hour=args.price)
                out["performance"] = perf.to_dict() if perf else None
                if rows:
                    out["sweep"] = _sweep_dicts(rows)
                if args.emit:
                    out["serving_command"] = serving_command(result, args.emit)
                print(json.dumps(out, indent=2))
            else:
                print(render_fit(result, color=color, price_per_hour=args.price))
                if rows:
                    print(render_sweep(rows, color=color))
                if args.emit:
                    print(serving_command(result, args.emit) + "\n")
            return 0 if result.fits else 1

        if args.command == "sweep":
            rows = sweep_kv_dtype(model, workload, device)
            if args.json:
                print(json.dumps({"model": model.name, "sweep": _sweep_dicts(rows)}, indent=2))
            else:
                print(render_sweep(rows, color=color, have_gpu=device is not None))
            return 0

    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
