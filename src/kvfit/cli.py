"""kvfit command-line interface. Uses only the standard library.

Subcommands:
    check    Full fit report for a model + workload + GPU.
    sweep    KV cache dtype what-if sweep.
    models   List built-in models.
    gpus     List built-in GPUs.
"""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .cpu import list_cpu_presets, resolve_cpu
from .fit import check_fit
from .gpus import list_gpus, resolve_gpu
from .models import DeviceSpec, Workload
from .report import render_fit, render_sweep
from .resolver import list_models, resolve_model
from .sweep import sweep_kv_dtype


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", "-m", required=True,
                   help="Model alias, local config.json path, or HF repo id.")
    p.add_argument("--context", "-c", type=int, required=True,
                   help="Maximum context length in tokens.")
    p.add_argument("--batch", "-b", type=int, default=1,
                   help="Batch size / concurrent sequences (default 1).")
    p.add_argument("--kv-dtype", default="fp16",
                   help="KV cache dtype: fp16, fp8, int8, int4 (default fp16).")
    p.add_argument("--weight-dtype", default="fp16",
                   help="Weight dtype (default fp16).")
    p.add_argument("--no-color", action="store_true",
                   help="Disable colored output.")


def _color_flag(args: argparse.Namespace) -> bool | None:
    if getattr(args, "no_color", False):
        return False
    return None


def _add_device(p: argparse.ArgumentParser, *, required: bool) -> None:
    grp = p.add_argument_group("target (choose one)")
    grp.add_argument("--gpu", "-g", default=None,
                     help="GPU alias (e.g. a100-40gb) or 'name:GiB'.")
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

    p_sweep = sub.add_parser("sweep", help="KV cache dtype what-if sweep.")
    _add_common(p_sweep)
    _add_device(p_sweep, required=False)

    sub.add_parser("models", help="List built-in models.")
    sub.add_parser("gpus", help="List built-in GPUs.")
    sub.add_parser("cpus", help="List built-in CPU/RAM presets.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    color = _color_flag(args)

    try:
        if args.command == "models":
            print("\n".join(list_models()))
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
        )

        if args.command == "check":
            device = _resolve_device(args)
            assert device is not None
            result = check_fit(model, workload, device)
            print(render_fit(result, color=color))
            if not args.no_sweep:
                print(render_sweep(sweep_kv_dtype(model, workload, device), color=color))
            return 0 if result.fits else 1

        if args.command == "sweep":
            device = _resolve_device(args)
            rows = sweep_kv_dtype(model, workload, device)
            print(render_sweep(rows, color=color, have_gpu=device is not None))
            return 0

    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
