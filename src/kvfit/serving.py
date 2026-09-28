"""Turn a fit check into a ready-to-run serving command.

The numbers kvfit just computed (max context, concurrency, memory fraction,
parallelism, KV dtype) map directly onto serving-engine flags, so there is no
reason to re-type them. Flags reflect each engine's documented CLI; check your
installed version if a flag is rejected.
"""

from __future__ import annotations

from .models import FitResult

ENGINES: tuple[str, ...] = ("vllm", "sglang", "llamacpp", "ollama")

_WEIGHT_FORMATS_NEEDING_CHECKPOINT = {"awq", "gptq", "nf4", "nvfp4", "mxfp4", "int4", "int8",
                                      "fp4"}
_GGUF_TYPES = {"q8_0": "Q8_0", "q6_k": "Q6_K", "q5_k_m": "Q5_K_M", "q4_k_m": "Q4_K_M",
               "q3_k_m": "Q3_K_M", "fp16": "F16", "bf16": "BF16", "int8": "Q8_0",
               "int4": "Q4_K_M"}


def _join(cmd: str, flags: list[str], comments: list[str]) -> str:
    lines = [f"# {c}" for c in comments]
    lines.append(" \\\n  ".join([cmd, *flags]))
    return "\n".join(lines)


def _model_ref(result: FitResult) -> str:
    return result.model.hf_repo or result.model.name


def _vllm(result: FitResult) -> str:
    wl, dev = result.workload, result.gpu
    flags = [
        f"--max-model-len {wl.context_length}",
        f"--max-num-seqs {wl.batch_size}",
        f"--gpu-memory-utilization {dev.usable_fraction:.2f}",
        f"--max-num-batched-tokens {wl.prefill_chunk}",
    ]
    comments: list[str] = []
    if wl.tp > 1:
        flags.append(f"--tensor-parallel-size {wl.tp}")
    if wl.pp > 1:
        flags.append(f"--pipeline-parallel-size {wl.pp}")
    kv = wl.kv_dtype.lower()
    if kv in ("fp8", "fp8_e4m3", "f8"):
        flags.append("--kv-cache-dtype fp8")
    elif kv == "fp8_e5m2":
        flags.append("--kv-cache-dtype fp8_e5m2")
    elif kv not in ("fp16", "f16", "bf16"):
        comments.append(f"kv_dtype={kv}: check which --kv-cache-dtype values your vLLM supports.")
    w = wl.weight_dtype.lower()
    if w in ("fp8", "fp8_e4m3"):
        flags.append("--quantization fp8")
    elif w in _WEIGHT_FORMATS_NEEDING_CHECKPOINT:
        comments.append(f"weight_dtype={w}: point at a pre-quantized {w.upper()} checkpoint.")
    return _join(f"vllm serve {_model_ref(result)}", flags, comments)


def _sglang(result: FitResult) -> str:
    wl, dev = result.workload, result.gpu
    flags = [
        f"--model-path {_model_ref(result)}",
        f"--context-length {wl.context_length}",
        f"--max-running-requests {wl.batch_size}",
        f"--mem-fraction-static {dev.usable_fraction - 0.02:.2f}",
        f"--chunked-prefill-size {wl.prefill_chunk}",
    ]
    comments: list[str] = []
    if wl.tp > 1:
        flags.append(f"--tp-size {wl.tp}")
    if wl.pp > 1:
        flags.append(f"--pp-size {wl.pp}")
    kv = wl.kv_dtype.lower()
    if kv in ("fp8", "fp8_e4m3", "f8"):
        flags.append("--kv-cache-dtype fp8_e4m3")
    elif kv not in ("fp16", "f16", "bf16"):
        comments.append(f"kv_dtype={kv}: check which --kv-cache-dtype values your SGLang supports.")
    w = wl.weight_dtype.lower()
    if w in ("fp8", "fp8_e4m3"):
        flags.append("--quantization fp8")
    elif w in _WEIGHT_FORMATS_NEEDING_CHECKPOINT:
        comments.append(f"weight_dtype={w}: point at a pre-quantized {w.upper()} checkpoint.")
    return _join("python -m sglang.launch_server", flags, comments)


def _gguf_name(result: FitResult) -> tuple[str, list[str]]:
    w = result.workload.weight_dtype.lower()
    quant = _GGUF_TYPES.get(w)
    comments = []
    if quant is None:
        quant = "Q4_K_M"
        comments.append(f"weight_dtype={w} has no GGUF equivalent; re-check with q4_k_m / q8_0.")
    return f"{result.model.name}-{quant}.gguf", comments


def _kv_cache_type(kv: str) -> str | None:
    kv = kv.lower()
    if kv in ("int8", "i8", "fp8", "fp8_e4m3", "fp8_e5m2", "f8"):
        return "q8_0"
    if kv in ("int4", "i4", "fp4", "nvfp4", "mxfp4"):
        return "q4_0"
    return None


def _llamacpp(result: FitResult) -> str:
    wl, dev = result.workload, result.gpu
    path, comments = _gguf_name(result)
    # Apple Silicon: unified memory, and llama.cpp offloads to the Metal GPU.
    offload = dev.is_gpu or "mac" in dev.name
    flags = [
        f"-m {path}",
        f"-c {wl.context_length * wl.batch_size}",  # total across all slots
        f"-np {wl.batch_size}",
        f"-ngl {99 if offload else 0}",
        f"-b {wl.prefill_chunk}",
        f"-ub {min(wl.prefill_chunk, 512)}",
        "-fa on",
    ]
    cache = _kv_cache_type(wl.kv_dtype)
    if cache:
        flags += [f"--cache-type-k {cache}", f"--cache-type-v {cache}"]
    if wl.num_devices > 1:
        comments.append(f"llama.cpp splits layers across all visible GPUs ({wl.num_devices}).")
    return _join("llama-server", flags, comments)


def _ollama(result: FitResult) -> str:
    wl = result.workload
    env = [
        f"OLLAMA_CONTEXT_LENGTH={wl.context_length}",
        f"OLLAMA_NUM_PARALLEL={wl.batch_size}",
        "OLLAMA_FLASH_ATTENTION=1",
    ]
    cache = _kv_cache_type(wl.kv_dtype)
    if cache:
        env.append(f"OLLAMA_KV_CACHE_TYPE={cache}")
    return " ".join([*env, "ollama serve"])


def serving_command(result: FitResult, engine: str) -> str:
    """A launch command for ``engine`` (vllm, sglang, llamacpp, ollama)."""
    engine = engine.lower().replace(".", "").replace("-", "")
    builders = {"vllm": _vllm, "sglang": _sglang, "llamacpp": _llamacpp, "ollama": _ollama}
    if engine not in builders:
        raise ValueError(f"Unknown engine {engine!r}. Choose from: {', '.join(ENGINES)}")
    out = builders[engine](result)
    m = result.model
    if m.max_position_embeddings and result.workload.context_length > m.max_position_embeddings:
        out = (f"# warning: {m.name} supports at most {m.max_position_embeddings:,} tokens.\n"
               + out)
    if not result.fits:
        out = "# warning: kvfit estimates this configuration does NOT fit.\n" + out
    return out
