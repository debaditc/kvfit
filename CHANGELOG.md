# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and this project adheres to
[Semantic Versioning](https://semver.org/).

## [0.2.0] — 2026-09-27

Brings the estimator up to date with 2025–26 model architectures. Several of
these were large errors in 0.1.0, not refinements.

### Fixed
- **Interleaved sliding-window models** (Gemma 2/3, gpt-oss, MiMo) were treated as
  windowed on *every* layer when resolved from `config.json`, underestimating KV
  cache up to ~4.5× (Gemma-2-9B @32k: 1.31 → 5.91 GiB).
- **MLA models** (DeepSeek-V3/R1, Kimi K2, GLM-5) now cache the compressed latent
  (`kv_lora_rank + qk_rope_head_dim`) instead of full K/V per head — 0.1.0
  overestimated DeepSeek-V3's cache ~25×.
- **MoE parameter estimates** now count every expert (DeepSeek-V3 from its config:
  38.6B → 671B), plus GQA-aware attention and tied embeddings.
- **Multimodal configs** (`text_config`: Gemma 3, Llama 4, Mistral Small 3.x, …)
  no longer fail to resolve.
- `max_context` / `max_batch` are solved against the same estimate as the verdict,
  so `check(context=max_context)` always fits (it could fail before).
- `use_sliding_window: false` (Qwen2/2.5) is respected; windows ≥ the model's max
  context are ignored.
- Hub errors are no longer swallowed: gated repos and network failures get
  actionable messages.
- When the weights alone exceed the device, suggestions now say how many GPUs are
  needed instead of recommending KV quantization.

### Added
- Per-layer attention model (`ModelConfig.layer_types`): full, sliding, chunked
  (Llama 4), linear/SSM (Qwen3-Next, Qwen3.8, Nemotron-H, Granite 4, Jamba) and
  MLP-only layers; fixed recurrent state for linear layers; DSA indexer cache.
- MoE awareness: `num_active_params`, MoE line in the report, expert-offload tip.
- Exact parameter counts from Hub safetensors metadata (`kvfit[hub]`).
- Dtypes: `nvfp4`, `mxfp4`, `fp4`, `fp8_e4m3/e5m2`, and effective-size weight
  formats `awq`, `gptq`, `nf4`, `q8_0`, `q6_k`, `q5_k_m`, `q4_k_m`, `q3_k_m`.
- Prefill-aware activations (`--prefill-chunk`) and fp32 logits buffer.
- Built-in registry rebuilt from real `config.json` files (37 models, incl. Llama
  3.1–4, Qwen3/3.8, Gemma 3, DeepSeek-V3, Kimi K2, gpt-oss, GLM-4.5/5.3, Phi-4,
  Nemotron-H, Granite 4) via `scripts/build_registry.py`.
- GPUs: B200, B300/GB300, GB200, H20, H100-NVL, MI300X/MI325X/MI355X, RTX PRO 6000,
  RTX 50-series, DGX Spark, Strix Halo, Apple M4 Max / M3 Ultra.
- `--json` output (`FitResult.to_dict()`), `kvfit models --details`, `--price`
  override, and a warning when context exceeds the model's supported maximum.
- Architecture notes: features kvfit can't model exactly are flagged as approximate.

### Added — deployment planning
- **Multi-GPU sharding** (`--tp`, `--pp`; `Workload.tp/pp`): memory is reported per
  GPU, with KV-head replication when `kv_heads < tp` and MLA latents replicated per
  tensor rank. Non-fitting checks suggest the smallest valid `--tp/--pp` layout.
- **Speed and cost per token** (`kvfit.estimate_performance`): roofline decode tok/s
  (MoE-aware expert reads), prefill time, per-layer latency for MoE / tensor
  parallel, and $ per 1M output tokens. GPU catalog gains bandwidth and TFLOPS; CPU
  presets gain RAM bandwidth.
- **`kvfit recommend`** (`kvfit.recommend`): ranks every GPU with its smallest fitting
  layout by $ per 1M output tokens.
- **Serving-command export** (`--emit vllm|sglang|llamacpp|ollama`,
  `kvfit.serving_command`).
- **MCP server** (`kvfit mcp`, `pip install "kvfit[mcp]"`, MCP SDK v2): tools
  `check_fit`, `recommend_hardware`, `kv_dtype_sweep`, `list_models`, `list_hardware`.
- GPU prices refreshed to September 2026 on-demand medians, with an as-of date shown.

### Changed
- `FitResult.to_dict()` reports `memory_gib_per_device` (was `memory_gib`) and
  `num_devices`.
- The unused `measure` extra was replaced by `mcp`.
- `mistral-7b` now means v0.3 (no sliding window); use `mistral-7b-v0.1` for the
  windowed original. `llama-3.1-8b` / `llama-3.3-70b` are separate entries.
- `max_context_for` returns `kvfit.UNLIMITED_CONTEXT` (was `1_000_000`) when the
  cache is bounded.
- `DeviceSpec` validates its inputs; `FitResult.device` added (alias of `.gpu`).
- `__version__` comes from package metadata.

## [0.1.0] — 2026-07-25

### Added
- KV cache sizing math engine (per-token, per-workload), GQA/MQA aware.
- Weight, activation, and framework-overhead memory estimation.
- GPU **and CPU/RAM** targets: check whether a model fits in VRAM or in system
  memory for CPU inference (`--cpu <GiB|auto|preset>`), with RAM-aware reporting.
- Fit checker with max-context and max-batch solvers and actionable suggestions.
- What-if sweep across KV cache dtypes (fp16 / fp8 / int4).
- Built-in registries for popular models, GPUs, and CPU/RAM presets; custom sizes
  via `name:GiB`.
- Local `config.json` resolver and optional Hugging Face Hub resolution (`kvfit[hub]`).
- Zero-dependency CLI (`kvfit check`, `sweep`, `models`, `gpus`, `cpus`) and Python API.
- Rough cloud cost estimates.
