# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and this project adheres to
[Semantic Versioning](https://semver.org/).

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
