"""Turn a model reference into a :class:`ModelConfig`.

Resolution order:
    1. A built-in registry of popular models (works offline, zero deps).
    2. A local ``config.json`` path (Hugging Face format).
    3. The Hugging Face Hub, if ``huggingface_hub`` is installed
       (``pip install "kvfit[hub]"``).

Keeping a small built-in registry means the common case ("does Llama-3-8B fit on
my A100?") works instantly with no network and no dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import ModelConfig

# ---------------------------------------------------------------------------
# Built-in registry. Numbers are approximate but drawn from published configs.
# Aliases are lowercased on lookup.
# ---------------------------------------------------------------------------
_REGISTRY: dict[str, ModelConfig] = {}


def _register(aliases: list[str], cfg: ModelConfig) -> None:
    for a in aliases:
        _REGISTRY[a.lower()] = cfg


_register(
    ["llama-3-8b", "llama-3.1-8b", "meta-llama/llama-3-8b", "llama3-8b"],
    ModelConfig("Llama-3-8B", num_layers=32, hidden_size=4096,
                num_attention_heads=32, num_kv_heads=8, num_params=8.03e9),
)
_register(
    ["llama-3-70b", "llama-3.1-70b", "llama3-70b"],
    ModelConfig("Llama-3-70B", num_layers=80, hidden_size=8192,
                num_attention_heads=64, num_kv_heads=8, num_params=70.6e9),
)
_register(
    ["llama-2-7b", "llama2-7b"],
    ModelConfig("Llama-2-7B", num_layers=32, hidden_size=4096,
                num_attention_heads=32, num_kv_heads=32, num_params=6.74e9),
)
_register(
    ["llama-2-13b", "llama2-13b"],
    ModelConfig("Llama-2-13B", num_layers=40, hidden_size=5120,
                num_attention_heads=40, num_kv_heads=40, num_params=13.0e9),
)
_register(
    ["mistral-7b", "mistral-7b-v0.1", "mistralai/mistral-7b"],
    ModelConfig("Mistral-7B", num_layers=32, hidden_size=4096,
                num_attention_heads=32, num_kv_heads=8, num_params=7.24e9,
                sliding_window=4096),
)
_register(
    ["qwen2.5-7b", "qwen2-7b", "qwen-7b"],
    ModelConfig("Qwen2.5-7B", num_layers=28, hidden_size=3584,
                num_attention_heads=28, num_kv_heads=4, num_params=7.6e9),
)
_register(
    ["qwen2.5-72b", "qwen2-72b"],
    ModelConfig("Qwen2.5-72B", num_layers=80, hidden_size=8192,
                num_attention_heads=64, num_kv_heads=8, num_params=72.7e9),
)
_register(
    ["phi-3-mini", "phi-3-mini-4k", "phi3-mini"],
    ModelConfig("Phi-3-mini", num_layers=32, hidden_size=3072,
                num_attention_heads=32, num_kv_heads=32, num_params=3.8e9),
)
_register(
    ["gemma-2-9b", "gemma2-9b"],
    ModelConfig("Gemma-2-9B", num_layers=42, hidden_size=3584,
                num_attention_heads=16, num_kv_heads=8, num_params=9.24e9,
                head_dim=256),
)
_register(
    ["gemma-2-27b", "gemma2-27b"],
    ModelConfig("Gemma-2-27B", num_layers=46, hidden_size=4608,
                num_attention_heads=32, num_kv_heads=16, num_params=27.2e9,
                head_dim=128),
)


def list_models() -> list[str]:
    """Sorted list of canonical built-in model names."""
    seen: dict[str, None] = {}
    for cfg in _REGISTRY.values():
        seen.setdefault(cfg.name, None)
    return sorted(seen)


def _from_hf_dict(name: str, data: dict[str, Any]) -> ModelConfig:
    """Build a ModelConfig from a Hugging Face ``config.json`` dict."""
    required = ("hidden_size", "num_attention_heads", "num_hidden_layers")
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(
            f"config.json is missing required fields: {', '.join(missing)}."
        )
    hidden = int(data["hidden_size"])
    heads = int(data["num_attention_heads"])
    kv_heads = int(data.get("num_key_value_heads", heads))
    layers = int(data["num_hidden_layers"])
    head_dim = data.get("head_dim")
    sliding = data.get("sliding_window")
    # config.json doesn't carry the param count; leave it 0 and let callers
    # override, or estimate from dimensions if absent.
    num_params = float(data.get("num_parameters", 0)) or _estimate_params(
        layers, hidden, int(data.get("intermediate_size", 4 * hidden)),
        int(data.get("vocab_size", 32000)),
    )
    return ModelConfig(
        name=name,
        num_layers=layers,
        hidden_size=hidden,
        num_attention_heads=heads,
        num_kv_heads=kv_heads,
        num_params=num_params,
        head_dim=int(head_dim) if head_dim else None,
        sliding_window=int(sliding) if sliding else None,
    )


def _estimate_params(layers: int, hidden: int, intermediate: int, vocab: int) -> float:
    """Very rough parameter-count estimate from dimensions (decoder-only)."""
    attn = 4 * hidden * hidden
    mlp = 3 * hidden * intermediate
    per_layer = attn + mlp
    embeddings = 2 * vocab * hidden
    return float(layers * per_layer + embeddings)


def resolve_model(ref: str, *, allow_hub: bool = True) -> ModelConfig:
    """Resolve ``ref`` (registry alias, local config.json path, or hub id)."""
    key = ref.strip()

    # 1) built-in registry
    if key.lower() in _REGISTRY:
        return _REGISTRY[key.lower()]

    # 2) local config.json
    path = Path(key)
    if path.suffix == ".json" and path.is_file():
        data = json.loads(path.read_text())
        return _from_hf_dict(path.stem, data)
    if path.is_dir() and (path / "config.json").is_file():
        data = json.loads((path / "config.json").read_text())
        return _from_hf_dict(path.name, data)

    # 3) Hugging Face Hub (optional dependency)
    if allow_hub:
        cfg = _try_hub(key)
        if cfg is not None:
            return cfg

    raise ValueError(
        f"Could not resolve model {ref!r}. It is not a built-in alias, "
        f"not a local config.json, and could not be fetched from the Hub. "
        f"Known models: {', '.join(list_models())}. "
        f"For arbitrary models install the hub extra: pip install 'kvfit[hub]'."
    )


def _try_hub(repo_id: str) -> ModelConfig | None:
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except ImportError:
        return None
    try:
        cfg_path = hf_hub_download(repo_id=repo_id, filename="config.json")
    except Exception:
        return None
    data = json.loads(Path(cfg_path).read_text())
    return _from_hf_dict(repo_id, data)
