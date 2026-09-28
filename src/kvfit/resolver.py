"""Turn a model reference into a :class:`ModelConfig`.

Resolution order:
    1. A built-in registry of popular models (works offline, zero deps).
    2. A local ``config.json`` path (Hugging Face format).
    3. The Hugging Face Hub, if ``huggingface_hub`` is installed
       (``pip install "kvfit[hub]"``).

The built-in registry (``data/models.json``) stores trimmed copies of each
model's real ``config.json`` and is parsed by the same code as a user-supplied
config, so there is exactly one code path that understands architectures.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

from .models import CHUNKED, FULL, LINEAR, NONE, SLIDING, ModelConfig

# Config keys for architecture features kvfit does not model exactly. Seeing one
# adds a note so the report says "approximate" instead of being silently wrong.
_UNMODELED_KEYS: dict[str, str] = {
    "compress_ratios": "per-layer KV compression",
    "kv_source_layer_ids": "cross-layer KV sharing",
    "engram_layer_ids": "n-gram memory tables",
    "swa_num_key_value_heads": "different KV heads on sliding layers",
    "indexer_compress_ratio": "compressed sparse-attention indexer",
    "ngram_vocab_size_base": "n-gram embedding tables",
    "hc_mult": "hyper-connections",
    "hc_count": "hyper-connections",
}

# Layer-type spellings used across Hugging Face configs.
_LAYER_TYPE_ALIASES: dict[str, str] = {
    "full_attention": FULL,
    "attention": FULL,
    "full": FULL,
    "global": FULL,
    "sliding_attention": SLIDING,
    "sliding": SLIDING,
    "local": SLIDING,
    "chunked_attention": CHUNKED,
    "linear_attention": LINEAR,
    "mamba": LINEAR,
    "mamba2": LINEAR,
    "ssm": LINEAR,
    "linear": LINEAR,
}


# ---------------------------------------------------------------------------
# Built-in registry
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _registry() -> tuple[dict[str, ModelConfig], tuple[str, ...]]:
    """Alias -> ModelConfig, plus canonical names in file order."""
    raw = resources.files("kvfit").joinpath("data/models.json").read_text()
    table: dict[str, ModelConfig] = {}
    names: list[str] = []
    for entry in json.loads(raw)["models"]:
        cfg = _from_hf_dict(entry["name"], entry["config"],
                            num_params=float(entry["num_params"]),
                            hf_repo=entry.get("repo"))
        names.append(cfg.name)
        for alias in [entry["name"], *entry["aliases"]]:
            table[alias.lower()] = cfg
    return table, tuple(names)


def list_models() -> list[str]:
    """Sorted list of canonical built-in model names."""
    return sorted(_registry()[1], key=str.lower)


# ---------------------------------------------------------------------------
# config.json parsing
# ---------------------------------------------------------------------------


def _text_config(data: dict[str, Any]) -> dict[str, Any]:
    """Multimodal configs nest the language model under ``text_config`` etc."""
    for key in ("text_config", "llm_config", "language_config"):
        sub = data.get(key)
        if isinstance(sub, dict) and "hidden_size" in sub:
            merged = {k: v for k, v in data.items() if not isinstance(v, dict)}
            merged.update(sub)
            if "quantization_config" in data:
                merged["quantization_config"] = data["quantization_config"]
            return merged
    return data


def _first(data: dict[str, Any], *keys: str) -> Any:
    for k in keys:
        v = data.get(k)
        if v is not None:
            return v
    return None


def _window(data: dict[str, Any]) -> int | None:
    w = _first(data, "sliding_window", "sliding_window_size", "attention_chunk_size")
    return int(w) if w else None


def _layer_types(data: dict[str, Any], layers: int) -> tuple[str, ...] | None:
    """Work out the per-layer attention pattern, or None if uniform."""
    raw = data.get("layer_types")
    if isinstance(raw, list) and len(raw) == layers:
        kinds = tuple(_LAYER_TYPE_ALIASES.get(str(t).lower(), FULL) for t in raw)
        return kinds

    block = data.get("layers_block_type")
    if isinstance(block, list) and len(block) == layers:
        return tuple(_LAYER_TYPE_ALIASES.get(str(t).lower(), FULL) for t in block)

    # Nemotron-H: "M" = Mamba, "*" = attention, "-" = MLP only.
    pattern = data.get("hybrid_override_pattern")
    if isinstance(pattern, str) and len(pattern) == layers:
        return tuple({"M": LINEAR, "*": FULL}.get(c, NONE) for c in pattern)

    # MiMo-V2: 1 = sliding-window layer, 0 = full attention.
    hybrid = data.get("hybrid_layer_pattern")
    if isinstance(hybrid, list) and len(hybrid) == layers:
        return tuple(SLIDING if v else FULL for v in hybrid)

    # Qwen3-Next style: linear attention with a full layer every N.
    interval = data.get("full_attention_interval")
    if interval:
        n = int(interval)
        return tuple(FULL if (i + 1) % n == 0 else LINEAR for i in range(layers))

    # Jamba: one attention layer every `attn_layer_period`, rest Mamba.
    period = data.get("attn_layer_period")
    if period:
        off = int(data.get("attn_layer_offset", 0))
        return tuple(FULL if i % int(period) == off else LINEAR for i in range(layers))

    # Llama 4: chunked attention on RoPE layers, global attention on NoPE ones.
    if data.get("attention_chunk_size"):
        nope = data.get("no_rope_layers")
        if isinstance(nope, list) and len(nope) == layers:
            return tuple(CHUNKED if v else FULL for v in nope)
        step = int(data.get("no_rope_layer_interval", 4))
        return tuple(FULL if (i + 1) % step == 0 else CHUNKED for i in range(layers))

    if not _window(data) or data.get("use_sliding_window") is False:
        return None

    # Gemma 3: every Nth layer is global, the rest are sliding.
    swp = data.get("sliding_window_pattern")
    if swp:
        n = int(swp)
        return tuple(FULL if (i + 1) % n == 0 else SLIDING for i in range(layers))

    # Gemma 2: alternating sliding (even) / global (odd).
    if str(data.get("model_type", "")).startswith("gemma2"):
        return tuple(SLIDING if i % 2 == 0 else FULL for i in range(layers))

    return None  # uniform sliding window (Mistral-7B v0.1, Phi-3)


def _linear_state_bytes(data: dict[str, Any], hidden: int) -> float:
    """Recurrent state held per linear-attention / SSM layer per sequence."""
    state_dtype = str(data.get("mamba_ssm_dtype", "")).lower()
    sb = 4.0 if state_dtype in ("float32", "fp32") else 2.0
    conv_b = 2.0

    # Gated DeltaNet (Qwen3-Next, Qwen3.8): per value head a dk x dv matrix.
    if "linear_num_value_heads" in data:
        nv = int(data["linear_num_value_heads"])
        nk = int(data.get("linear_num_key_heads", nv))
        dk = int(data.get("linear_key_head_dim", 128))
        dv = int(data.get("linear_value_head_dim", 128))
        conv = int(data.get("linear_conv_kernel_dim", 4)) - 1
        return nv * dk * dv * sb + conv * (2 * nk * dk + nv * dv) * conv_b

    # Mamba-2 (Nemotron-H, Granite 4, Falcon-H1).
    heads = _first(data, "mamba_num_heads", "mamba_n_heads")
    if heads:
        hd = int(_first(data, "mamba_head_dim", "mamba_d_head") or 64)
        d_state = int(_first(data, "ssm_state_size", "mamba_state_dim", "mamba_d_state") or 128)
        groups = int(_first(data, "n_groups", "mamba_n_groups", "mamba_num_groups") or 1)
        d_conv = int(_first(data, "conv_kernel", "mamba_d_conv") or 4)
        inner = int(heads) * hd
        return inner * d_state * sb + (d_conv - 1) * (inner + 2 * groups * d_state) * conv_b

    # Mamba-1 (Jamba).
    d_state = _first(data, "mamba_d_state", "ssm_state_size", "state_size")
    if d_state:
        inner = int(data.get("mamba_expand", 2)) * hidden
        d_conv = int(_first(data, "mamba_d_conv", "conv_kernel") or 4)
        return inner * int(d_state) * sb + (d_conv - 1) * inner * conv_b
    return 0.0


def _moe_layout(data: dict[str, Any], layers: int) -> tuple[int, int, int, int, int]:
    """(n_experts, experts_per_token, expert_size, shared_size, dense_layers)."""
    n_exp = int(_first(data, "n_routed_experts", "num_local_experts", "num_experts") or 0)
    if n_exp <= 1:
        return 0, 0, 0, 0, layers
    k = int(_first(data, "num_experts_per_tok", "experts_per_token",
                   "num_experts_per_token") or 1)
    expert = int(_first(data, "moe_intermediate_size", "intermediate_size") or 0)
    shared = int(_first(data, "shared_expert_intermediate_size",
                        "shared_intermediate_size") or 0)
    if not shared and data.get("n_shared_experts"):
        shared = int(data["n_shared_experts"]) * expert
    if not shared and str(data.get("model_type", "")).startswith("llama4"):
        shared = expert  # Llama 4 always pairs the routed experts with one shared expert

    dense = 0
    mlp_types = data.get("mlp_layer_types")
    freq = data.get("moe_layer_freq")
    if isinstance(mlp_types, list):
        dense = sum(1 for t in mlp_types if str(t).lower() == "dense")
    elif isinstance(freq, list):
        dense = sum(1 for v in freq if not v)
    elif data.get("first_k_dense_replace"):
        dense = int(data["first_k_dense_replace"])
    elif isinstance(data.get("mlp_only_layers"), list):
        dense = len(data["mlp_only_layers"])
    return n_exp, k, expert, shared, min(dense, layers)


def _attn_params(data: dict[str, Any], hidden: int, heads: int, kv: int, hd: int) -> float:
    if data.get("kv_lora_rank"):
        rank = int(data["kv_lora_rank"])
        rope = int(data.get("qk_rope_head_dim", 64))
        nope = int(data.get("qk_nope_head_dim", hd))
        v = int(data.get("v_head_dim", hd))
        q_rank = data.get("q_lora_rank")
        q = (hidden * int(q_rank) + int(q_rank) * heads * (nope + rope)
             if q_rank else hidden * heads * (nope + rope))
        kv_p = hidden * (rank + rope) + rank * heads * (nope + v)
        return float(q + kv_p + heads * v * hidden)
    return float(hidden * heads * hd + 2 * hidden * kv * hd + heads * hd * hidden)


def _estimate_params(
    data: dict[str, Any], layers: int, hidden: int, heads: int, kv: int, hd: int,
) -> tuple[float, float]:
    """Rough (total, active) parameter counts from dimensions (decoder-only).

    Only used when the true count isn't known (e.g. a local config.json without
    Hub access). Accounts for GQA/MLA attention, MoE experts and tied embeddings.
    """
    attn = _attn_params(data, hidden, heads, kv, hd) * layers
    n_exp, k, expert, shared, dense_layers = _moe_layout(data, layers)
    dense_inter = int(_first(data, "intermediate_size_mlp", "intermediate_size") or 4 * hidden)
    if n_exp:
        if "moe_intermediate_size" not in data and "intermediate_size_mlp" not in data:
            dense_inter = expert
        moe_layers = layers - dense_layers
        dense_mlp = 3 * hidden * dense_inter * dense_layers
        routed = 3 * hidden * expert * moe_layers
        shared_p = 3 * hidden * shared * moe_layers
        total_mlp = dense_mlp + routed * n_exp + shared_p + hidden * n_exp * moe_layers
        active_mlp = dense_mlp + routed * k + shared_p
    else:
        total_mlp = active_mlp = 3 * hidden * dense_inter * layers
    vocab = int(data.get("vocab_size", 32000))
    embeddings = vocab * hidden * (1 if data.get("tie_word_embeddings") else 2)
    return float(attn + total_mlp + embeddings), float(attn + active_mlp + embeddings)


def _active_mlp_width(data: dict[str, Any], layers: int, hidden: int) -> int:
    n_exp, k, expert, shared, _ = _moe_layout(data, layers)
    if n_exp:
        return k * expert + shared
    return int(data.get("intermediate_size") or 4 * hidden)


def _notes(data: dict[str, Any]) -> tuple[str, ...]:
    notes: list[str] = []
    found = [desc for key, desc in _UNMODELED_KEYS.items() if key in data]
    if found:
        notes.append(
            "Uses features kvfit doesn't model exactly ("
            + ", ".join(dict.fromkeys(found))
            + "); treat the KV cache estimate as approximate."
        )
    q = data.get("quantization_config")
    if isinstance(q, dict):
        method = str(q.get("quant_method", "")).lower()
        hint = {"mxfp4": "mxfp4", "fp8": "fp8", "awq": "awq", "gptq": "gptq",
                "compressed-tensors": None, "bitsandbytes": "nf4"}.get(method)
        if hint:
            notes.append(f"Checkpoint ships {method}-quantized; try weight_dtype={hint}.")
    return tuple(notes)


def _from_hf_dict(
    name: str,
    data: dict[str, Any],
    *,
    num_params: float | None = None,
    hf_repo: str | None = None,
) -> ModelConfig:
    """Build a ModelConfig from a Hugging Face ``config.json`` dict."""
    data = _text_config(data)
    required = ("hidden_size", "num_attention_heads", "num_hidden_layers")
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(
            f"config.json is missing required fields: {', '.join(missing)}."
        )
    hidden = int(data["hidden_size"])
    heads = int(data["num_attention_heads"])
    kv_heads = int(data.get("num_key_value_heads") or heads)
    layers = int(data["num_hidden_layers"])
    head_dim = data.get("head_dim")
    hd = int(head_dim) if head_dim else hidden // heads
    is_mla = bool(data.get("kv_lora_rank"))

    layer_types = _layer_types(data, layers)
    window = _window(data)
    if layer_types is None and data.get("use_sliding_window") is False:
        window = None
    if layer_types is not None and not ({SLIDING, CHUNKED} & set(layer_types)):
        window = None
    # A window at least as long as the model's max context is effectively full.
    max_pos = data.get("max_position_embeddings")
    if layer_types is None and window and max_pos and window >= int(max_pos):
        window = None

    # Sparse-attention indexer (DeepSeek-V3.2 / GLM-5 DSA).
    indexer_dim = _first(data, "index_head_dim", "indexer_head_dim")
    indexer_layers = None
    idx_types = data.get("indexer_types")
    if indexer_dim and isinstance(idx_types, list):
        indexer_layers = sum(1 for t in idx_types if str(t).lower() == "full")

    est_total, est_active = _estimate_params(data, layers, hidden, heads, kv_heads, hd)
    notes = list(_notes(data))
    if num_params is None:
        num_params = float(data.get("num_parameters") or 0) or est_total
        if not data.get("num_parameters"):
            notes.append("Parameter count estimated from config dimensions.")
    n_exp, k_exp = _moe_layout(data, layers)[:2]
    active = None
    if n_exp:
        # Scale the estimated active share onto the (possibly exact) total.
        active = num_params * (est_active / est_total)

    v_dim = data.get("v_head_dim")
    return ModelConfig(
        name=name,
        num_layers=layers,
        hidden_size=hidden,
        num_attention_heads=heads,
        num_kv_heads=min(kv_heads, heads),
        num_params=num_params,
        head_dim=int(head_dim) if head_dim else None,
        sliding_window=window,
        layer_types=layer_types,
        v_head_dim=int(v_dim) if v_dim and not is_mla and int(v_dim) != hd else None,
        kv_lora_rank=int(data["kv_lora_rank"]) if is_mla else None,
        qk_rope_head_dim=int(data.get("qk_rope_head_dim", 64)) if is_mla else None,
        indexer_head_dim=int(indexer_dim) if indexer_dim else None,
        indexer_layers=indexer_layers,
        linear_state_bytes=(_linear_state_bytes(data, hidden)
                            if layer_types and LINEAR in layer_types else 0.0),
        num_active_params=active,
        num_experts=n_exp or None,
        experts_per_token=k_exp or None,
        intermediate_size=_active_mlp_width(data, layers, hidden),
        vocab_size=int(data["vocab_size"]) if data.get("vocab_size") else None,
        max_position_embeddings=int(max_pos) if max_pos else None,
        hf_repo=hf_repo,
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# Public resolution
# ---------------------------------------------------------------------------


def resolve_model(ref: str, *, allow_hub: bool = True) -> ModelConfig:
    """Resolve ``ref`` (registry alias, local config.json path, or hub id)."""
    key = ref.strip()
    table, _ = _registry()

    # 1) built-in registry
    if key.lower() in table:
        return table[key.lower()]

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
    """Fetch config.json (and the exact parameter count) from the Hub.

    Returns None when the hub extra isn't installed or the repo doesn't exist,
    and raises a helpful ValueError for gated repos or network failures.
    """
    try:
        from huggingface_hub import (  # type: ignore[import-not-found,unused-ignore]
            HfApi,
            hf_hub_download,
        )
    except ImportError:
        return None
    try:
        cfg_path = hf_hub_download(repo_id=repo_id, filename="config.json")
    except Exception as exc:  # huggingface_hub's error classes move between versions
        kind = type(exc).__name__
        if "Gated" in kind:
            raise ValueError(
                f"{repo_id!r} is a gated repo. Accept its license on huggingface.co "
                f"and log in (`hf auth login`), or pass a local config.json."
            ) from exc
        if kind in ("RepositoryNotFoundError", "HFValidationError", "EntryNotFoundError",
                    "RevisionNotFoundError"):
            return None
        raise ValueError(
            f"Could not fetch {repo_id!r} from the Hugging Face Hub ({kind}: {exc}). "
            f"If you're offline, pass a local config.json instead."
        ) from exc
    data = json.loads(Path(cfg_path).read_text())

    num_params: float | None = None
    try:
        info = HfApi().model_info(repo_id, expand=["safetensors"])
        st = getattr(info, "safetensors", None)
        if st is not None and getattr(st, "total", None):
            num_params = float(st.total)
    except Exception:
        pass  # fall back to the estimate from dimensions
    return _from_hf_dict(repo_id, data, num_params=num_params, hf_repo=repo_id)
