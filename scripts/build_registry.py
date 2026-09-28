"""Regenerate src/kvfit/data/models.json from real Hugging Face configs.

Each built-in model is a trimmed copy of its published ``config.json`` (kept
under tests/fixtures/configs/) plus the exact parameter count reported by the
Hub's safetensors metadata. To add a model:

    1. curl -L https://huggingface.co/<repo>/resolve/main/config.json \
         -o tests/fixtures/configs/<org>_<name>.json
    2. Look up the parameter count:
         https://huggingface.co/api/models/<repo>?expand[]=safetensors  (-> total)
    3. Add a row to MODELS below and run:  python scripts/build_registry.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kvfit.resolver import _UNMODELED_KEYS, _text_config  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "configs"
OUT = ROOT / "src" / "kvfit" / "data" / "models.json"

# (canonical name, fixture file, exact params from HF safetensors, aliases)
MODELS: list[tuple[str, str, int, list[str]]] = [
    # --- Llama ---
    ("Llama-2-7B", "NousResearch_Llama-2-7b-hf", 6738417664, ["llama2-7b"]),
    ("Llama-2-13B", "NousResearch_Llama-2-13b-hf", 13015866880, ["llama2-13b"]),
    ("Llama-3-8B", "unsloth_llama-3-8b", 8030261248, ["llama3-8b", "meta-llama/llama-3-8b"]),
    ("Llama-3-70B", "NousResearch_Meta-Llama-3-70B", 70553706496, ["llama3-70b"]),
    ("Llama-3.1-8B", "unsloth_Llama-3.1-8B-Instruct", 8030261248,
     ["llama3.1-8b", "meta-llama/llama-3.1-8b-instruct"]),
    ("Llama-3.2-3B", "unsloth_Llama-3.2-3B-Instruct", 3212749824, ["llama3.2-3b"]),
    ("Llama-3.3-70B", "unsloth_Llama-3.3-70B-Instruct", 70553706496,
     ["llama-3.1-70b", "llama3.3-70b", "meta-llama/llama-3.3-70b-instruct"]),
    ("Llama-4-Scout", "unsloth_Llama-4-Scout-17B-16E-Instruct", 108641793536,
     ["llama-4-scout-17b-16e", "llama4-scout"]),
    # --- Mistral ---
    ("Mistral-7B-v0.1", "mistralai_Mistral-7B-v0.1", 7241732096, ["mistralai/mistral-7b-v0.1"]),
    ("Mistral-7B-v0.3", "mistralai_Mistral-7B-Instruct-v0.3", 7248023552,
     ["mistral-7b", "mistralai/mistral-7b"]),
    ("Mistral-Nemo-12B", "mistralai_Mistral-Nemo-Instruct-2407", 12247782400, ["mistral-nemo"]),
    ("Mistral-Small-3.2-24B", "mistralai_Mistral-Small-3.2-24B-Instruct-2506", 24011361280,
     ["mistral-small-3.2", "mistral-small-24b"]),
    # --- Qwen ---
    ("Qwen2.5-7B", "Qwen_Qwen2.5-7B-Instruct", 7615616512, ["qwen2-7b", "qwen-7b"]),
    ("Qwen2.5-72B", "Qwen_Qwen2.5-72B-Instruct", 72706203648, ["qwen2-72b"]),
    ("Qwen3-8B", "Qwen_Qwen3-8B", 8190735360, ["qwen/qwen3-8b"]),
    ("Qwen3-32B", "Qwen_Qwen3-32B", 32762123264, ["qwen/qwen3-32b"]),
    ("Qwen3-30B-A3B", "Qwen_Qwen3-30B-A3B", 30532122624, ["qwen/qwen3-30b-a3b"]),
    ("Qwen3-235B-A22B", "Qwen_Qwen3-235B-A22B", 235093634560, ["qwen/qwen3-235b-a22b"]),
    ("Qwen3-Next-80B-A3B", "Qwen_Qwen3-Next-80B-A3B-Instruct", 81324862720, ["qwen3-next"]),
    ("Qwen3.8-27B", "Qwen_Qwen3.8-27B", 27781427952, ["qwen/qwen3.8-27b"]),
    # --- Gemma ---
    ("Gemma-2-9B", "unsloth_gemma-2-9b-it", 9241705984, ["gemma2-9b"]),
    ("Gemma-2-27B", "unsloth_gemma-2-27b-it", 27227128320, ["gemma2-27b"]),
    ("Gemma-3-4B", "unsloth_gemma-3-4b-it", 4300079472, ["gemma3-4b"]),
    ("Gemma-3-12B", "unsloth_gemma-3-12b-it", 12187325040, ["gemma3-12b"]),
    ("Gemma-3-27B", "unsloth_gemma-3-27b-it", 27432406640, ["gemma3-27b"]),
    # --- Phi ---
    ("Phi-3-mini", "microsoft_Phi-3-mini-4k-instruct", 3821079552, ["phi-3-mini-4k", "phi3-mini"]),
    ("Phi-4", "microsoft_phi-4", 14659507200, ["phi4"]),
    ("Phi-4-mini", "microsoft_Phi-4-mini-instruct", 3836021760, ["phi4-mini"]),
    # --- MoE / MLA frontier ---
    ("DeepSeek-V3", "deepseek-ai_DeepSeek-V3", 684531386000,
     ["deepseek-r1", "deepseek-v3-0324", "deepseek-ai/deepseek-v3"]),
    ("Kimi-K2", "moonshotai_Kimi-K2-Instruct", 1026408235864, ["kimi-k2-instruct"]),
    ("gpt-oss-20b", "openai_gpt-oss-20b", 20914757184, ["openai/gpt-oss-20b"]),
    ("gpt-oss-120b", "openai_gpt-oss-120b", 116829156672, ["openai/gpt-oss-120b"]),
    ("GLM-4.5", "zai-org_GLM-4.5", 358337791296, ["zai-org/glm-4.5"]),
    ("GLM-4.5-Air", "zai-org_GLM-4.5-Air", 110468824832, ["zai-org/glm-4.5-air"]),
    ("GLM-5.3", "zai-org_GLM-5.3", 753329940480, ["zai-org/glm-5.3"]),
    # --- Hybrid SSM ---
    ("Nemotron-Nano-9B-v2", "nvidia_NVIDIA-Nemotron-Nano-9B-v2", 8888227328,
     ["nemotron-nano-9b"]),
    ("Granite-4.0-H-Small", "ibm-granite_granite-4.0-h-small", 32207337984,
     ["granite-4.0-h-small"]),
]

# Official repo ids where the fixture came from a mirror (gated upstream repos).
OFFICIAL_REPO: dict[str, str] = {
    "Llama-2-7B": "meta-llama/Llama-2-7b-hf",
    "Llama-2-13B": "meta-llama/Llama-2-13b-hf",
    "Llama-3-8B": "meta-llama/Meta-Llama-3-8B-Instruct",
    "Llama-3-70B": "meta-llama/Meta-Llama-3-70B-Instruct",
    "Llama-3.1-8B": "meta-llama/Llama-3.1-8B-Instruct",
    "Llama-3.2-3B": "meta-llama/Llama-3.2-3B-Instruct",
    "Llama-3.3-70B": "meta-llama/Llama-3.3-70B-Instruct",
    "Llama-4-Scout": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "Gemma-2-9B": "google/gemma-2-9b-it",
    "Gemma-2-27B": "google/gemma-2-27b-it",
    "Gemma-3-4B": "google/gemma-3-4b-it",
    "Gemma-3-12B": "google/gemma-3-12b-it",
    "Gemma-3-27B": "google/gemma-3-27b-it",
}

# Only the fields kvfit's resolver reads (plus markers for unmodeled features).
KEEP = {
    "model_type", "hidden_size", "num_attention_heads", "num_hidden_layers",
    "num_key_value_heads", "head_dim", "v_head_dim", "kv_lora_rank", "qk_rope_head_dim",
    "qk_nope_head_dim", "q_lora_rank", "sliding_window", "sliding_window_size",
    "attention_chunk_size", "use_sliding_window", "layer_types", "layers_block_type",
    "hybrid_override_pattern", "hybrid_layer_pattern", "full_attention_interval",
    "attn_layer_period", "attn_layer_offset", "no_rope_layers", "no_rope_layer_interval",
    "sliding_window_pattern", "mamba_ssm_dtype", "linear_num_value_heads",
    "linear_num_key_heads", "linear_key_head_dim", "linear_value_head_dim",
    "linear_conv_kernel_dim", "mamba_num_heads", "mamba_n_heads", "mamba_head_dim",
    "mamba_d_head", "ssm_state_size", "mamba_state_dim", "mamba_d_state", "n_groups",
    "mamba_n_groups", "mamba_num_groups", "conv_kernel", "mamba_d_conv", "mamba_expand",
    "state_size", "n_routed_experts", "num_local_experts", "num_experts",
    "num_experts_per_tok", "experts_per_token", "num_experts_per_token",
    "moe_intermediate_size", "intermediate_size", "intermediate_size_mlp",
    "shared_expert_intermediate_size", "shared_intermediate_size", "n_shared_experts",
    "mlp_layer_types", "moe_layer_freq", "first_k_dense_replace", "mlp_only_layers",
    "vocab_size", "tie_word_embeddings", "max_position_embeddings", "index_head_dim",
    "indexer_head_dim", "indexer_types", *_UNMODELED_KEYS,
}


def main() -> None:
    models = []
    for name, fixture, params, aliases in MODELS:
        data = _text_config(json.loads((FIXTURES / f"{fixture}.json").read_text()))
        cfg = {k: v for k, v in data.items() if k in KEEP and v is not None}
        q = data.get("quantization_config")
        if isinstance(q, dict) and q.get("quant_method"):
            cfg["quantization_config"] = {"quant_method": q["quant_method"]}
        source = fixture.replace("_", "/", 1)
        models.append({"name": name, "aliases": aliases, "num_params": params,
                       "repo": OFFICIAL_REPO.get(name, source), "config": cfg})
    OUT.write_text(json.dumps({"models": models}, indent=1) + "\n")
    print(f"wrote {len(models)} models to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
