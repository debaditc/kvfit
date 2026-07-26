"""Tests for the core sizing math."""


import pytest

from kvfit.math_engine import (
    cached_tokens,
    estimate_memory,
    kv_bytes_per_token,
    kv_cache_bytes,
    weight_bytes,
)
from kvfit.models import ModelConfig, Workload

# A Llama-3-8B-like config (GQA: 8 KV heads on 32 query heads).
LLAMA3_8B = ModelConfig(
    "Llama-3-8B", num_layers=32, hidden_size=4096,
    num_attention_heads=32, num_kv_heads=8, num_params=8.03e9,
)
# A Llama-2-7B-like config (MHA: 32 KV heads).
LLAMA2_7B = ModelConfig(
    "Llama-2-7B", num_layers=32, hidden_size=4096,
    num_attention_heads=32, num_kv_heads=32, num_params=6.74e9,
)


def test_kv_bytes_per_token_matches_hand_calculation():
    # 2 * 32 layers * 8 kv_heads * 128 head_dim * 2 bytes (fp16)
    expected = 2 * 32 * 8 * 128 * 2
    assert kv_bytes_per_token(LLAMA3_8B, "fp16") == expected
    assert expected == 131072  # 128 KiB per token


def test_head_dim_derived_from_hidden_and_heads():
    assert LLAMA3_8B.effective_head_dim == 4096 // 32 == 128


def test_gqa_uses_less_cache_than_mha():
    gqa = kv_bytes_per_token(LLAMA3_8B, "fp16")
    mha = kv_bytes_per_token(LLAMA2_7B, "fp16")
    # 8 vs 32 KV heads -> exactly 4x difference.
    assert mha == 4 * gqa


def test_attention_kind_classification():
    assert LLAMA3_8B.attention_kind == "GQA"
    assert LLAMA2_7B.attention_kind == "MHA"
    mqa = ModelConfig("mqa", 32, 4096, 32, 1, 7e9)
    assert mqa.attention_kind == "MQA"


def test_kv_dtype_scales_linearly():
    fp16 = kv_bytes_per_token(LLAMA3_8B, "fp16")
    int8 = kv_bytes_per_token(LLAMA3_8B, "int8")
    int4 = kv_bytes_per_token(LLAMA3_8B, "int4")
    assert int8 == fp16 / 2
    assert int4 == fp16 / 4


def test_kv_cache_scales_with_context_and_batch():
    wl1 = Workload(context_length=1000, batch_size=1)
    wl2 = Workload(context_length=2000, batch_size=1)
    wl3 = Workload(context_length=1000, batch_size=4)
    base = kv_cache_bytes(LLAMA3_8B, wl1)
    assert kv_cache_bytes(LLAMA3_8B, wl2) == 2 * base
    assert kv_cache_bytes(LLAMA3_8B, wl3) == 4 * base


def test_sliding_window_caps_cached_tokens():
    windowed = ModelConfig(
        "win", num_layers=32, hidden_size=4096,
        num_attention_heads=32, num_kv_heads=8, num_params=7e9,
        sliding_window=4096,
    )
    assert cached_tokens(windowed, 2000) == 2000
    assert cached_tokens(windowed, 100000) == 4096


def test_weight_bytes():
    # 8.03e9 params * 2 bytes (fp16)
    assert weight_bytes(LLAMA3_8B, "fp16") == pytest.approx(8.03e9 * 2)
    assert weight_bytes(LLAMA3_8B, "int8") == pytest.approx(8.03e9 * 1)


def test_estimate_memory_components_sum_to_total():
    wl = Workload(context_length=8192, batch_size=16)
    mem = estimate_memory(LLAMA3_8B, wl)
    total = (mem.weights_bytes + mem.kv_cache_bytes
             + mem.activation_bytes + mem.framework_overhead_bytes)
    assert mem.total_bytes == pytest.approx(total)
    assert all(v >= 0 for v in mem.as_gib().values())


def test_invalid_dtype_raises():
    with pytest.raises(ValueError):
        kv_bytes_per_token(LLAMA3_8B, "fp13")
