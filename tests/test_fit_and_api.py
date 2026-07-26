"""Tests for fit logic, resolution, sweep, high-level API, and CLI."""

import json

import pytest

import kvfit
from kvfit.cli import main
from kvfit.cpu import detect_cpu_ram_gib, list_cpu_presets, resolve_cpu
from kvfit.fit import check_fit, max_batch_for, max_context_for
from kvfit.gpus import resolve_gpu
from kvfit.models import Workload
from kvfit.resolver import resolve_model
from kvfit.sweep import sweep_kv_dtype


def test_small_workload_fits_big_gpu():
    r = kvfit.check("llama-3-8b", gpu="a100-80gb", context=4096, batch=1)
    assert r.fits
    assert r.headroom_gib > 0


def test_huge_workload_does_not_fit_small_gpu():
    r = kvfit.check("llama-3-70b", gpu="t4", context=8192, batch=1)
    assert not r.fits
    assert r.headroom_gib < 0
    assert r.suggestions  # should offer remedies


def test_max_context_monotonic_in_gpu_size():
    model = resolve_model("llama-3-8b")
    small = max_context_for(model, resolve_gpu("l4"))          # 24 GiB
    big = max_context_for(model, resolve_gpu("a100-80gb"))     # 80 GiB
    assert big > small > 0


def test_max_batch_decreases_with_context():
    model = resolve_model("llama-3-8b")
    gpu = resolve_gpu("a100-80gb")
    short = max_batch_for(model, gpu, context_length=2048)
    long = max_batch_for(model, gpu, context_length=32768)
    assert short > long


def test_quantizing_kv_helps_fit():
    # Pick a workload that's borderline in fp16 and check int4 lowers total.
    model = resolve_model("llama-3-70b")
    gpu = resolve_gpu("a100-80gb")
    fp16 = check_fit(model, Workload(32768, 1, kv_dtype="fp16"), gpu)
    int4 = check_fit(model, Workload(32768, 1, kv_dtype="int4"), gpu)
    assert int4.breakdown.kv_cache_bytes < fp16.breakdown.kv_cache_bytes


def test_sweep_orders_smaller_dtype_smaller_cache():
    rows = sweep_kv_dtype(
        resolve_model("llama-3-8b"),
        Workload(16384, 8),
        resolve_gpu("a100-40gb"),
    )
    caches = [r.kv_cache_bytes for r in rows]
    assert caches == sorted(caches, reverse=True)  # fp16 > fp8 > int4


def test_custom_gpu_size():
    g = resolve_gpu("mycard:48")
    assert g.memory_gib == 48
    assert g.name == "mycard"


def test_resolver_reads_local_config(tmp_path):
    cfg = {
        "num_hidden_layers": 12,
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_key_value_heads": 4,
        "intermediate_size": 3072,
        "vocab_size": 32000,
    }
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    m = resolve_model(str(p))
    assert m.num_layers == 12
    assert m.num_kv_heads == 4
    assert m.attention_kind == "GQA"


def test_resolver_unknown_raises():
    with pytest.raises(ValueError):
        resolve_model("definitely-not-a-real-model-xyz", allow_hub=False)


def test_cli_check_returns_exit_code(capsys):
    code = main(["check", "-m", "llama-3-8b", "-g", "a100-80gb",
                 "-c", "4096", "-b", "1", "--no-color", "--no-sweep"])
    out = capsys.readouterr().out
    assert code == 0
    assert "FITS" in out


def test_cli_check_nonfit_exit_code_one(capsys):
    code = main(["check", "-m", "llama-3-70b", "-g", "t4",
                 "-c", "8192", "--no-color", "--no-sweep"])
    assert code == 1


def test_cli_models_lists(capsys):
    code = main(["models"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Llama-3-8B" in out


# --- CPU / RAM target --------------------------------------------------------

def test_cpu_target_marks_kind_and_label():
    dev = resolve_cpu(32)
    assert dev.kind == "cpu"
    assert not dev.is_gpu
    assert dev.memory_label == "RAM"


def test_cpu_fit_check_small_model():
    # Phi-3-mini in int4 comfortably fits 16 GiB of RAM.
    r = kvfit.check("phi-3-mini", cpu=16, context=4096, weight_dtype="int4")
    assert r.fits
    assert r.gpu.kind == "cpu"


def test_cpu_big_model_does_not_fit_laptop():
    r = kvfit.check("llama-3-70b", cpu="laptop-16gb", context=8192)
    assert not r.fits
    assert r.suggestions


def test_resolve_cpu_forms_agree():
    assert resolve_cpu("32").memory_gib == 32
    assert resolve_cpu("cpu:32").memory_gib == 32
    assert resolve_cpu("laptop-32gb").memory_gib == 32


def test_detect_cpu_ram_returns_positive_or_none():
    ram = detect_cpu_ram_gib()
    assert ram is None or ram > 0


def test_check_requires_exactly_one_target():
    with pytest.raises(ValueError):
        kvfit.check("llama-3-8b", context=4096)  # neither gpu nor cpu
    with pytest.raises(ValueError):
        kvfit.check("llama-3-8b", gpu="t4", cpu=16, context=4096)  # both


def test_cli_check_cpu_target(capsys):
    code = main(["check", "-m", "phi-3-mini", "--cpu", "16",
                 "-c", "4096", "--weight-dtype", "int4",
                 "--no-color", "--no-sweep"])
    out = capsys.readouterr().out
    assert code == 0
    assert "RAM" in out
    assert "FITS" in out


def test_cli_cpus_lists(capsys):
    code = main(["cpus"])
    out = capsys.readouterr().out
    assert code == 0
    assert "laptop-16gb" in out


def test_cpu_presets_nonempty():
    assert "laptop-16gb" in list_cpu_presets()
