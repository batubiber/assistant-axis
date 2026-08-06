"""`scripts/04_generate_rollouts.py` içindeki saf mantığın testleri.

Ağa çıkmaz, vLLM/transformers'a hiç dokunmaz: yalnızca `select_specs`
(role/default spec seçimi) ve `build_arg_parser` (argparse tanımı) test
edilir — ikisi de modül import edildiğinde (yani `main()` çağrılmadan) zaten
tanımlı, saf fonksiyonlardır. Script dosya adı bir rakamla başladığı için
(`04_generate_rollouts.py`) normal `import` ile içe aktarılamaz; `importlib`
ile dosya yolundan yüklenir (bkz. `tests/test_judge_gate.py`).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from aax.prompts import RolloutSpec

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "04_generate_rollouts.py"
)


def _load_script():
    spec = importlib.util.spec_from_file_location("generate_rollouts", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


gr = _load_script()


def test_module_is_registered_in_sys_modules():
    """Repo kuralı (bkz. test_judge_gate.py, test_generate_role_data.py):
    rakamla başlayan script'i importlib ile yüklerken modülü sys.modules'e
    de kaydet."""
    assert sys.modules["generate_rollouts"] is gr


# --- Bulgu 1: FlashInfer sampler'ın devre dışı bırakılması ------------------


def test_flashinfer_sampler_disabled_by_default(monkeypatch):
    """Değişken ortamda tanımlı değilse, script import edilince '0'a düşer."""
    monkeypatch.delenv("VLLM_USE_FLASHINFER_SAMPLER", raising=False)
    _load_script()  # env değişkenini görmesi için modülü taze çalıştır
    import os

    assert os.environ["VLLM_USE_FLASHINFER_SAMPLER"] == "0"


def test_flashinfer_sampler_operator_override_respected(monkeypatch):
    """Araç zinciri düzeltilmiş bir operatör kendi export'uyla geçersiz kılabilmeli."""
    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "1")
    _load_script()
    import os

    assert os.environ["VLLM_USE_FLASHINFER_SAMPLER"] == "1"


# --- Bulgu 2: --limit role/default oranını korumalı -------------------------


def _role_spec(i):
    return RolloutSpec(
        kind="role", role=f"rol{i}", system_prompt=f"instr {i}", question="q", sample_index=0
    )


def _default_spec(i):
    return RolloutSpec(
        kind="default", role=None, system_prompt=None, question="q", sample_index=i
    )


def test_small_limit_yields_both_kinds():
    # Gerçek koşuya yakın oran: 14.400 role / 1.600 default ~= 9:1.
    role_specs = [_role_spec(i) for i in range(90)]
    default_specs = [_default_spec(i) for i in range(10)]

    specs, n_role, n_default = gr.select_specs(role_specs, default_specs, limit=10)

    assert n_role > 0
    assert n_default > 0
    assert n_role + n_default == 10
    assert {s.kind for s in specs} == {"role", "default"}


def test_limit_preserves_within_group_order_and_ratio():
    role_specs = [_role_spec(i) for i in range(90)]
    default_specs = [_default_spec(i) for i in range(10)]

    specs, n_role, n_default = gr.select_specs(role_specs, default_specs, limit=20)

    # 90:10 oranı korunmalı: limit=20 -> ~18 role, ~2 default.
    assert n_role == 18
    assert n_default == 2
    assert specs[:n_role] == role_specs[:n_role]
    assert specs[n_role:] == default_specs[:n_default]


def test_limit_none_returns_full_concatenation_unchanged():
    role_specs = [_role_spec(i) for i in range(5)]
    default_specs = [_default_spec(i) for i in range(3)]

    specs, n_role, n_default = gr.select_specs(role_specs, default_specs, limit=None)

    assert specs == role_specs + default_specs
    assert (n_role, n_default) == (5, 3)


def test_limit_larger_than_total_caps_at_total():
    role_specs = [_role_spec(i) for i in range(5)]
    default_specs = [_default_spec(i) for i in range(3)]

    specs, n_role, n_default = gr.select_specs(role_specs, default_specs, limit=1000)

    assert specs == role_specs + default_specs
    assert (n_role, n_default) == (5, 3)


# --- Bulgu 4: argparse help metinleri ----------------------------------------


def test_all_generation_args_have_help_text():
    parser = gr.build_arg_parser()
    actions = {a.dest: a for a in parser._actions if a.dest != "help"}
    for dest in ("limit", "max_new_tokens", "gpu_memory_utilization", "samples_per_default_prompt"):
        assert actions[dest].help, f"--{dest} için help metni eksik"
