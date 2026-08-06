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
    # Grup içi SIRA korunur (artan indeks) ama artık dilim değil, adım
    # örneklemesi: seçilenler her grubun alt kümesidir ve kaynak sıradadır.
    assert specs[:n_role] == [s for s in role_specs if s in specs[:n_role]]
    assert set(specs[:n_role]) <= set(role_specs)
    assert set(specs[n_role:]) <= set(default_specs)


# --- C6: adım örneklemesi rol çeşitliliğini korumalı ------------------------


def test_limit_draws_roles_from_across_the_catalog_not_just_the_first():
    """`role_specs` rol-ana sıralıdır: tam ölçekte her rol 120 ardışık satır
    kaplar. Dilimleme (`role_specs[:n]`) yüzünden `--limit 100`'ün 90 rol
    satırının HEPSİ rol 0'dan geliyordu — duman testi tek bir sistem promptu
    ailesini sınıyordu. Burada 10 rol × 12 satır ile aynı yapı kuruluyor."""
    role_specs = [
        RolloutSpec(
            kind="role",
            role=f"rol{r}",
            system_prompt=f"instr {r}-{k}",
            question="q",
            sample_index=0,
        )
        for r in range(10)
        for k in range(12)
    ]
    default_specs = [_default_spec(i) for i in range(20)]

    specs, n_role, _ = gr.select_specs(role_specs, default_specs, limit=20)

    chosen_roles = {s.role for s in specs if s.kind == "role"}
    # eski dilimleme davranışı: {"rol0"} (12 satır rol0'dan, kalanı rol1'den)
    assert len(chosen_roles) >= n_role // 2
    assert "rol9" in chosen_roles  # katalogun sonuna da ulaşılmalı


def test_stride_sample_is_deterministic_and_order_preserving():
    items = list(range(100))
    first = gr.stride_sample(items, 7)
    assert first == gr.stride_sample(items, 7)  # tohumsuz, saf aritmetik
    assert first == sorted(first)  # grup içi sıra korunur
    assert len(set(first)) == 7  # tekrar yok
    assert gr.stride_sample(items, 0) == []
    assert gr.stride_sample(items, 500) == items  # n >= len: hepsi


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


# --- D3: main() kapsaması (model yok, ağ yok) --------------------------------
#
# `04`'ün `main()`'i hiç test edilmiyordu. Buradaki testler vLLM ve
# transformers'ı sys.modules üzerinden sahteleriyle değiştirir — ikisi de
# `main()` İÇİNDE import edildiği için bu yeterlidir; ne model indirilir ne
# GPU'ya dokunulur (`tests/conftest.py` zaten soketleri kapatıyor).


class _FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return " | ".join(f"{m['role']}:{m['content']}" for m in messages)


class _FakeOutput:
    def __init__(self, text: str) -> None:
        self.outputs = [type("O", (), {"text": text})()]


class _FakeLLM:
    """Her prompt için `answers` listesinden sırayla bir yanıt döndürür."""

    instances: list["_FakeLLM"] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        _FakeLLM.instances.append(self)

    def generate(self, prompts, sampling):
        return [_FakeOutput(_FakeLLM.answers[i % len(_FakeLLM.answers)]) for i in range(len(prompts))]


def _install_fake_engines(monkeypatch, answers: list[str]):
    import types

    _FakeLLM.answers = answers
    _FakeLLM.instances = []

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = _FakeLLM
    fake_vllm.SamplingParams = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = type(
        "AutoTokenizer", (), {"from_pretrained": staticmethod(lambda *a, **k: _FakeTokenizer())}
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)


def _write_inputs(tmp_path, *, n_roles: int = 3, n_questions: int = 2):
    import json as _json

    roles = {
        "complete": True,
        "limit": None,
        "requested": n_roles,
        "catalog_size": n_roles,
        "roles": [
            {
                "role": f"rol{i}",
                "description": f"the role of a rol{i}",
                "instructions": [f"You are rol{i}.", f"Act as rol{i}."],
                "questions": ["q"],
            }
            for i in range(n_roles)
        ],
    }
    (tmp_path / "roles.json").write_text(_json.dumps(roles), encoding="utf-8")
    (tmp_path / "questions.json").write_text(
        _json.dumps({"shared_questions": [f"soru {i}?" for i in range(n_questions)]}),
        encoding="utf-8",
    )


def _patch_out_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(gr.config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(gr, "OUT_PATH", tmp_path / "rollouts.jsonl")
    monkeypatch.setattr(gr, "META_PATH", tmp_path / "rollouts_meta.json")


def test_main_writes_rollouts_and_meta_for_a_full_run(tmp_path, monkeypatch, capsys):
    import json as _json

    _patch_out_paths(monkeypatch, tmp_path)
    _write_inputs(tmp_path)
    _install_fake_engines(monkeypatch, ["cevap"])

    assert gr.main([]) == 0

    records = [
        _json.loads(line)
        for line in (tmp_path / "rollouts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    meta = _json.loads((tmp_path / "rollouts_meta.json").read_text(encoding="utf-8"))
    assert meta["limit"] is None  # tam koşu
    assert meta["n"] == len(records)
    assert len(meta["run_id"]) == 16
    assert {r["kind"] for r in records} == {"role", "default"}
    assert "PİLOT" not in capsys.readouterr().out


def test_main_marks_a_limited_run_as_pilot(tmp_path, monkeypatch, capsys):
    """C5: `--limit` kanonik yola yazıyor ve dosyada bunu belli eden hiçbir
    şey yoktu. Aşama 0'ın `roles.json` zarfıyla aynı çözüm: yanına künye."""
    import json as _json

    _patch_out_paths(monkeypatch, tmp_path)
    _write_inputs(tmp_path)
    _install_fake_engines(monkeypatch, ["cevap"])

    assert gr.main(["--limit", "10"]) == 0

    meta = _json.loads((tmp_path / "rollouts_meta.json").read_text(encoding="utf-8"))
    assert meta["limit"] == 10
    assert meta["n"] == 10
    out = capsys.readouterr().out
    assert "PİLOT KOŞU" in out
    assert "--allow-pilot" in out


def test_main_skips_empty_answers_and_counts_them(tmp_path, monkeypatch, capsys):
    import json as _json

    _patch_out_paths(monkeypatch, tmp_path)
    _write_inputs(tmp_path)
    _install_fake_engines(monkeypatch, ["dolu", "   "])  # her ikinci yanıt boş

    assert gr.main([]) == 0

    records = [
        _json.loads(line)
        for line in (tmp_path / "rollouts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    meta = _json.loads((tmp_path / "rollouts_meta.json").read_text(encoding="utf-8"))
    assert all(r["answer"].strip() for r in records)
    assert meta["n"] == len(records)
    assert "boş yanıt atlandı" in capsys.readouterr().out


def test_meta_run_id_changes_with_the_rollout_set(tmp_path, monkeypatch):
    """Künye kimliği İÇERİKTEN türetilir: farklı rol kümesi -> farklı kimlik.
    `05` ve `07` bayatlık kontrollerini buna dayandırıyor."""
    import json as _json

    _patch_out_paths(monkeypatch, tmp_path)
    _write_inputs(tmp_path, n_roles=3)
    _install_fake_engines(monkeypatch, ["cevap"])
    gr.main([])
    first = _json.loads((tmp_path / "rollouts_meta.json").read_text(encoding="utf-8"))["run_id"]

    _write_inputs(tmp_path, n_roles=4)
    _install_fake_engines(monkeypatch, ["cevap"])
    gr.main([])
    second = _json.loads((tmp_path / "rollouts_meta.json").read_text(encoding="utf-8"))["run_id"]

    assert first != second
