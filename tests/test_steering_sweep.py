"""`scripts/08_steering_sweep.py` testleri.

İlk 8 test (Task 4'ün ilk turu) yalnızca 4 saf yardımcıyı kapsıyordu. Fix
Round 1 (bkz. `.superpowers/sdd/p3-task-4-fix1-brief.md`) `main()`'i sahte
`load_hf_model` / `generate_steered` ile uçtan uca koşan testler ekliyor —
`tests/test_extract_axis.py` ve `tests/test_label_and_train_probe.py` ile
aynı desen (`monkeypatch` + sahte yol/veriyle `main()`'i çağırmak). Model,
GPU, ağ yok: tüm veri sentetik, tüm yollar `tmp_path`'e yönlendirilir.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_P = Path(__file__).resolve().parents[1] / "scripts" / "08_steering_sweep.py"


def _load():
    spec = importlib.util.spec_from_file_location("steering_sweep", _P)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ss = _load()


def test_module_is_registered_in_sys_modules():
    assert "steering_sweep" in sys.modules


def test_plan_counts_layers_times_strengths_times_roles_times_questions():
    n = ss.planned_generation_count(n_layers=2, n_strengths=7, n_roles=50, n_questions=5)
    assert n == 2 * 7 * 50 * 5


def test_plan_rejects_zero_dimensions():
    with pytest.raises(ValueError, match="sıfır"):
        ss.planned_generation_count(n_layers=0, n_strengths=7, n_roles=50, n_questions=5)


def test_record_carries_every_field_downstream_needs():
    r = ss.sweep_record(layer=14, strength=-0.4, role="analyst",
                        question="Who are you?", answer="I am Alex.")
    assert r == {"layer": 14, "strength": -0.4, "role": "analyst",
                 "question": "Who are you?", "answer": "I am Alex."}


def test_record_rejects_blank_answer():
    with pytest.raises(ValueError, match="boş"):
        ss.sweep_record(layer=14, strength=0.0, role="analyst",
                        question="Who are you?", answer="   ")


def test_write_is_atomic_and_leaves_no_temp(tmp_path):
    path = tmp_path / "sweep.jsonl"
    ss.write_sweep(path, [ss.sweep_record(layer=14, strength=0.0, role="r",
                                          question="q", answer="a")])
    assert [p.name for p in tmp_path.iterdir()] == ["sweep.jsonl"]


def test_write_failure_leaves_existing_file_untouched(tmp_path):
    path = tmp_path / "sweep.jsonl"
    path.write_text("ONCEKI", encoding="utf-8")

    class Boom(list):
        def __iter__(self):
            yield ss.sweep_record(layer=1, strength=0.0, role="r", question="q", answer="a")
            raise RuntimeError("bilerek")

    with pytest.raises(RuntimeError):
        ss.write_sweep(path, Boom())
    assert path.read_text(encoding="utf-8") == "ONCEKI"
    assert [p.name for p in tmp_path.iterdir()] == ["sweep.jsonl"]


def test_read_rejects_a_truncated_file(tmp_path):
    path = tmp_path / "sweep.jsonl"
    path.write_text('{"layer": 14}\n{"layer": 1', encoding="utf-8")
    with pytest.raises(ValueError, match="satır"):
        ss.read_sweep(path)


# --- F2: meta yazımı da atomik ------------------------------------------------


def test_meta_write_is_atomic_and_leaves_no_temp(tmp_path):
    path = tmp_path / "meta.json"
    ss.write_json_atomic(path, {"a": 1})
    assert [p.name for p in tmp_path.iterdir()] == ["meta.json"]
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1}


def test_meta_write_failure_leaves_existing_file_untouched(tmp_path, monkeypatch):
    path = tmp_path / "meta.json"
    path.write_text("ONCEKI", encoding="utf-8")

    def boom_replace(*_args, **_kwargs):
        raise RuntimeError("bilerek")

    monkeypatch.setattr(ss.os, "replace", boom_replace)

    with pytest.raises(RuntimeError):
        ss.write_json_atomic(path, {"a": 1})
    assert path.read_text(encoding="utf-8") == "ONCEKI"
    assert [p.name for p in tmp_path.iterdir()] == ["meta.json"]


# --- main() uçtan uca testleri: sahte load_hf_model / generate_steered -------
#
# `select_assistant_end_roles`/`role_vectors` gerçek boyut kontrolleri
# yaptığı için sabitler tutarlı olmalı: N_LAYERS her aktivasyon/eksen
# dizisinde aynı, D_MODEL de öyle.

N_LAYERS = 20
D_MODEL = 6


def _patch_paths(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    model_data = data_dir / "models" / "m"
    model_results = tmp_path / "results" / "models" / "m"
    monkeypatch.setattr(ss.config, "DATA_DIR", data_dir)
    monkeypatch.setattr(ss.config, "model_data_dir", lambda model_id=None: model_data)
    monkeypatch.setattr(ss.config, "model_results_dir", lambda model_id=None: model_results)
    return model_data, model_results / "axis"


def _write_fixture(
    tmp_path,
    monkeypatch,
    *,
    n_role_vectors: int = 5,
    n_default_rows: int = 8,
    roles_in_catalog: list[str] | None = None,
):
    """Aşama 3 artifact'lerinin (eksen, rol vektörleri, aktivasyon indeksi/
    matrisi) ve rol kataloğunun sentetik bir kopyasını `tmp_path`'e yaz."""
    model_data, axis_dir = _patch_paths(monkeypatch, tmp_path)
    model_data.mkdir(parents=True, exist_ok=True)
    axis_dir.mkdir(parents=True, exist_ok=True)
    ss.config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    axis = rng.normal(size=(N_LAYERS, D_MODEL)).astype(np.float32)
    np.save(axis_dir / "assistant_axis.npy", axis)

    names = [f"role{i}" for i in range(n_role_vectors)]
    vectors = rng.normal(size=(n_role_vectors, N_LAYERS, D_MODEL)).astype(np.float32)
    np.save(axis_dir / "role_vectors.npy", vectors)
    (axis_dir / "role_names.json").write_text(json.dumps(names), encoding="utf-8")

    rows = [{"kind": "role", "role": name} for name in names]
    rows += [{"kind": "default"} for _ in range(n_default_rows)]
    acts = rng.normal(size=(len(rows), N_LAYERS, D_MODEL)).astype(np.float32)
    np.save(model_data / "activations.npy", acts)
    (model_data / "activations_index.json").write_text(
        json.dumps({"rows": rows, "run_id": "testrun00000001"}), encoding="utf-8"
    )

    catalog_roles = roles_in_catalog if roles_in_catalog is not None else names
    (ss.config.DATA_DIR / "roles.json").write_text(
        json.dumps({
            "roles": [
                {"role": r, "instructions": [f"You are a {r}."]} for r in catalog_roles
            ]
        }),
        encoding="utf-8",
    )
    return model_data


class _FakeBundle:
    n_layers = N_LAYERS
    d_model = D_MODEL


def _fake_load_hf_model():
    return _FakeBundle()


def _fake_generate_steered(bundle, messages, *, layer, strength, layer_norm,
                            max_new_tokens, **_kwargs):
    return f"yanit L{layer} g{strength}"


def _refuse_to_load_model():
    pytest.fail("model YÜKLENMEMELİYDİ")


# --- F3: Task 3'ün yeni ValueError'ları — traceback + çıkış 1 değil, 2 -------


def test_more_roles_requested_than_exist_exits_2_not_1(tmp_path, monkeypatch, capsys):
    _write_fixture(tmp_path, monkeypatch, n_role_vectors=3)
    monkeypatch.setattr(ss, "load_hf_model", _refuse_to_load_model)

    exit_code = ss.main(["--layers", "14", "--n-roles", "200"])

    assert exit_code == 2
    err = capsys.readouterr().err
    assert "BAŞARISIZ" in err
    assert "200" in err and "3" in err
    assert "Traceback" not in err


def test_limit_roles_zero_exits_2_not_1(tmp_path, monkeypatch, capsys):
    _write_fixture(tmp_path, monkeypatch, n_role_vectors=3)
    monkeypatch.setattr(ss, "load_hf_model", _refuse_to_load_model)

    exit_code = ss.main(["--layers", "14", "--n-roles", "3", "--limit-roles", "0"])

    assert exit_code == 2
    err = capsys.readouterr().err
    assert "BAŞARISIZ" in err
    assert "sıfır" in err
    assert "Traceback" not in err


# --- F4: default satırsız/sonlu-olmayan norm → model YÜKLENMEDEN 2 ----------


def test_missing_default_rows_exits_2_before_loading_model(tmp_path, monkeypatch, capsys):
    _write_fixture(tmp_path, monkeypatch, n_role_vectors=3, n_default_rows=0)
    monkeypatch.setattr(ss, "load_hf_model", _refuse_to_load_model)

    exit_code = ss.main(["--layers", "14", "--n-roles", "3"])

    assert exit_code == 2
    err = capsys.readouterr().err
    assert "BAŞARISIZ" in err
    assert "default" in err
    assert "Traceback" not in err


def test_non_finite_layer_norm_exits_2_before_loading_model(tmp_path, monkeypatch, capsys):
    model_data = _write_fixture(tmp_path, monkeypatch, n_role_vectors=3, n_default_rows=5)
    acts_path = model_data / "activations.npy"
    acts = np.load(acts_path)
    acts[-5:, 14, :] = np.nan  # tüm 'default' satırlarında L14'ü boz
    np.save(acts_path, acts)
    monkeypatch.setattr(ss, "load_hf_model", _refuse_to_load_model)

    exit_code = ss.main(["--layers", "14", "--n-roles", "3"])

    assert exit_code == 2
    err = capsys.readouterr().err
    assert "BAŞARISIZ" in err
    assert "sonlu" in err


# --- F5: main()'in uçtan uca kapsamı ------------------------------------------


def test_main_end_to_end_happy_path_writes_expected_schema_and_meta(tmp_path, monkeypatch):
    model_data = _write_fixture(tmp_path, monkeypatch, n_role_vectors=5)
    monkeypatch.setattr(ss, "load_hf_model", _fake_load_hf_model)
    monkeypatch.setattr(ss, "generate_steered", _fake_generate_steered)

    exit_code = ss.main(["--layers", "14", "--n-roles", "5"])

    assert exit_code == 0
    total = 1 * len(ss.STRENGTHS) * 5 * len(ss.INTROSPECTIVE_QUESTIONS)
    records = ss.read_sweep(model_data / "steering_sweep.jsonl")
    assert len(records) == total
    for r in records:
        assert set(r) == {"layer", "strength", "role", "question", "answer"}

    meta = json.loads((model_data / "steering_sweep_meta.json").read_text(encoding="utf-8"))
    assert meta["planned"] == total
    assert meta["attempted"] == total
    assert meta["produced"] == total
    assert meta["complete"] is True


def test_main_reports_missing_stage3_artifacts_cleanly(tmp_path, monkeypatch, capsys):
    _patch_paths(monkeypatch, tmp_path)
    # Hiçbir artifact yazılmadı.

    exit_code = ss.main(["--layers", "14"])

    assert exit_code == 2
    err = capsys.readouterr().err
    assert "BAŞARISIZ" in err
    assert "Traceback" not in err


def test_main_rejects_out_of_range_layer(tmp_path, monkeypatch, capsys):
    _write_fixture(tmp_path, monkeypatch, n_role_vectors=3)

    exit_code = ss.main(["--layers", "999", "--n-roles", "3"])

    assert exit_code == 2
    assert "aralık dışı" in capsys.readouterr().err


def test_main_fails_when_selected_role_missing_from_catalog(tmp_path, monkeypatch, capsys):
    _write_fixture(tmp_path, monkeypatch, n_role_vectors=3, roles_in_catalog=["role0"])

    exit_code = ss.main(["--layers", "14", "--n-roles", "3"])

    assert exit_code == 2
    err = capsys.readouterr().err
    assert "katalogda yok" in err


# --- F1: artımlı kalıcılık — çökme o ana kadarki kayıtları kaybettirmez -----


def test_writes_records_incrementally_during_the_loop(tmp_path, monkeypatch):
    model_data = _write_fixture(tmp_path, monkeypatch, n_role_vectors=10)
    monkeypatch.setattr(ss, "load_hf_model", _fake_load_hf_model)
    monkeypatch.setattr(ss, "generate_steered", _fake_generate_steered)

    seen_lengths: list[int] = []
    real_write_sweep = ss.write_sweep

    def spy_write_sweep(path, records):
        seen_lengths.append(len(records))
        real_write_sweep(path, records)

    monkeypatch.setattr(ss, "write_sweep", spy_write_sweep)

    # 1 katman × 7 güç × 10 rol × 5 soru = 350 üretim -> PROGRESS_PERIOD=100
    # sınırını en az iki kez geçer, yani döngü içinde en az iki ARA yazım +
    # döngü sonunda bir final yazım olmalı.
    exit_code = ss.main(["--layers", "14", "--n-roles", "10"])

    assert exit_code == 0
    assert len(seen_lengths) >= 3
    assert seen_lengths == sorted(seen_lengths)
    assert seen_lengths[0] < seen_lengths[-1]
    assert (model_data / "steering_sweep.jsonl").exists()


def test_exception_during_generation_persists_progress_and_exits_2(
    tmp_path, monkeypatch, capsys
):
    model_data = _write_fixture(tmp_path, monkeypatch, n_role_vectors=3)
    monkeypatch.setattr(ss, "load_hf_model", _fake_load_hf_model)

    calls = {"n": 0}

    def boom_generate(bundle, messages, **kwargs):
        calls["n"] += 1
        if calls["n"] > 5:
            raise RuntimeError("simüle edilmiş CUDA OOM")
        return f"yanit {calls['n']}"

    monkeypatch.setattr(ss, "generate_steered", boom_generate)

    exit_code = ss.main(["--layers", "14", "--n-roles", "3"])

    assert exit_code == 2
    err = capsys.readouterr().err
    assert "BAŞARISIZ" in err
    assert "RuntimeError" in err
    assert "simüle edilmiş CUDA OOM" in err

    out_path = model_data / "steering_sweep.jsonl"
    assert out_path.exists()
    records = ss.read_sweep(out_path)
    assert len(records) == 5

    meta_path = model_data / "steering_sweep_meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["attempted"] == 5
    assert meta["produced"] == 5
    assert meta["complete"] is False


def test_keyboard_interrupt_preserves_todays_behavior_exit_0(tmp_path, monkeypatch):
    """F1'in Gereksinimi: `KeyboardInterrupt` bugünkü davranışını korusun —
    o ana kadarki kayıtlar yazılır ve çıkış kodu 0 kalır (Exception'dan
    farklı olarak — operatörün Ctrl-C'si bir "BAŞARISIZ" tanısına dönüşmez)."""
    model_data = _write_fixture(tmp_path, monkeypatch, n_role_vectors=3)
    monkeypatch.setattr(ss, "load_hf_model", _fake_load_hf_model)

    calls = {"n": 0}

    def interrupt_after_a_few(bundle, messages, **kwargs):
        calls["n"] += 1
        if calls["n"] > 4:
            raise KeyboardInterrupt
        return f"yanit {calls['n']}"

    monkeypatch.setattr(ss, "generate_steered", interrupt_after_a_few)

    exit_code = ss.main(["--layers", "14", "--n-roles", "3"])

    assert exit_code == 0
    records = ss.read_sweep(model_data / "steering_sweep.jsonl")
    assert len(records) == 4
    meta = json.loads(
        (model_data / "steering_sweep_meta.json").read_text(encoding="utf-8")
    )
    assert meta["attempted"] == 4
    assert meta["complete"] is False


# --- Ek gereksinim: `complete` artık `attempted == planned`, `produced` değil -


def test_complete_reflects_attempted_not_produced_when_some_answers_are_blank(
    tmp_path, monkeypatch
):
    model_data = _write_fixture(tmp_path, monkeypatch, n_role_vectors=2)
    monkeypatch.setattr(ss, "load_hf_model", _fake_load_hf_model)

    calls = {"n": 0}

    def sometimes_blank(bundle, messages, **kwargs):
        calls["n"] += 1
        return "" if calls["n"] % 7 == 0 else "yanit"

    monkeypatch.setattr(ss, "generate_steered", sometimes_blank)

    exit_code = ss.main(["--layers", "14", "--n-roles", "2"])

    assert exit_code == 0
    total = 1 * len(ss.STRENGTHS) * 2 * len(ss.INTROSPECTIVE_QUESTIONS)
    meta = json.loads(
        (model_data / "steering_sweep_meta.json").read_text(encoding="utf-8")
    )
    assert meta["attempted"] == total
    assert meta["produced"] < total
    assert meta["complete"] is True
