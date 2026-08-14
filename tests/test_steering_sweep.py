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


def _new_recording_fake():
    """`generate_steered`'ın gerçek imzasıyla (`src/aax/steering.py:158-166`)
    BİREBİR aynı keyword-only parametreleri alan bir sahte döndürür —
    `**_kwargs` YOKTUR. Eski sahte (`**_kwargs` yutan) bir çağrı sitesinden
    `direction=direction` gibi bir keyword'ün SİLİNMESİNİ fark edemiyordu;
    bu sahte onu `TypeError` ile yakalar (bkz. Fix Round 2 brief, M4+M5).

    Ayrıca her çağrıyı `.calls`'a kaydeder ki testler üretici fonksiyona
    ulaşan GERÇEK (layer, strength, layer_norm, direction) değerlerini
    doğrulayabilsin — eski sahte yalnızca `layer, strength`'i taşıyan sabit
    bir string döndürüyordu, `direction`/`layer_norm`'un doğru mu yanlış mı
    geçtiğini hiçbir test göremiyordu.
    """
    calls: list[dict] = []

    def fake(bundle, messages, *, layer, direction, strength, layer_norm,
              max_new_tokens=120):
        calls.append({
            "layer": layer,
            "direction": np.asarray(direction),
            "strength": strength,
            "layer_norm": layer_norm,
            "messages": messages,
            "max_new_tokens": max_new_tokens,
        })
        return f"yanit L{layer} g{strength}"

    fake.calls = calls
    return fake


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
    monkeypatch.setattr(ss, "generate_steered", _new_recording_fake())

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
    monkeypatch.setattr(ss, "generate_steered", _new_recording_fake())

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


# --- Fix Round 2, M1: main() öngörülmemiş bir I/O hatasını sarmalar --------
#
# `.superpowers/sdd/p3-task-4-fix2-brief.md`: `write_sweep`/`write_json_atomic`
# tam diskte (ENOSPC) sarmasız fırlarsa, çıplak Python traceback + çıkış
# kodu 1 ile çıkardı — bu projede 1 "kriter değerlendirildi ve düştü" demek.
# `main()` artık `07_extract_axis.py:609-637`'deki desenle bunu yakalayıp
# temiz bir Türkçe teşhisle çıkış 2 döner.


def test_main_wraps_unexpected_write_failure_as_exit_2_not_a_bare_traceback(
    tmp_path, monkeypatch, capsys
):
    _write_fixture(tmp_path, monkeypatch, n_role_vectors=3)
    monkeypatch.setattr(ss, "load_hf_model", _fake_load_hf_model)
    monkeypatch.setattr(ss, "generate_steered", _new_recording_fake())

    def boom_write_sweep(*_args, **_kwargs):
        raise OSError("[Errno 28] No space left on device (simüle)")

    monkeypatch.setattr(ss, "write_sweep", boom_write_sweep)

    exit_code = ss.main(["--layers", "14", "--n-roles", "3"])

    assert exit_code == 2
    err = capsys.readouterr().err
    assert "BAŞARISIZ" in err
    assert "Traceback" not in err


def test_main_lets_keyboard_interrupt_that_escapes_run_propagate(monkeypatch):
    """`main()`'in `except Exception` bloğu `KeyboardInterrupt`'ı (bir
    `BaseException`) hiç yakalamamalı — sarmasız bile geçseydi bu zaten
    doğru olurdu, ama sarmalayıcının `except KeyboardInterrupt: raise`
    satırı BİLEREK açık: operatörün Ctrl-C'si bir "BAŞARISIZ" tanısına asla
    dönüşmemeli."""

    def boom_run(argv=None):
        raise KeyboardInterrupt

    monkeypatch.setattr(ss, "_run", boom_run)

    with pytest.raises(KeyboardInterrupt):
        ss.main(["--layers", "14"])


# --- Fix Round 2, M2: meta döngü öncesi + her periyodik write ile tazelenir -


def test_meta_is_written_before_the_loop_and_at_each_periodic_flush(
    tmp_path, monkeypatch
):
    """M2: meta artık (a) döngü BAŞLAMADAN ÖNCE bir kez ve (b) her periyodik
    `write_sweep`'in yanında `complete: false` ile yazılıyor. Önceki
    davranışta meta yalnızca döngü SONUNDA yazıldığı için bir SIGKILL/
    elektrik kesintisi taze bir kısmi `.jsonl`'in yanına ÖNCEKİ koşunun
    meta'sını bırakabiliyordu."""
    model_data = _write_fixture(tmp_path, monkeypatch, n_role_vectors=10)
    monkeypatch.setattr(ss, "load_hf_model", _fake_load_hf_model)
    monkeypatch.setattr(ss, "generate_steered", _new_recording_fake())

    seen: list[dict] = []
    real_write_json_atomic = ss.write_json_atomic

    def spy_write_json_atomic(path, payload):
        seen.append(dict(payload))
        real_write_json_atomic(path, payload)

    monkeypatch.setattr(ss, "write_json_atomic", spy_write_json_atomic)

    # 1 katman × 7 güç × 10 rol × 5 soru = 350 üretim -> PROGRESS_PERIOD=100
    # sınırını done=100/200/300'de geçer: 1 döngü-öncesi + 3 periyodik +
    # 1 final = 5 meta yazımı.
    exit_code = ss.main(["--layers", "14", "--n-roles", "10"])

    assert exit_code == 0
    assert len(seen) == 5
    assert seen[0]["attempted"] == 0
    assert seen[0]["complete"] is False
    for payload in seen[:-1]:
        assert payload["complete"] is False
    attempted_seq = [p["attempted"] for p in seen]
    assert attempted_seq == sorted(attempted_seq)
    assert seen[-1]["complete"] is True
    assert seen[-1]["attempted"] == 350
    meta_on_disk = json.loads(
        (model_data / "steering_sweep_meta.json").read_text(encoding="utf-8")
    )
    assert meta_on_disk == seen[-1]


# --- Fix Round 2, M3: mevcut bir sweep sessizce ezilmeden önce kenara alınır -


def test_existing_sweep_is_archived_before_being_overwritten(
    tmp_path, monkeypatch, capsys
):
    model_data = _write_fixture(tmp_path, monkeypatch, n_role_vectors=3)
    monkeypatch.setattr(ss, "load_hf_model", _fake_load_hf_model)
    monkeypatch.setattr(ss, "generate_steered", _new_recording_fake())

    out_path = model_data / "steering_sweep.jsonl"
    ss.write_sweep(out_path, [
        ss.sweep_record(layer=1, strength=0.0, role="r", question="q", answer="ONCEKI KOŞU")
    ])

    exit_code = ss.main(["--layers", "14", "--n-roles", "3"])

    assert exit_code == 0
    prev_path = model_data / "steering_sweep.jsonl.prev"
    assert prev_path.exists()
    prev_records = ss.read_sweep(prev_path)
    assert len(prev_records) == 1
    assert prev_records[0]["answer"] == "ONCEKI KOŞU"

    err = capsys.readouterr().err
    assert "UYARI" in err
    assert "1" in err
    assert str(prev_path) in err

    # Yeni koşu hedef dosyanın üzerine yazmış olmalı (sessizce EZMEMİŞ, ama
    # tam resume KAPSAM DIŞI — yeni koşu KENDİ kayıtlarını üretir).
    new_records = ss.read_sweep(out_path)
    assert all(r["answer"] != "ONCEKI KOŞU" for r in new_records)


def test_no_prior_sweep_means_no_prev_file_and_no_warning(tmp_path, monkeypatch, capsys):
    _write_fixture(tmp_path, monkeypatch, n_role_vectors=3)
    monkeypatch.setattr(ss, "load_hf_model", _fake_load_hf_model)
    monkeypatch.setattr(ss, "generate_steered", _new_recording_fake())

    exit_code = ss.main(["--layers", "14", "--n-roles", "3"])

    assert exit_code == 0
    err = capsys.readouterr().err
    assert "UYARI" not in err


# --- Fix Round 2, M4+M5: testler steering'in FİİLEN uygulandığını sabitler -
#
# Mutasyonla doğrulandı (brief): `strength=strength` → `strength=0.0`
# (steering'i kapatmak), `direction=direction` satırını SİLMEK, ve
# `strength=strength` → `STRENGTHS[0]` (kayıt alanını sabitlemek) — üçü de
# eski (kwargs-yutan, çağrı kaydetmeyen) sahteyle 22/22 testten GEÇİYORDU.
# Aşağıdaki testler `_new_recording_fake()`'in kaydettiği ÇAĞRILARI ve
# üretilen KAYITLARI karşılaştırarak bu boşluğu kapatır.


def test_generator_receives_exactly_the_full_strength_set(tmp_path, monkeypatch):
    _write_fixture(tmp_path, monkeypatch, n_role_vectors=5)
    monkeypatch.setattr(ss, "load_hf_model", _fake_load_hf_model)
    gen = _new_recording_fake()
    monkeypatch.setattr(ss, "generate_steered", gen)

    exit_code = ss.main(["--layers", "14", "--n-roles", "5"])

    assert exit_code == 0
    assert {c["strength"] for c in gen.calls} == set(ss.STRENGTHS)


def test_layer_norm_passed_to_generator_matches_that_calls_own_layer(
    tmp_path, monkeypatch
):
    """`layer_norms[layer]` yerine `layer_norms[args.layers[0]]` gibi tek bir
    katmana sabitleyen bir mutasyonu yakalar — bunu görebilmek için EN AZ
    iki farklı katman gerekir (`--layers 14 5`)."""
    model_data = _write_fixture(tmp_path, monkeypatch, n_role_vectors=5)
    monkeypatch.setattr(ss, "load_hf_model", _fake_load_hf_model)
    gen = _new_recording_fake()
    monkeypatch.setattr(ss, "generate_steered", gen)

    acts = np.load(model_data / "activations.npy")
    index = json.loads(
        (model_data / "activations_index.json").read_text(encoding="utf-8")
    )
    default_rows = [i for i, r in enumerate(index["rows"]) if r["kind"] == "default"]
    expected = {
        L: ss.mean_residual_norm(acts[default_rows[:1000]], L) for L in (14, 5)
    }
    assert expected[14] != expected[5]  # sağlama: fikstür ayırt edilebilir olmalı

    exit_code = ss.main(["--layers", "14", "5", "--n-roles", "5"])

    assert exit_code == 0
    assert {c["layer"] for c in gen.calls} == {14, 5}
    for call in gen.calls:
        assert call["layer_norm"] == pytest.approx(expected[call["layer"]])


def test_direction_passed_to_generator_matches_axis_of_that_calls_own_layer(
    tmp_path, monkeypatch
):
    """`direction=direction` satırının SİLİNMESİNİ (gerçek fonksiyonun
    keyword-only ZORUNLU parametresi) `TypeError` ile, `axis[args.layers[0]]`
    gibi tek bir katmana sabitlemeyi ise aşağıdaki karşılaştırmayla yakalar."""
    _write_fixture(tmp_path, monkeypatch, n_role_vectors=5)
    monkeypatch.setattr(ss, "load_hf_model", _fake_load_hf_model)
    gen = _new_recording_fake()
    monkeypatch.setattr(ss, "generate_steered", gen)

    axis = np.load(ss.config.model_results_dir() / "axis" / "assistant_axis.npy")
    assert not np.allclose(axis[14], axis[5])  # sağlama: ayırt edilebilir olmalı

    exit_code = ss.main(["--layers", "14", "5", "--n-roles", "5"])

    assert exit_code == 0
    assert {c["layer"] for c in gen.calls} == {14, 5}
    for call in gen.calls:
        np.testing.assert_allclose(call["direction"], axis[call["layer"]])


def test_record_strength_field_matches_the_strength_that_actually_generated_it(
    tmp_path, monkeypatch
):
    """`sweep_record(..., strength=strength, ...)`'daki `strength=strength`'in
    `STRENGTHS[0]` gibi sabit bir değere değiştirilmesini yakalar: kayıt
    şeması ve sayısı bu mutasyon altında da doğru kalır, ama TEK BİR alan
    DEĞERİ (`strength`) üretici çağrısının gerçek gücüyle uyuşmaz olur."""
    model_data = _write_fixture(tmp_path, monkeypatch, n_role_vectors=5)
    monkeypatch.setattr(ss, "load_hf_model", _fake_load_hf_model)
    gen = _new_recording_fake()
    monkeypatch.setattr(ss, "generate_steered", gen)

    exit_code = ss.main(["--layers", "14", "--n-roles", "5"])

    assert exit_code == 0
    records = ss.read_sweep(model_data / "steering_sweep.jsonl")
    # Sahte hiçbir zaman boş yanıt döndürmez, yani her çağrı tam olarak bir
    # kayıt üretir ve sıra korunur (itertools.product'ın ürettiği sırayla).
    assert len(records) == len(gen.calls)
    for record, call in zip(records, gen.calls):
        # Cevap metni üretici çağrısının GERÇEK gücünü taşır (sahtenin
        # döndürdüğü string, `call["strength"]`'ten üretildi); kayıt alanı
        # bununla uyuşmalı.
        assert record["answer"] == f"yanit L{call['layer']} g{call['strength']}"
        assert record["strength"] == call["strength"]


# --- Fix Round 3: role ekseni çökmesini yakalar - system prompt ve soru ------
#
# Reviewer'ın uyguladığı mutasyon: `catalog[role]` → `catalog[role_keys[0]]`.
# Bu, her üretimi ilk rolün system prompt'uyla yapar, role ekseni çöker. Eski
# sahte (`**_kwargs` yutan) bunu göremezdi çünkü mesajları kaydetmiyordu ve
# yapılan kayıtlara bakıp sistem promptuyla ilişki kuramıyordu. Aşağıdaki
# testler `.calls`'taki kaydedilmiş `messages` ve record'ların `role` alanıyla
# birlikte ilişkiyi kontrol eder.


def test_generator_receives_system_prompt_for_each_selected_role(tmp_path, monkeypatch):
    """Üretici her seçili rol için o rolün instructions[0]'ını almalı.
    Sistem promptu adı role tarafından belirlenmiş şekilde codlanmıştır:
    `catalog[role]` tam bir role adı alır, yani prompt seti `catalog[role]`'den,
    `catalog[role_keys[0]]`'dan değil gelir."""
    _write_fixture(tmp_path, monkeypatch, n_role_vectors=5)
    monkeypatch.setattr(ss, "load_hf_model", _fake_load_hf_model)
    gen = _new_recording_fake()
    monkeypatch.setattr(ss, "generate_steered", gen)

    exit_code = ss.main(["--layers", "14", "--n-roles", "5"])

    assert exit_code == 0
    # Seçili roller: role0, role1, role2, role3, role4
    expected_prompts = {f"You are a role{i}." for i in range(5)}
    # Her çağrının messages[0]["content"] bir beklenen prompt olmalı
    actual_prompts = {call["messages"][0]["content"] for call in gen.calls}
    assert actual_prompts == expected_prompts


def test_generator_receives_all_introspective_questions(tmp_path, monkeypatch):
    """Üretici tüm INTROSPECTIVE_QUESTIONS değerlerini almalı. messages[1]['content']
    (user message) kümesi tam olarak INTROSPECTIVE_QUESTIONS olmalı."""
    _write_fixture(tmp_path, monkeypatch, n_role_vectors=5)
    monkeypatch.setattr(ss, "load_hf_model", _fake_load_hf_model)
    gen = _new_recording_fake()
    monkeypatch.setattr(ss, "generate_steered", gen)

    exit_code = ss.main(["--layers", "14", "--n-roles", "5"])

    assert exit_code == 0
    # Her çağrının messages[1]["content"] bir INTROSPECTIVE_QUESTION olmalı
    actual_questions = {call["messages"][1]["content"] for call in gen.calls}
    assert actual_questions == set(ss.INTROSPECTIVE_QUESTIONS)


def test_system_prompt_and_question_pairs_form_full_cross_product(
    tmp_path, monkeypatch
):
    """(system_prompt, question) çiftleri tam cross product oluşturmalı:
    her role × her soru kombinasyonu en az bir kez görülmeli (beri verilen kısmı
    layer × strength tarafsız geçtiği için)."""
    _write_fixture(tmp_path, monkeypatch, n_role_vectors=3)
    monkeypatch.setattr(ss, "load_hf_model", _fake_load_hf_model)
    gen = _new_recording_fake()
    monkeypatch.setattr(ss, "generate_steered", gen)

    exit_code = ss.main(["--layers", "14", "--n-roles", "3"])

    assert exit_code == 0
    # Çiftler: system_prompt (messages[0]["content"]) ve question (messages[1]["content"])
    actual_pairs = {
        (call["messages"][0]["content"], call["messages"][1]["content"])
        for call in gen.calls
    }
    # Beklenen: {f"You are a role{i}."} × INTROSPECTIVE_QUESTIONS
    expected_pairs = {
        (f"You are a role{i}.", q)
        for i in range(3)
        for q in ss.INTROSPECTIVE_QUESTIONS
    }
    assert actual_pairs == expected_pairs


def test_role_field_in_record_matches_system_prompt_used(tmp_path, monkeypatch):
    """`catalog[role]` yerine `catalog[role_keys[0]]` gibi bir sabitlemesi
    katça yakalar: kayıtlar hâlâ role bilgisini taşır ama sistem prompt'u
    farklı bir rolün olur. Burada üretici sahte her çağrıyı sırası ile işliyor
    (`itertools.product`'ın sırasıyla kayıtlar sırasıyla eşleşir), ve her
    kayıt'ın role alanı o kayıt'ı üreten çağrı'nın system prompt'uyla uyuşmalı."""
    model_data = _write_fixture(tmp_path, monkeypatch, n_role_vectors=3)
    monkeypatch.setattr(ss, "load_hf_model", _fake_load_hf_model)
    gen = _new_recording_fake()
    monkeypatch.setattr(ss, "generate_steered", gen)

    exit_code = ss.main(["--layers", "14", "--n-roles", "3"])

    assert exit_code == 0
    records = ss.read_sweep(model_data / "steering_sweep.jsonl")
    # Sahte hiçbir zaman boş yanıt döndürmez, yani her çağrı tam olarak bir
    # kayıt üretir ve sıra korunur (itertools.product'ın ürettiği sırayla).
    assert len(records) == len(gen.calls)
    for record, call in zip(records, gen.calls):
        # call["messages"][0]["content"] = "You are a role{X}."
        # record["role"] = "role{X}"
        # Bunlar uyuşmalı
        system_prompt = call["messages"][0]["content"]
        expected_role = system_prompt.replace("You are a ", "").replace(".", "")
        assert record["role"] == expected_role


# --- Task 2: --direction / --seed / --variant --------------------------------
#
# `.superpowers/sdd/p4-task-2-brief.md`: aynı script'i
# `results/control_preregistration.json`'daki kontrol koşusu için de
# kullanabilmek. En üstteki kısıt: `axis` koşusunun davranışı (artefakt
# adları, meta içeriği — üç yeni alan hariç, üretim yolu, çıkış kodları)
# hiçbir koşulda değişmemeli; yukarıdaki 35 test bunu zaten sabitliyor.


def test_variant_writes_suffixed_names_and_leaves_default_names_untouched(
    tmp_path, monkeypatch
):
    """`--variant ctrl` verilince artefaktlar `steering_sweep_ctrl.jsonl` /
    `steering_sweep_ctrl_meta.json` olarak yazılmalı; bugünkü
    `steering_sweep.jsonl` / `steering_sweep_meta.json` adları hiç
    dokunulmadan kalmalı (var olsalar bile)."""
    model_data = _write_fixture(tmp_path, monkeypatch, n_role_vectors=3)
    monkeypatch.setattr(ss, "load_hf_model", _fake_load_hf_model)
    monkeypatch.setattr(ss, "generate_steered", _new_recording_fake())

    # Committed Aşama 4 sweep'ini taklit eden var olan dosyalar — variant
    # koşusu bunlara HİÇ dokunmamalı.
    default_out = model_data / "steering_sweep.jsonl"
    default_meta = model_data / "steering_sweep_meta.json"
    default_out.write_text('{"ONCEKI": true}\n', encoding="utf-8")
    default_meta.write_text('{"ONCEKI": true}', encoding="utf-8")

    exit_code = ss.main(["--layers", "14", "--n-roles", "3", "--variant", "ctrl"])

    assert exit_code == 0
    variant_out = model_data / "steering_sweep_ctrl.jsonl"
    variant_meta = model_data / "steering_sweep_ctrl_meta.json"
    assert variant_out.exists()
    assert variant_meta.exists()
    # Bugünkü adlar BİREBİR aynı içerikle kalmış olmalı — hiç yazılmamış.
    assert default_out.read_text(encoding="utf-8") == '{"ONCEKI": true}\n'
    assert default_meta.read_text(encoding="utf-8") == '{"ONCEKI": true}'
    # Ve variant'ın kendi `.prev` arşivi de olmamalı (var olan bir şeyin
    # üzerine yazmadı, o yüzden arşivlenecek bir şey yoktu).
    assert not (model_data / "steering_sweep_ctrl.jsonl.prev").exists()


def test_control_direction_without_variant_exits_2_and_writes_nothing(
    tmp_path, monkeypatch, capsys
):
    """`--direction gaussian` `--variant` OLMADAN verilirse çıkış 2 dönmeli
    ve hiçbir artefakt (Stage 3 artifact'leri bile OKUNMADAN) yazılmamalı —
    bir kontrol koşusu committed Aşama 4 sweep'ini asla ezmemeli."""
    model_data = _write_fixture(tmp_path, monkeypatch, n_role_vectors=3)
    monkeypatch.setattr(ss, "load_hf_model", _refuse_to_load_model)

    exit_code = ss.main(["--layers", "14", "--n-roles", "3", "--direction", "gaussian"])

    assert exit_code == 2
    err = capsys.readouterr().err
    assert "BAŞARISIZ" in err
    assert "--variant" in err
    assert "Traceback" not in err
    assert not (model_data / "steering_sweep.jsonl").exists()
    assert not (model_data / "steering_sweep_meta.json").exists()


def test_meta_gains_three_direction_fields_axis_run_has_null_seed(
    tmp_path, monkeypatch
):
    """Her koşuda meta'ya üç alan eklenir: `direction_kind`, `direction_seed`,
    `direction_sha256`. `axis` koşusunda `direction_kind == "axis"` ve
    `direction_seed` her zaman `null` (--seed verilse bile — tohum yalnızca
    kontrol yönleri için anlamlı)."""
    model_data = _write_fixture(tmp_path, monkeypatch, n_role_vectors=3)
    monkeypatch.setattr(ss, "load_hf_model", _fake_load_hf_model)
    monkeypatch.setattr(ss, "generate_steered", _new_recording_fake())

    exit_code = ss.main(["--layers", "14", "--n-roles", "3", "--seed", "7"])

    assert exit_code == 0
    meta = json.loads((model_data / "steering_sweep_meta.json").read_text(encoding="utf-8"))
    assert meta["direction_kind"] == "axis"
    assert meta["direction_seed"] is None
    assert isinstance(meta["direction_sha256"], str) and len(meta["direction_sha256"]) == 16

    axis = np.load(ss.config.model_results_dir() / "axis" / "assistant_axis.npy")
    expected_sha = ss.direction_fingerprint(np.stack([axis[14]]))
    assert meta["direction_sha256"] == expected_sha


def test_control_direction_seed_recorded_and_matches_control_direction(
    tmp_path, monkeypatch
):
    """Kontrol koşusunda `direction_kind`/`direction_seed` verilen değerleri
    taşımalı ve `direction_sha256`, `aax.controls.control_direction` ile
    (TÜM rol vektörleriyle — yalnızca seçili roller değil) bağımsızca
    hesaplanan yönün parmak iziyle eşleşmeli."""
    _write_fixture(tmp_path, monkeypatch, n_role_vectors=5)
    monkeypatch.setattr(ss, "load_hf_model", _fake_load_hf_model)
    monkeypatch.setattr(ss, "generate_steered", _new_recording_fake())

    exit_code = ss.main([
        "--layers", "14", "--n-roles", "5",
        "--direction", "gaussian", "--seed", "3", "--variant", "ctrl",
    ])

    assert exit_code == 0
    model_data = ss.config.model_data_dir()
    meta = json.loads(
        (model_data / "steering_sweep_ctrl_meta.json").read_text(encoding="utf-8")
    )
    assert meta["direction_kind"] == "gaussian"
    assert meta["direction_seed"] == 3

    axis = np.load(ss.config.model_results_dir() / "axis" / "assistant_axis.npy")
    vectors = np.load(ss.config.model_results_dir() / "axis" / "role_vectors.npy")
    expected_direction = ss.control_direction(
        "gaussian", axis_layer=axis[14], role_vectors_layer=vectors[:, 14, :], seed=3,
    )
    expected_sha = ss.direction_fingerprint(np.stack([expected_direction]))
    assert meta["direction_sha256"] == expected_sha


def test_same_seed_gives_same_sha_different_seed_gives_different_sha(
    tmp_path, monkeypatch
):
    """Aynı tohum aynı `direction_sha256`'yı vermeli; farklı tohum farklı
    (yeniden üretilebilirlik ön kayıtta zorunlu — bkz. control_preregistration.json)."""
    def _run_with_seed(seed, variant):
        _write_fixture(tmp_path / variant, monkeypatch, n_role_vectors=5)
        monkeypatch.setattr(ss, "load_hf_model", _fake_load_hf_model)
        monkeypatch.setattr(ss, "generate_steered", _new_recording_fake())
        exit_code = ss.main([
            "--layers", "14", "--n-roles", "5",
            "--direction", "gaussian", "--seed", str(seed), "--variant", variant,
        ])
        assert exit_code == 0
        model_data = ss.config.model_data_dir()
        meta = json.loads(
            (model_data / f"steering_sweep_{variant}_meta.json").read_text(encoding="utf-8")
        )
        return meta["direction_sha256"]

    sha_seed0 = _run_with_seed(0, "a")
    sha_seed0_again = _run_with_seed(0, "b")
    sha_seed1 = _run_with_seed(1, "c")

    assert sha_seed0 == sha_seed0_again
    assert sha_seed0 != sha_seed1


def test_control_direction_reaches_generator_and_differs_from_axis(
    tmp_path, monkeypatch
):
    """Kontrol yönünün `generate_steered`'a FİİLEN ulaştığını sabitler —
    önceki turda bir mutasyon steering'i devre dışı bırakıp tüm testlerden
    geçmişti (bkz. modül docstring'i, Fix Round 2, M4+M5). Kaydeden sahteyle
    `direction` argümanını yakalayıp hem beklenen kontrol yönüyle eşit hem de
    eksenden FARKLI olduğunu doğrular."""
    _write_fixture(tmp_path, monkeypatch, n_role_vectors=5)
    monkeypatch.setattr(ss, "load_hf_model", _fake_load_hf_model)
    gen = _new_recording_fake()
    monkeypatch.setattr(ss, "generate_steered", gen)

    exit_code = ss.main([
        "--layers", "14", "--n-roles", "5",
        "--direction", "shuffled", "--seed", "0", "--variant", "ctrl",
    ])
    assert exit_code == 0
    assert len(gen.calls) > 0

    axis = np.load(ss.config.model_results_dir() / "axis" / "assistant_axis.npy")
    vectors = np.load(ss.config.model_results_dir() / "axis" / "role_vectors.npy")
    expected_direction = ss.control_direction(
        "shuffled", axis_layer=axis[14], role_vectors_layer=vectors[:, 14, :], seed=0,
    )

    for call in gen.calls:
        np.testing.assert_allclose(call["direction"], expected_direction)
        # Kontrol yönü eksenden GERÇEKTEN farklı olmalı — aksi hâlde bu test
        # steering'in devre dışı kaldığı bir mutasyonu yakalayamaz.
        assert not np.allclose(call["direction"], axis[14])


def test_default_run_still_writes_axis_direction_and_default_names(
    tmp_path, monkeypatch
):
    """Sağlama: `--direction`/`--variant` hiç verilmeyince (bugünkü tek
    kullanım biçimi) `direction` çağrıya eksenin KENDİSİ olarak ulaşır ve
    artefaktlar bugünkü adlarla yazılır — üstteki 35 test bunu zaten
    kapsıyor, burası yalnızca yeni `directions` sözlüğünün axis-kolunu
    (`directions[layer] = axis[layer]`) doğrudan hedefler."""
    model_data = _write_fixture(tmp_path, monkeypatch, n_role_vectors=5)
    monkeypatch.setattr(ss, "load_hf_model", _fake_load_hf_model)
    gen = _new_recording_fake()
    monkeypatch.setattr(ss, "generate_steered", gen)

    exit_code = ss.main(["--layers", "14", "--n-roles", "5"])

    assert exit_code == 0
    axis = np.load(ss.config.model_results_dir() / "axis" / "assistant_axis.npy")
    for call in gen.calls:
        np.testing.assert_allclose(call["direction"], axis[14])
    assert (model_data / "steering_sweep.jsonl").exists()
    assert (model_data / "steering_sweep_meta.json").exists()


def test_strengths_default_is_the_susceptibility_strengths_grid(tmp_path, monkeypatch):
    """`--strengths` verilmezse varsayılan `aax.susceptibility.STRENGTHS`
    olmalı — hem üretici fonksiyona ulaşan güç kümesinde hem de meta'da."""
    model_data = _write_fixture(tmp_path, monkeypatch, n_role_vectors=3)
    monkeypatch.setattr(ss, "load_hf_model", _fake_load_hf_model)
    gen = _new_recording_fake()
    monkeypatch.setattr(ss, "generate_steered", gen)

    exit_code = ss.main(["--layers", "14", "--n-roles", "3"])

    assert exit_code == 0
    assert {c["strength"] for c in gen.calls} == set(ss.STRENGTHS)
    meta = json.loads((model_data / "steering_sweep_meta.json").read_text(encoding="utf-8"))
    assert meta["strengths"] == list(ss.STRENGTHS)


def test_strengths_flag_is_honored_by_generator_and_meta(tmp_path, monkeypatch):
    """`--strengths` verilince üretici fonksiyona ulaşan güç kümesi TAM
    olarak verilen kümeye eşit olmalı — ön-tescilin 3 gücü (-0.6 -0.4 -0.2),
    varsayılanın 7'si değil. Aksi hâlde kontrol koşusu 225 yerine 525 hakem
    çağrısına mal olurdu (bkz. control_preregistration.json)."""
    model_data = _write_fixture(tmp_path, monkeypatch, n_role_vectors=3)
    monkeypatch.setattr(ss, "load_hf_model", _fake_load_hf_model)
    gen = _new_recording_fake()
    monkeypatch.setattr(ss, "generate_steered", gen)

    exit_code = ss.main([
        "--layers", "14", "--n-roles", "3",
        "--strengths", "-0.6", "-0.4", "-0.2",
    ])

    assert exit_code == 0
    assert {c["strength"] for c in gen.calls} == {-0.6, -0.4, -0.2}
    meta = json.loads((model_data / "steering_sweep_meta.json").read_text(encoding="utf-8"))
    assert meta["strengths"] == [-0.6, -0.4, -0.2]
    total = 1 * 3 * 3 * len(ss.INTROSPECTIVE_QUESTIONS)
    records = ss.read_sweep(model_data / "steering_sweep.jsonl")
    assert len(records) == total


def test_invalid_strengths_value_fails_cleanly(tmp_path, monkeypatch, capsys):
    """Sayısal olmayan bir `--strengths` değeri ('abc') temiz bir başarısızlıkla
    (çıkış 2, çıplak Python traceback YOK) sonuçlanmalı — argparse'ın kendi
    `type=float` doğrulaması bunu üretim döngüsüne hiç girmeden yakalar."""
    _write_fixture(tmp_path, monkeypatch, n_role_vectors=3)
    monkeypatch.setattr(ss, "load_hf_model", _refuse_to_load_model)

    with pytest.raises(SystemExit) as exc_info:
        ss.main(["--layers", "14", "--n-roles", "3", "--strengths", "abc"])

    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "Traceback" not in err


def test_invalid_direction_choice_fails_cleanly(tmp_path, monkeypatch, capsys):
    """Bilinmeyen bir `--direction` değeri de aynı şekilde temiz başarısız
    olmalı (argparse `choices` doğrulaması, çıkış 2)."""
    _write_fixture(tmp_path, monkeypatch, n_role_vectors=3)
    monkeypatch.setattr(ss, "load_hf_model", _refuse_to_load_model)

    with pytest.raises(SystemExit) as exc_info:
        ss.main(["--layers", "14", "--n-roles", "3", "--direction", "bogus"])

    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "Traceback" not in err


# --- Fix Round 1, M1: per-layer seeding doğrulaması ----------------------
#
# Reviewer'ın mutasyonu: `axis_layer=axis[L]` → `axis_layer=axis[args.layers[0]]`.
# Bu, her layer'ın direction'ını ilk katmanın axis row'undan oluşturur.
# Test aşağıda bu mutasyonu yakalar — iki farklı katman, iki farklı direction.


def test_control_direction_uses_per_layer_axis_and_role_vectors(
    tmp_path, monkeypatch
):
    """M1: iki katman ile kontrol yönü koşusu her katman için kendisinin
    axis'ini ve role vektörlerini kullanmalı (ilk katmanınkinden değil).
    Sahte kaydedilmiş `direction` argümanlarını kontrol ederek bunu sabitler.

    Mutasyon: `control_direction(..., axis_layer=axis[L], ...)` →
    `control_direction(..., axis_layer=axis[args.layers[0]], ...)`
    Bu durumda tüm katmanlar AYNI yöne sahip olur (ikisi de L14'ün axis'i ile
    kurulur); fark görülebilmesi için iki FARKLI katman (`[14, 5]`) gerekir.

    Bu test mutasyonu yakalar ve yolların doğru şekilde per-layer seeding yaptığını
    doğrular."""
    _write_fixture(tmp_path, monkeypatch, n_role_vectors=5)
    monkeypatch.setattr(ss, "load_hf_model", _fake_load_hf_model)
    gen = _new_recording_fake()
    monkeypatch.setattr(ss, "generate_steered", gen)

    # İki katman, kontrol yönü (shuffled) ile koşu
    exit_code = ss.main([
        "--layers", "14", "5",
        "--n-roles", "5",
        "--direction", "shuffled", "--seed", "0", "--variant", "test_m1",
    ])

    assert exit_code == 0
    axis = np.load(ss.config.model_results_dir() / "axis" / "assistant_axis.npy")
    vectors = np.load(ss.config.model_results_dir() / "axis" / "role_vectors.npy")

    # Her katman için beklenen yönü bağımsızca hesapla
    expected_14 = ss.control_direction(
        "shuffled", axis_layer=axis[14], role_vectors_layer=vectors[:, 14, :], seed=0
    )
    expected_5 = ss.control_direction(
        "shuffled", axis_layer=axis[5], role_vectors_layer=vectors[:, 5, :], seed=0
    )

    # İki yön FARKLI olmalı (fikstür kontrol: axis[14] != axis[5])
    assert not np.allclose(expected_14, expected_5), "Farklı katmanlar farklı yön vermelidir"

    # Üretici çağrılarında görülen yönler beklenenlerle eşleşmeli
    calls_by_layer = {}
    for call in gen.calls:
        layer = call["layer"]
        if layer not in calls_by_layer:
            calls_by_layer[layer] = []
        calls_by_layer[layer].append(call)

    for layer in (14, 5):
        expected = expected_14 if layer == 14 else expected_5
        for call in calls_by_layer[layer]:
            np.testing.assert_allclose(
                call["direction"], expected,
                err_msg=f"Layer {layer} yönü kendisinin axis'ini kullanmıyor"
            )
