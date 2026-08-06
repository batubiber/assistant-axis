"""`scripts/05_capture_activations.py` testleri.

Ağa çıkmaz, HF/torch model yüklemeye hiç dokunmaz: `load_hf_model` ve
`mean_response_activations` her `main()` testinde sahteleriyle değiştirilir.
Script dosya adı bir rakamla başladığı için (`05_capture_activations.py`)
normal `import` ile içe aktarılamaz; `importlib` ile dosya yolundan yüklenir
(bkz. `tests/test_judge_gate.py`).
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from aax.rollouts import rollouts_run_id, write_rollouts, write_rollouts_meta

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "05_capture_activations.py"
)


def _load_script():
    spec = importlib.util.spec_from_file_location("capture_activations", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ca = _load_script()


def test_module_is_registered_in_sys_modules():
    """Repo kuralı (bkz. test_judge_gate.py, test_generate_role_data.py):
    rakamla başlayan script'i importlib ile yüklerken modülü sys.modules'e
    de kaydet."""
    assert sys.modules["capture_activations"] is ca


def test_all_args_have_help_text():
    parser = ca.build_arg_parser()
    actions = {a.dest: a for a in parser._actions if a.dest != "help"}
    for dest in ("batch_size", "start_row", "checkpoint_every", "allow_pilot"):
        assert actions[dest].help, f"--{dest} için help metni eksik"


def test_checkpoint_every_default_is_250_not_25():
    """Önemli 4: planlanan ölçekte (16.000 satır, batch 8) eski varsayılan
    25, 200 satırda bir checkpoint demekti — ~80 tam matris (~3.67 GB)
    yeniden yazımı, ~290 GB toplam. 250 bunu ~6-7 yazıma indirir."""
    parser = ca.build_arg_parser()
    assert parser.get_default("checkpoint_every") == 250


# --- run_id: activations_index.json'ı kaynağa geri bağla ---------------------


def _make_records(roles: list[str]) -> list[dict]:
    return [
        {
            "kind": "role",
            "role": role,
            "system_prompt": f"{role} ol",
            "question": "q?",
            "answer": f"{role} cevabı",
        }
        for role in roles
    ]


def test_compute_run_id_is_deterministic_for_the_same_content():
    """`00_generate_role_data.py::compute_run_id` ile aynı desen: içerikten
    türetilir, aynı içerik her koşuda aynı kimliği verir."""
    a = _make_records(["pirate", "sage"])
    b = _make_records(["pirate", "sage"])
    assert ca.compute_run_id(a) == ca.compute_run_id(b)
    assert len(ca.compute_run_id(a)) == 16


def test_compute_run_id_changes_when_the_row_content_changes():
    """Farklı bir rollout kümesi farklı bir kimlik üretmeli — aksi hâlde
    `criterion_a.json`'daki `run_id` kaynağı ayırt edemez."""
    a = _make_records(["pirate", "sage"])
    b = _make_records(["pirate", "ghost"])
    assert ca.compute_run_id(a) != ca.compute_run_id(b)


def test_compute_run_id_is_not_clock_based():
    """Kimlik içerikten türetilir — aynı içerik her zaman aynı kimlik
    (bkz. `00_generate_role_data.py::test_run_id_is_not_clock_based`)."""
    records = _make_records(["pirate"])
    first = ca.compute_run_id(records)
    time.sleep(0.01)
    assert ca.compute_run_id(records) == first


def test_compute_run_id_matches_the_shared_helper():
    """04 (rollouts_meta.json), 05 (activations_index.json) ve 06
    (role_expression.json) AYNI sayıyı yazmak zorunda — 07 son ikisinin eşit
    olmasını şart koşuyor. Tek kaynak: `aax.rollouts.rollouts_run_id`."""
    records = _make_records(["pirate", "sage"])
    assert ca.compute_run_id(records) == rollouts_run_id(records)


# --- main(): D3 kapsaması, C4 (ilerleme/checkpoint) ve C5 (pilot reddi) ------


class _FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return " | ".join(f"{m['role']}:{m['content']}" for m in messages)

    def __call__(self, text, add_special_tokens=True):
        # Uzunluk metinden türesin ki satırlar birbirinden ayırt edilebilsin.
        return {"input_ids": list(range(1, min(len(text), 6) + 1))}


class _FakeBundle:
    n_layers = 2
    d_model = 3
    middle_layer = 1

    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()


def _patch_script_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(ca.config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(ca, "ACTS_PATH", tmp_path / "activations.npy")
    monkeypatch.setattr(ca, "INDEX_PATH", tmp_path / "activations_index.json")
    monkeypatch.setattr(ca, "PARTIAL_PATH", tmp_path / "activations_partial.json")
    monkeypatch.setattr(ca, "ROLLOUTS_PATH", tmp_path / "rollouts.jsonl")
    monkeypatch.setattr(ca, "ROLLOUTS_META_PATH", tmp_path / "rollouts_meta.json")


def _make_rollouts(n_role: int = 6, n_default: int = 4) -> list[dict]:
    records = []
    for i in range(n_role):
        records.append(
            {
                "kind": "role",
                "role": f"rol{i % 3}",
                "system_prompt": f"You are rol{i % 3}.",
                "question": f"soru {i}?",
                "sample_index": 0,
                "answer": f"cevap {i}",
            }
        )
    for i in range(n_default):
        records.append(
            {
                "kind": "default",
                "role": None,
                "system_prompt": None,
                "question": f"soru {i}?",
                "sample_index": i,
                "answer": f"varsayılan cevap {i}",
            }
        )
    return records


def _setup(monkeypatch, tmp_path, records, *, limit=None, fail_at_batch=None):
    """rollouts.jsonl + künye yaz, model/yakalama fonksiyonlarını sahtele.

    Sahte `mean_response_activations` her satıra kendi GLOBAL satır
    numarasını yazar — böylece hangi satırın nereye yazıldığı testte
    doğrulanabilir."""
    _patch_script_paths(monkeypatch, tmp_path)
    write_rollouts(tmp_path / "rollouts.jsonl", records)
    write_rollouts_meta(tmp_path / "rollouts_meta.json", records, limit)

    monkeypatch.setattr(ca, "load_hf_model", lambda *a, **k: _FakeBundle())
    monkeypatch.setattr(ca, "free_vram_mib", lambda: 1234)

    calls = {"n": 0, "sizes": []}

    def fake_mean(bundle, items, *, batch_size=8):
        calls["n"] += 1
        calls["sizes"].append(len(items))
        if fail_at_batch is not None and calls["n"] == fail_at_batch:
            raise RuntimeError("simüle edilmiş CUDA OOM")
        # İçerik: her satır, prompt uzunluğuyla işaretlenir; testler yalnızca
        # şekil ve satır konumlarıyla ilgileniyor.
        block = np.zeros((len(items), bundle.n_layers, bundle.d_model), dtype=np.float32)
        for row, (prompt_ids, answer_ids) in enumerate(items):
            block[row, :, :] = len(prompt_ids) * 100 + len(answer_ids)
        return block

    monkeypatch.setattr(ca, "mean_response_activations", fake_mean)
    return calls


def test_main_writes_one_index_row_per_rollout_in_the_same_order(tmp_path, monkeypatch):
    """Aşama 1'in SATIR KİMLİĞİ SÖZLEŞMESİ: `activations.npy`'nin i'nci
    satırı `rollouts.jsonl`'ın i'nci kaydına aittir ve `activations_index.
    json`'ın `rows[i]`'si onu tarif eder. `07_extract_axis.py`'nin rol/default
    ayrımı tamamen buna dayanır ve hiçbir yerde sınanmıyordu."""
    records = _make_rollouts()
    _setup(monkeypatch, tmp_path, records)

    assert ca.main([]) == 0

    acts = np.load(tmp_path / "activations.npy")
    index = json.loads((tmp_path / "activations_index.json").read_text(encoding="utf-8"))
    assert acts.shape == (len(records), 2, 3)
    assert index["n_rows"] == len(records) == len(index["rows"])
    for row, record in zip(index["rows"], records):
        assert row["kind"] == record["kind"]
        assert row["role"] == record["role"]
        assert row["system_prompt"] == record["system_prompt"]
    assert index["run_id"] == rollouts_run_id(records)
    assert index["middle_layer"] == 1
    assert not (tmp_path / "activations_partial.json").exists()


def test_main_prints_per_batch_progress(tmp_path, monkeypatch, capsys):
    """C4: bu geçiş ~2.000 batch sürüyordu ve TEK bir satır bile basmıyordu."""
    records = _make_rollouts(n_role=6, n_default=2)
    _setup(monkeypatch, tmp_path, records)

    assert ca.main(["--batch-size", "2"]) == 0

    out = capsys.readouterr().out
    assert "batch 1/4" in out
    assert "batch 4/4" in out
    assert "8/8 satır" in out
    assert "geçen" in out


def test_main_reports_the_failing_batch_and_saves_partial_results(
    tmp_path, monkeypatch, capsys
):
    """C4: tek bir hata tüm geçişi çöpe atıyordu, üstelik hangi batch'te
    olduğu bile yazılmıyordu."""
    records = _make_rollouts(n_role=6, n_default=2)
    _setup(monkeypatch, tmp_path, records, fail_at_batch=3)

    assert ca.main(["--batch-size", "2"]) == 2

    err = capsys.readouterr().err
    assert "BAŞARISIZ" in err
    assert "batch 3/4" in err
    assert "simüle edilmiş CUDA OOM" in err
    assert "--start-row 4" in err  # kaldığı yer

    partial = json.loads((tmp_path / "activations_partial.json").read_text(encoding="utf-8"))
    assert partial["rows_done"] == 4
    assert partial["run_id"] == rollouts_run_id(records)
    assert (tmp_path / "activations.npy").exists()
    # Yarım matrisin yanında ASLA eksiksiz görünümlü bir indeks durmamalı.
    assert not (tmp_path / "activations_index.json").exists()


def test_start_row_resumes_from_a_partial_run_without_recomputing(
    tmp_path, monkeypatch, capsys
):
    records = _make_rollouts(n_role=6, n_default=2)
    _setup(monkeypatch, tmp_path, records, fail_at_batch=3)
    assert ca.main(["--batch-size", "2"]) == 2
    first_half = np.load(tmp_path / "activations.npy")[:4].copy()
    capsys.readouterr()

    calls = _setup(monkeypatch, tmp_path, records)  # aynı veri, hatasız
    assert ca.main(["--batch-size", "2", "--start-row", "4"]) == 0

    assert sum(calls["sizes"]) == 4, "yalnızca kalan 4 satır yeniden hesaplanmalı"
    acts = np.load(tmp_path / "activations.npy")
    assert np.array_equal(acts[:4], first_half), "devam eden koşu ilk yarıyı korumalı"
    assert (tmp_path / "activations_index.json").exists()
    assert not (tmp_path / "activations_partial.json").exists()


def test_start_row_refuses_a_partial_file_from_another_run(tmp_path, monkeypatch, capsys):
    records = _make_rollouts(n_role=6, n_default=2)
    _setup(monkeypatch, tmp_path, records, fail_at_batch=3)
    ca.main(["--batch-size", "2"])
    capsys.readouterr()

    # Farklı bir rollout kümesi: aynı satır sayısı, farklı içerik.
    other = _make_rollouts(n_role=6, n_default=2)
    other[0]["role"] = "bambaska"
    _setup(monkeypatch, tmp_path, other)

    assert ca.main(["--batch-size", "2", "--start-row", "4"]) == 2
    assert "BAŞARISIZ" in capsys.readouterr().err


def test_start_row_refuses_when_no_partial_marker_exists_even_if_shape_matches(
    tmp_path, monkeypatch, capsys
):
    """Önemli 2: şekil eşleşmesi TEK BAŞINA yeterli değil.

    Gerçek senaryo: önceki TAM bir koşu `activations.npy`'yi bırakır (kısmi
    işaret başarı sonunda silinir, bkz. `test_main_writes_one_index_row_per_
    rollout_in_the_same_order`); rollout'lar aynı satır sayısıyla yeniden
    üretilir (`04` tekrar koşulur); yeni yakalama geçişi OS düzeyinde
    (`kill -9`, OOM-killer) ilk checkpoint'ten ÖNCE öldürülür — hiçbir kısmi
    işaret hiç YAZILMAZ. Operatör "kaldığı yerden devam" niyetiyle
    `--start-row` verirse, düzeltme öncesi kod yalnızca `existing.shape ==
    acts.shape` bakardı ve bu SESSİZCE geçerdi — tam olarak "aynı satır
    sayısı, farklı rollout kümesi" durumu, `C5`'in künye kontrolünün ayrı
    tuttuğu türden bir bayatlık. Burada basitleştirilmiş biçimde: TAMAMLANMIŞ
    bir koşunun ardından (kısmi işaret hiç yok) `--start-row` denenir ve
    ŞEKİL TAM UYSA BİLE reddedilmeli."""
    records = _make_rollouts(n_role=6, n_default=2)
    _setup(monkeypatch, tmp_path, records)
    assert ca.main([]) == 0  # tam, başarılı koşu — kısmi işaret hiç yazılmadı
    assert not (tmp_path / "activations_partial.json").exists()
    capsys.readouterr()

    assert ca.main(["--batch-size", "2", "--start-row", "4"]) == 2

    err = capsys.readouterr().err
    assert "BAŞARISIZ" in err
    assert "activations_partial.json" in err


def test_main_rejects_a_pilot_rollout_set(tmp_path, monkeypatch, capsys):
    """C5: `--limit 100` kanonik yola yazıyordu ve `05` farkı göremiyordu.
    Aşama 0'ın `load_role_catalog` sert reddiyle aynı desen."""
    records = _make_rollouts()
    _setup(monkeypatch, tmp_path, records, limit=100)

    assert ca.main([]) == 2

    err = capsys.readouterr().err
    assert "BAŞARISIZ" in err
    assert "PİLOT" in err
    assert "--allow-pilot" in err
    assert not (tmp_path / "activations.npy").exists()


def test_allow_pilot_lets_a_pilot_run_through_with_a_warning(tmp_path, monkeypatch, capsys):
    records = _make_rollouts()
    _setup(monkeypatch, tmp_path, records, limit=100)

    assert ca.main(["--allow-pilot"]) == 0

    out = capsys.readouterr().out
    assert "UYARI" in out and "PİLOT" in out
    assert (tmp_path / "activations_index.json").exists()


def test_main_rejects_a_missing_or_stale_meta_file(tmp_path, monkeypatch, capsys):
    records = _make_rollouts()
    _setup(monkeypatch, tmp_path, records)
    (tmp_path / "rollouts_meta.json").unlink()

    assert ca.main([]) == 2
    assert "BAŞARISIZ" in capsys.readouterr().err

    # Künye var ama BAŞKA bir rollout kümesini tarif ediyor.
    write_rollouts_meta(tmp_path / "rollouts_meta.json", _make_rollouts(n_role=2), None)
    assert ca.main([]) == 2
    assert "BAŞARISIZ" in capsys.readouterr().err


def test_main_removes_a_stale_index_before_overwriting_the_matrix(
    tmp_path, monkeypatch, capsys
):
    """Önceki bir koşudan kalma eksiksiz görünümlü indeks, üzerine yazılan
    matrisin yanında YALAN olur; geçiş yarıda kalırsa 07 sıfır satırları
    gerçek aktivasyon sanardı (satır sayısı ve run_id aynı kalabilir)."""
    records = _make_rollouts(n_role=6, n_default=2)
    _setup(monkeypatch, tmp_path, records, fail_at_batch=2)
    (tmp_path / "activations_index.json").write_text('{"eski": true}', encoding="utf-8")

    assert ca.main(["--batch-size", "2"]) == 2

    assert not (tmp_path / "activations_index.json").exists()
    assert "Bayat" in capsys.readouterr().out


@pytest.mark.parametrize("start_row", [-1, 99])
def test_main_rejects_an_out_of_range_start_row(tmp_path, monkeypatch, capsys, start_row):
    records = _make_rollouts()
    _setup(monkeypatch, tmp_path, records)

    assert ca.main(["--start-row", str(start_row)]) == 2
    assert "--start-row" in capsys.readouterr().err


# --- Önemli 4: checkpoint yazımı atomik olmalı --------------------------------


def test_atomic_save_npy_no_temp_left_on_success(tmp_path):
    path = tmp_path / "acts.npy"
    ca._atomic_save_npy(path, np.zeros((2, 3), dtype=np.float32))

    leftovers = [p.name for p in tmp_path.iterdir() if p.name != "acts.npy"]
    assert leftovers == []
    assert np.array_equal(np.load(path), np.zeros((2, 3), dtype=np.float32))


def test_atomic_save_npy_preserves_existing_file_and_leaves_no_temp_on_failure(
    tmp_path, monkeypatch
):
    """Gerçek atomiklik garantisi: `aax.rollouts.write_rollouts`'un
    `test_write_failure_partway_leaves_no_temp_and_preserves_existing_target`
    testiyle aynı desen, ama `.npy` için. Sahte, tempfile'sız bir düz
    `np.save(path, array)` bu testi geçemezdi: yarıda kesilen bir yazım
    hedefi kısmi/bozuk içerikle üzerine yazardı — planlanan ölçekte
    (~80 checkpoint yazımı) bu, `--start-row`'un okuyacağı dosyanın tam da
    devam ederken bozulması demekti."""
    path = tmp_path / "acts.npy"
    original = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    np.save(path, original)

    def boom(*_args, **_kwargs):
        raise RuntimeError("simüle edilmiş çökme (disk dolu / Ctrl-C / OOM)")

    monkeypatch.setattr(ca.np, "save", boom)

    with pytest.raises(RuntimeError, match="simüle edilmiş çökme"):
        ca._atomic_save_npy(path, np.zeros(3, dtype=np.float32))

    leftovers = [p.name for p in tmp_path.iterdir() if p.name != "acts.npy"]
    assert leftovers == [], "yarıda kesilen yazım tempfile bırakmamalı"
    assert np.array_equal(np.load(path), original), "hedef dosya bozulmamalı"
