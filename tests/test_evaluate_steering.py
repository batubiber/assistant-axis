"""`scripts/09_evaluate_steering.py` testleri.

İlk 7 test brief'in Adım 2'sinden BİREBİR — bu script'in Task 5 sözleşmesinin
en temel şekli. Geri kalanı supplement'in (`.superpowers/sdd/
p3-task-5-supplement.md`) E1-E3 maddelerini kapsar: böl-ve-kurtar (06'nın
`_label_batch` deseninin persona sınıflandırmasındaki karşılığı), yarım kalmış
bir sweep'ten karar üretilmemesi, ve boş bir karar kümesinin artık "GEÇTİ"
sayılmaması.

Ağa çıkmaz: gerçek `GatewayClient`'a hiç dokunulmaz, `build_default_client`
her testte monkeypatch'lenir (`tests/conftest.py` zaten soketleri kapatıyor,
bu ikinci bir savunma katmanı). Script dosya adı bir rakamla başladığı için
normal `import` ile içe aktarılamaz; `importlib` ile dosya yolundan yüklenir
(bkz. `tests/test_label_and_train_probe.py`, `tests/test_steering_sweep.py`).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_P = Path(__file__).resolve().parents[1] / "scripts" / "09_evaluate_steering.py"


def _load():
    spec = importlib.util.spec_from_file_location("evaluate_steering", _P)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ev = _load()


def test_module_is_registered_in_sys_modules():
    assert "evaluate_steering" in sys.modules


def test_groups_records_by_layer_and_strength():
    records = [
        {"layer": 14, "strength": 0.0, "role": "a", "question": "q", "answer": "x"},
        {"layer": 14, "strength": 0.0, "role": "b", "question": "q", "answer": "y"},
        {"layer": 19, "strength": -0.6, "role": "a", "question": "q", "answer": "z"},
    ]
    groups = ev.group_by_layer_strength(records)
    assert sorted(groups) == [(14, 0.0), (19, -0.6)]
    assert len(groups[(14, 0.0)]) == 2


def test_rates_are_computed_per_layer():
    labels = {
        (14, 0.0): ["assistant", "assistant", "human_role", "assistant"],
        (14, -0.6): ["human_role", "nonhuman_role", "weird_role", "assistant"],
    }
    rates = ev.rates_by_layer(labels)
    assert rates[14][0.0] == pytest.approx(0.25)
    assert rates[14][-0.6] == pytest.approx(0.75)


def test_missing_zero_strength_for_a_layer_is_a_hard_error():
    labels = {(14, -0.6): ["assistant"]}
    with pytest.raises(ValueError, match="0.0"):
        ev.evaluate_all_layers(ev.rates_by_layer(labels))


def test_evaluate_all_layers_returns_one_verdict_per_layer():
    rates = {14: {0.0: 0.10, -0.6: 0.40}, 19: {0.0: 0.10, -0.6: 0.20}}
    out = ev.evaluate_all_layers(rates)
    assert out[14]["passed"] is True
    assert out[19]["passed"] is False


def test_overall_exit_code_is_zero_only_if_every_layer_passes():
    assert ev.overall_exit_code({14: {"passed": True}, 19: {"passed": True}}) == 0
    assert ev.overall_exit_code({14: {"passed": True}, 19: {"passed": False}}) == 1
    assert ev.overall_exit_code({14: {"passed": False}}) == 1


def test_missing_sweep_file_exits_two_with_a_diagnostic(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ev.config, "model_data_dir", lambda: tmp_path)
    assert ev.main([]) == 2
    err = capsys.readouterr().err
    assert "BAŞARISIZ" in err
    assert "Traceback" not in err


# --- E3: boş bir karar kümesi artık "GEÇTİ" (0) sayılmaz --------------------
#
# `all([])` Python'da `True`'dur — bu projede daha önce görülen "tanımsız
# veriden GEÇTİ basmak" hatasının aynı sınıfı.


def test_overall_exit_code_is_two_for_empty_verdicts():
    assert ev.overall_exit_code({}) == 2


# --- ortak test yardımcıları -------------------------------------------------


def _patch_paths(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    results_dir = tmp_path / "results"
    monkeypatch.setattr(ev.config, "model_data_dir", lambda: data_dir)
    monkeypatch.setattr(ev.config, "model_results_dir", lambda: results_dir)
    return data_dir, results_dir


def _write_sweep(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )


def _write_meta(
    path: Path, *, planned: int, attempted: int, complete: bool, axis_run_id: str | None = None
) -> None:
    """`axis_run_id` varsayılan `None` — gerçek `08_steering_sweep.py` meta'sı
    bu alanı HER ZAMAN yazar (bkz. `_meta_payload`), ama F1'den ÖNCEKİ
    testlerin çoğu bununla ilgilenmiyor; `None` tutarlı bir varsayılan (aynı
    testte iki kez yazılsa bile aynı değeri üretir, sahte bir uyuşmazlık
    doğurmaz)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "planned": planned, "attempted": attempted, "complete": complete,
            "axis_run_id": axis_run_id,
        }),
        encoding="utf-8",
    )


def _sweep_sha256(path: Path) -> str:
    """Testin sweep dosyası için `_run`'ın hesaplayacağı sha256'nın AYNISI —
    F1 testlerinin, script'in kendi mantığını TEKRARLAMADAN, "gerçek" parmak
    izini üretebilmesi için."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_records(layer: int, strengths: list[float], n_roles: int, n_questions: int) -> list[dict]:
    records = []
    for s in strengths:
        for r in range(n_roles):
            for q in range(n_questions):
                records.append({
                    "layer": layer,
                    "strength": s,
                    "role": f"role{r}",
                    "question": f"question{q}",
                    "answer": f"answer L{layer} s{s} r{r} q{q}",
                })
    return records


class FakeClient:
    """Ağsız sahte hakem istemcisi.

    `default_label`: içerik ne olursa olsun döndürülecek sabit persona
    kategorisi. `label_rule`: verilirse her `[ITEM ...]` bloğu İÇİN AYRI
    çağrılır (`block: str -> kategori`) — gruplar arasında FARKLI oranlar
    üretmek için (ör. güce göre etiket değişsin). `poison_marker`: içeriğinde
    bu alt dize geçen HERHANGİ bir batch/yarı/tekil istek `JudgeParseError`
    fırlatır — hakemin belirli bir öğeyi hiçbir boyutta ayrıştıramamasının
    sahtesi. `exceptions`: sırayla `.chat()` çağrılarının ÖNÜNE geçip
    fırlatılır.
    """

    def __init__(
        self,
        *,
        default_label: str = "assistant",
        label_rule=None,
        poison_marker: str | None = None,
        exceptions: list | None = None,
        stage_remaining: int = 10_000,
        global_remaining: int = 10_000,
    ) -> None:
        self.default_label = default_label
        self.label_rule = label_rule
        self.poison_marker = poison_marker
        self._exceptions = list(exceptions or [])
        self._stage_remaining = stage_remaining
        self._global_remaining = global_remaining
        self.calls = 0
        self.sends_made = 0
        self.sizes_seen: list[int] = []

    def would_call(self, messages, *, temperature=0.0, max_tokens=1024):
        return True

    def remaining_budget(self, stage):
        return self._stage_remaining, self._global_remaining

    def chat(self, messages, *, stage, temperature=0.0, max_tokens=1024):
        self.calls += 1
        self.sends_made += 1
        if self._exceptions:
            raise self._exceptions.pop(0)
        content = messages[0]["content"]
        if self.poison_marker and self.poison_marker in content:
            raise ev.JudgeParseError("hakem ayrıştıramadı: uzunluk uyuşmazlığı")
        blocks = content.split("[ITEM ")[1:]
        self.sizes_seen.append(len(blocks))
        if self.label_rule is not None:
            labels = [self.label_rule(block) for block in blocks]
        else:
            labels = [self.default_label] * len(blocks)
        return json.dumps(labels)


class RaisesOnNthCallClient:
    """N'inci `.chat()` çağrısında verilen istisnayı fırlatan, öncesi/sonrası
    normal çalışan sahte istemci — bir grup TAMAMEN etiketlendikten SONRA
    bir sonraki grubun patlaması senaryosunu üretmek içindir."""

    def __init__(self, exc: Exception, *, n: int = 2, default_label: str = "assistant") -> None:
        self._exc = exc
        self._n = n
        self.default_label = default_label
        self.calls = 0
        self.sends_made = 0

    def remaining_budget(self, stage):
        return 10_000, 10_000

    def chat(self, messages, *, stage, temperature=0.0, max_tokens=1024):
        self.calls += 1
        self.sends_made += 1
        if self.calls == self._n:
            raise self._exc
        content = messages[0]["content"]
        n_items = content.count("[ITEM ")
        return json.dumps([self.default_label] * n_items)


# --- E1: böl-ve-kurtar (`_classify_batch`) — 06'nın `_label_batch`'iyle -----
# AYNI desen -------------------------------------------------------------


def test_classify_batch_recovers_from_a_bad_batch_via_split():
    class FailsWholeBatchOnlyClient:
        def __init__(self) -> None:
            self.calls = 0
            self.sizes_seen: list[int] = []

        def chat(self, messages, *, stage, temperature=0.0, max_tokens=1024):
            self.calls += 1
            content = messages[0]["content"]
            n_items = content.count("[ITEM ")
            self.sizes_seen.append(n_items)
            if n_items == 10:
                raise ev.JudgeParseError("uzunluk uyuşmazlığı: 11 != 10")
            return json.dumps(["assistant"] * n_items)

    client = FailsWholeBatchOnlyClient()
    positions = list(range(10))
    items = [(f"soru {i}", f"cevap {i}") for i in positions]

    labels, unlabelled, split = ev._classify_batch(
        client, positions=positions, items=items, stage=ev.STAGE
    )

    assert unlabelled == []
    assert labels == {p: "assistant" for p in positions}
    assert split == 1
    # 1 (tüm batch, başarısız) + 2 (yarılar, ikisi de başarılı) = 3 gönderim.
    assert client.calls == 3
    assert client.sizes_seen == [10, 5, 5]


def test_classify_batch_worst_case_cost_is_bounded_at_thirteen_sends_for_ten_items():
    class AlwaysFailsClient:
        def __init__(self) -> None:
            self.calls = 0

        def chat(self, messages, *, stage, temperature=0.0, max_tokens=1024):
            self.calls += 1
            raise ev.JudgeParseError("hep bozuk")

    client = AlwaysFailsClient()
    positions = list(range(10))
    items = [(f"soru {i}", f"cevap {i}") for i in positions]

    labels, unlabelled, split = ev._classify_batch(
        client, positions=positions, items=items, stage=ev.STAGE
    )

    assert labels == {}
    assert sorted(unlabelled) == positions
    assert split == 1
    assert client.calls == 13


def test_classify_batch_size_one_half_is_not_retried_with_an_identical_payload():
    class AlwaysFailsClient:
        def __init__(self) -> None:
            self.calls = 0

        def chat(self, messages, *, stage, temperature=0.0, max_tokens=1024):
            self.calls += 1
            raise ev.JudgeParseError("hep bozuk")

    client = AlwaysFailsClient()
    labels, unlabelled, split = ev._classify_batch(
        client, positions=[0, 1], items=[("s0", "c0"), ("s1", "c1")], stage=ev.STAGE
    )

    assert labels == {}
    assert sorted(unlabelled) == [0, 1]
    assert split == 1
    # 1 (tüm batch) + 2 (yarılar, İKİSİ de zaten tekil) = 3 — YİNELENEN
    # tekil deneme YOK.
    assert client.calls == 3


def test_item_that_fails_even_alone_is_recorded_unlabelled_and_run_continues(
    tmp_path, monkeypatch, capsys
):
    """Bir öğe (batch -> yarı -> tekil) her boyutta başarısız olsa bile koşu
    DÜŞMEZ: 'etiketlenemedi' sayılır, koşu devam eder, sayı artefakta
    yazılır (E1)."""
    data_dir, results_dir = _patch_paths(monkeypatch, tmp_path)
    records = _make_records(14, [0.0, -0.6], n_roles=2, n_questions=2)
    _write_sweep(data_dir / "steering_sweep.jsonl", records)
    _write_meta(data_dir / "steering_sweep_meta.json", planned=8, attempted=8, complete=True)
    # (14, 0.0) grubunun İLK öğesinin cevabı ASLA ayrıştırılamayan "zehirli" öğe.
    poison_marker = records[0]["answer"]
    client = FakeClient(default_label="assistant", poison_marker=poison_marker)
    monkeypatch.setattr(ev, "build_default_client", lambda: client)

    exit_code = ev.main([])

    assert exit_code == 1  # kriter DEĞERLENDİRİLDİ (rate her iki grupta da 0.0 — düştü)
    err = capsys.readouterr().err
    assert "UYARI" in err
    assert "toplam 1 öğe" in err

    labels_payload = json.loads((data_dir / "steering_labels.json").read_text(encoding="utf-8"))
    assert "0" not in labels_payload["labels"]["14|0.0"], "zehirli öğenin konumu (0) etiketlenemedi"
    assert set(labels_payload["labels"]["14|0.0"].values()) == {"assistant"}
    assert len(labels_payload["labels"]["14|0.0"]) == 3
    assert labels_payload["unlabelled_positions"]["14|0.0"] == [0]
    assert set(labels_payload["labels"]["14|-0.6"].values()) == {"assistant"}

    criterion_payload = json.loads(
        (results_dir / "steering" / "criterion_b.json").read_text(encoding="utf-8")
    )
    assert criterion_payload["unlabelled_by_group"]["14|0.0"] == 1
    assert criterion_payload["unlabelled_by_group"]["14|-0.6"] == 0


def test_entirely_unlabelled_cell_exits_two_not_traceback(tmp_path, monkeypatch, capsys):
    """Bir hücrenin TÜM öğeleri etiketlenemezse (`non_assistant_rate` boş
    listede zaten `ValueError` atar) bu temiz bir 2'ye dönüşmeli, traceback'e
    değil (E1)."""
    data_dir, results_dir = _patch_paths(monkeypatch, tmp_path)
    records = _make_records(14, [0.0, -0.6], n_roles=1, n_questions=1)
    _write_sweep(data_dir / "steering_sweep.jsonl", records)
    _write_meta(data_dir / "steering_sweep_meta.json", planned=2, attempted=2, complete=True)
    # -0.6 grubunun TEK öğesi her zaman zehirli; 0.0 grubu temiz kalır.
    client = FakeClient(default_label="assistant", poison_marker="s-0.6")
    monkeypatch.setattr(ev, "build_default_client", lambda: client)

    exit_code = ev.main([])

    assert exit_code == 2
    err = capsys.readouterr().err
    assert "BAŞARISIZ" in err
    assert "Traceback" not in err


def test_labels_file_left_behind_by_an_interrupted_run_is_valid_json(
    tmp_path, monkeypatch, capsys
):
    """Kesintiye uğrayan bir koşu (`BudgetExceeded`) o ana kadarki
    ilerlemeyi GEÇERLİ JSON olarak diske bırakmalı (E1)."""
    data_dir, results_dir = _patch_paths(monkeypatch, tmp_path)
    records = _make_records(14, [0.0, -0.6], n_roles=2, n_questions=2)
    _write_sweep(data_dir / "steering_sweep.jsonl", records)
    _write_meta(data_dir / "steering_sweep_meta.json", planned=8, attempted=8, complete=True)
    # sorted(groups) -> [(14, -0.6), (14, 0.0)]: ilk grup (-0.6) TAMAMLANIR,
    # ikinci grubun (0.0) TEK batch'i BudgetExceeded ile patlar.
    client = RaisesOnNthCallClient(ev.BudgetExceeded("'stage4_steering' bütçesi doldu"), n=2)
    monkeypatch.setattr(ev, "build_default_client", lambda: client)

    exit_code = ev.main([])

    assert exit_code == 2
    err = capsys.readouterr().err
    assert "DURDURULDU" in err
    assert "kalıcı olarak" in err

    labels_path = data_dir / "steering_labels.json"
    assert labels_path.exists()
    payload = json.loads(labels_path.read_text(encoding="utf-8"))  # geçerli JSON olmalı
    assert len(payload["labels"]["14|-0.6"]) == 4
    assert "14|0.0" not in payload["labels"] or payload["labels"]["14|0.0"] == {}
    assert not (results_dir / "steering" / "criterion_b.json").exists()


def test_gateway_error_stops_cleanly_after_persisting_progress(tmp_path, monkeypatch, capsys):
    data_dir, results_dir = _patch_paths(monkeypatch, tmp_path)
    records = _make_records(14, [0.0, -0.6], n_roles=2, n_questions=2)
    _write_sweep(data_dir / "steering_sweep.jsonl", records)
    _write_meta(data_dir / "steering_sweep_meta.json", planned=8, attempted=8, complete=True)
    client = RaisesOnNthCallClient(ev.GatewayError("HTTP 500"), n=2)
    monkeypatch.setattr(ev, "build_default_client", lambda: client)

    exit_code = ev.main([])

    assert exit_code == 2
    err = capsys.readouterr().err
    assert "BAŞARISIZ" in err
    assert "kalıcı olarak" in err
    assert (data_dir / "steering_labels.json").exists()


def test_resume_skips_already_completed_groups_and_reuses_saved_labels(
    tmp_path, monkeypatch, capsys
):
    """Önceki (kesintiye uğramış) bir koşudan kalma etiketler diskten
    yüklenir ve o GRUP hakeme HİÇ tekrar sorulmaz."""
    data_dir, results_dir = _patch_paths(monkeypatch, tmp_path)
    records = _make_records(14, [0.0, -0.6], n_roles=2, n_questions=2)
    sweep_path = data_dir / "steering_sweep.jsonl"
    _write_sweep(sweep_path, records)
    _write_meta(
        data_dir / "steering_sweep_meta.json", planned=8, attempted=8, complete=True,
        axis_run_id="axis-resume",
    )
    # F1: elle yazılan durumun parmak izi GERÇEK sweep'inkiyle uyuşmalı,
    # yoksa `_run` bunu bayat sayıp atar (bu testin amacı bu DEĞİL — o,
    # aşağıdaki F1 testlerinde ayrıca sabitleniyor).
    ev.save_group_labels(
        data_dir / "steering_labels.json",
        {(14, -0.6): {0: "assistant", 1: "assistant", 2: "assistant", 3: "assistant"}},
        {(14, -0.6): set()},
        sweep_sha256=_sweep_sha256(sweep_path),
        axis_run_id="axis-resume",
        record_counts={(14, -0.6): 4, (14, 0.0): 4},
    )
    client = FakeClient(default_label="human_role")
    monkeypatch.setattr(ev, "build_default_client", lambda: client)

    exit_code = ev.main([])

    assert exit_code in (0, 1)
    out = capsys.readouterr().out
    assert "1/2 grup diskten yüklendi" in out
    # yalnızca (14, 0.0) grubu için (4 öğe, batch_size=10) TEK bir gönderim.
    assert client.calls == 1

    labels_payload = json.loads((data_dir / "steering_labels.json").read_text(encoding="utf-8"))
    assert labels_payload["labels"]["14|-0.6"] == {
        "0": "assistant", "1": "assistant", "2": "assistant", "3": "assistant"
    }
    assert set(labels_payload["labels"]["14|0.0"].values()) == {"human_role"}


def test_corrupt_labels_file_is_reported_cleanly_not_silently_discarded(
    tmp_path, monkeypatch, capsys
):
    data_dir, results_dir = _patch_paths(monkeypatch, tmp_path)
    records = _make_records(14, [0.0, -0.6], n_roles=1, n_questions=1)
    _write_sweep(data_dir / "steering_sweep.jsonl", records)
    _write_meta(data_dir / "steering_sweep_meta.json", planned=2, attempted=2, complete=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "steering_labels.json").write_text("{bozuk json", encoding="utf-8")
    client = FakeClient()
    monkeypatch.setattr(ev, "build_default_client", lambda: client)

    exit_code = ev.main([])

    assert exit_code == 2
    err = capsys.readouterr().err
    assert "BAŞARISIZ" in err
    assert "bozuk" in err
    assert client.calls == 0, "bozuk etiket dosyası tespit edildiyse istek atılmamalı"


# --- E2: yarım kalmış bir sweep'ten B kriteri kararı ÜRETİLMEMELİ ------------


def test_missing_meta_file_exits_two(tmp_path, monkeypatch, capsys):
    """F6 (Fix Round 1): kardeşi (`test_meta_says_incomplete_exits_two_
    with_counts`) `build_default_client`'ı monkeypatch'leyip `client.calls
    == 0` doğruluyordu, bu test doğrulamıyordu — meta guard'ı yanlışlıkla
    izin veren bir `return` ile değiştirmek (guard'ın kendisi yerine) 27
    testi de geçiriyordu, çünkü kontrol akışı `build_default_client`'a
    düşüyor, o da anahtar tanımsız olduğu için AYRI bir çıkış-2 üretiyordu.
    Burada `build_default_client` GERÇEKTEN ÇALIŞAN bir `FakeClient`
    döndürüyor — guard kaldırılırsa client.calls artık 0 KALMAZ (sweep
    normal işlenir), bu da mutasyonu YAKALAR."""
    data_dir, results_dir = _patch_paths(monkeypatch, tmp_path)
    records = _make_records(14, [0.0, -0.6], n_roles=1, n_questions=1)
    _write_sweep(data_dir / "steering_sweep.jsonl", records)
    # meta dosyası KASITLI olarak hiç yazılmadı.
    client = FakeClient()
    monkeypatch.setattr(ev, "build_default_client", lambda: client)

    exit_code = ev.main([])

    assert exit_code == 2
    err = capsys.readouterr().err
    assert "BAŞARISIZ" in err
    assert "Traceback" not in err
    assert "sweep meta'sı okunamadı" in err, "meta'ya ÖZGÜ teşhis — başka bir çıkış-2 yolu DEĞİL"
    assert "steering_sweep_meta.json" in err
    assert client.calls == 0, "meta eksikse hakem hiç çağrılmamalı (build_default_client'a düşülmemeli)"


def test_meta_says_incomplete_exits_two_with_counts(tmp_path, monkeypatch, capsys):
    data_dir, results_dir = _patch_paths(monkeypatch, tmp_path)
    records = _make_records(14, [0.0, -0.6], n_roles=1, n_questions=1)
    _write_sweep(data_dir / "steering_sweep.jsonl", records)
    _write_meta(data_dir / "steering_sweep_meta.json", planned=100, attempted=40, complete=False)
    client = FakeClient()
    monkeypatch.setattr(ev, "build_default_client", lambda: client)

    exit_code = ev.main([])

    assert exit_code == 2
    err = capsys.readouterr().err
    assert "BAŞARISIZ" in err
    assert "40" in err and "100" in err
    assert "--allow-incomplete" in err
    assert client.calls == 0, "eksik sweep tespit edildiyse hakem hiç çağrılmamalı"


def test_allow_incomplete_flag_evaluates_and_marks_artifact(tmp_path, monkeypatch, capsys):
    data_dir, results_dir = _patch_paths(monkeypatch, tmp_path)
    records = _make_records(14, [0.0, -0.6], n_roles=3, n_questions=1)
    _write_sweep(data_dir / "steering_sweep.jsonl", records)
    _write_meta(data_dir / "steering_sweep_meta.json", planned=100, attempted=6, complete=False)
    client = FakeClient(default_label="human_role")
    monkeypatch.setattr(ev, "build_default_client", lambda: client)

    exit_code = ev.main(["--allow-incomplete"])

    assert exit_code == 1  # taban == uzak oran (ikisi de human_role) -> düştü, ama DEĞERLENDİRİLDİ
    err = capsys.readouterr().err
    assert "UYARI" in err
    assert "KISMİ SWEEP" in err

    criterion_path = results_dir / "steering" / "criterion_b.json"
    assert criterion_path.exists()
    payload = json.loads(criterion_path.read_text(encoding="utf-8"))
    assert payload["incomplete_sweep"] is True
    assert payload["sweep_attempted"] == 6
    assert payload["sweep_planned"] == 100


def test_complete_sweep_does_not_mark_the_artifact_as_incomplete(tmp_path, monkeypatch):
    data_dir, results_dir = _patch_paths(monkeypatch, tmp_path)
    records = _make_records(14, [0.0, -0.6], n_roles=3, n_questions=1)
    _write_sweep(data_dir / "steering_sweep.jsonl", records)
    _write_meta(data_dir / "steering_sweep_meta.json", planned=6, attempted=6, complete=True)
    client = FakeClient(default_label="human_role")
    monkeypatch.setattr(ev, "build_default_client", lambda: client)

    ev.main([])

    payload = json.loads(
        (results_dir / "steering" / "criterion_b.json").read_text(encoding="utf-8")
    )
    assert "incomplete_sweep" not in payload


# --- bütçe / gateway kurulumu ------------------------------------------------


def test_main_missing_api_key_produces_diagnostic(tmp_path, monkeypatch, capsys):
    data_dir, results_dir = _patch_paths(monkeypatch, tmp_path)
    records = _make_records(14, [0.0, -0.6], n_roles=1, n_questions=1)
    _write_sweep(data_dir / "steering_sweep.jsonl", records)
    _write_meta(data_dir / "steering_sweep_meta.json", planned=2, attempted=2, complete=True)

    def raise_missing_key():
        raise RuntimeError(
            "APP_KEY_JAILBREAK ortam değişkeni tanımlı değil. "
            "Dağıtım ortamınızın .env dosyasından alıp kabuğunuzda export edin."
        )

    monkeypatch.setattr(ev, "build_default_client", raise_missing_key)

    exit_code = ev.main([])

    assert exit_code == 2
    err = capsys.readouterr().err
    assert "BAŞARISIZ" in err
    assert "gateway istemcisi kurulamadı" in err


def test_dry_run_reports_plan_and_exits_zero_when_under_budget(tmp_path, monkeypatch, capsys):
    data_dir, results_dir = _patch_paths(monkeypatch, tmp_path)
    records = _make_records(14, [0.0, -0.6], n_roles=5, n_questions=2)  # 10 öğe/grup
    _write_sweep(data_dir / "steering_sweep.jsonl", records)
    _write_meta(data_dir / "steering_sweep_meta.json", planned=20, attempted=20, complete=True)
    client = FakeClient(stage_remaining=100, global_remaining=1000)
    monkeypatch.setattr(ev, "build_default_client", lambda: client)

    exit_code = ev.main(["--dry-run"])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Grup sayısı: 2" in out
    assert client.calls == 0, "--dry-run tek bir hakem çağrısı bile atmamalı"


def test_dry_run_fails_when_plan_exceeds_remaining_budget(tmp_path, monkeypatch, capsys):
    data_dir, results_dir = _patch_paths(monkeypatch, tmp_path)
    records = _make_records(14, [0.0, -0.6], n_roles=5, n_questions=2)
    _write_sweep(data_dir / "steering_sweep.jsonl", records)
    _write_meta(data_dir / "steering_sweep_meta.json", planned=20, attempted=20, complete=True)
    client = FakeClient(stage_remaining=1, global_remaining=1000)
    monkeypatch.setattr(ev, "build_default_client", lambda: client)

    exit_code = ev.main(["--dry-run"])

    assert exit_code == 2
    assert "HATA" in capsys.readouterr().err


def test_real_run_fails_closed_when_plan_exceeds_remaining_budget_without_spending(
    tmp_path, monkeypatch, capsys
):
    data_dir, results_dir = _patch_paths(monkeypatch, tmp_path)
    records = _make_records(14, [0.0, -0.6], n_roles=5, n_questions=2)
    _write_sweep(data_dir / "steering_sweep.jsonl", records)
    _write_meta(data_dir / "steering_sweep_meta.json", planned=20, attempted=20, complete=True)
    client = FakeClient(stage_remaining=1, global_remaining=1000)
    monkeypatch.setattr(ev, "build_default_client", lambda: client)

    exit_code = ev.main([])

    assert exit_code == 2
    err = capsys.readouterr().err
    assert "BAŞARISIZ" in err
    assert "sığmıyor" in err
    assert client.calls == 0


# --- uçtan uca mutlu yol: üç artefakt de doğru şemayla yazılır --------------


def test_full_happy_path_writes_all_artifacts_with_correct_verdicts(tmp_path, monkeypatch):
    data_dir, results_dir = _patch_paths(monkeypatch, tmp_path)
    records = _make_records(14, [0.0, -0.6], n_roles=3, n_questions=2)  # 6 öğe/grup
    _write_sweep(data_dir / "steering_sweep.jsonl", records)
    _write_meta(data_dir / "steering_sweep_meta.json", planned=12, attempted=12, complete=True)
    client = FakeClient(
        label_rule=lambda block: "human_role" if "s-0.6" in block else "assistant"
    )
    monkeypatch.setattr(ev, "build_default_client", lambda: client)

    exit_code = ev.main([])

    assert exit_code == 0

    labels_payload = json.loads(
        (data_dir / "steering_labels.json").read_text(encoding="utf-8")
    )
    assert set(labels_payload["labels"]) == {"14|0.0", "14|-0.6"}
    assert len(labels_payload["labels"]["14|0.0"]) == 6
    assert len(labels_payload["labels"]["14|-0.6"]) == 6

    rate_payload = json.loads(
        (results_dir / "steering" / "rate_by_strength.json").read_text(encoding="utf-8")
    )
    assert rate_payload["14"]["0.0"] == pytest.approx(0.0)
    assert rate_payload["14"]["-0.6"] == pytest.approx(1.0)

    criterion_payload = json.loads(
        (results_dir / "steering" / "criterion_b.json").read_text(encoding="utf-8")
    )
    assert criterion_payload["layers"]["14"]["passed"] is True
    assert criterion_payload["labelled_by_group"] == {"14|0.0": 6, "14|-0.6": 6}
    assert criterion_payload["unlabelled_by_group"] == {"14|0.0": 0, "14|-0.6": 0}
    assert "incomplete_sweep" not in criterion_payload
    assert "PAYDA" in criterion_payload["note"]


def test_two_layers_get_independent_verdicts(tmp_path, monkeypatch):
    """B kriteri KATMAN BAŞINA değerlendirilir: bir katman geçebilir,
    diğeri aynı koşuda düşebilir."""
    data_dir, results_dir = _patch_paths(monkeypatch, tmp_path)
    records = (
        _make_records(14, [0.0, -0.6], n_roles=3, n_questions=2)
        + _make_records(19, [0.0, -0.6], n_roles=3, n_questions=2)
    )
    _write_sweep(data_dir / "steering_sweep.jsonl", records)
    _write_meta(data_dir / "steering_sweep_meta.json", planned=24, attempted=24, complete=True)

    def label_rule(block: str) -> str:
        # L14: -0.6'da tamamen role geçer (geçer). L19: hiç geçmez (düşer).
        if "L14" in block and "s-0.6" in block:
            return "human_role"
        return "assistant"

    client = FakeClient(label_rule=label_rule)
    monkeypatch.setattr(ev, "build_default_client", lambda: client)

    exit_code = ev.main([])

    assert exit_code == 1  # en az bir katman düştü
    criterion_payload = json.loads(
        (results_dir / "steering" / "criterion_b.json").read_text(encoding="utf-8")
    )
    assert criterion_payload["layers"]["14"]["passed"] is True
    assert criterion_payload["layers"]["19"]["passed"] is False


# --- F1 (Fix Round 1): steering_labels.json'ın üretildiği sweep'e bağlanması,
# bkz. `.superpowers/sdd/p3-task-5-fix1-brief.md`. `axis_run_id` TEK BAŞINA
# yetmez (aktivasyon indeksinden gelir, sweep'in kendisinden değil — aynı
# eksenle YENİDEN üretilen bir sweep aynı `axis_run_id`'yi taşır ama
# `do_sample=True, temperature=1.0` yüzünden TÜM yanıtlar yeni metindir);
# asıl doğrulama sweep'in KENDİ baytlarının sha256'sı.
# -----------------------------------------------------------------------


def test_stale_labels_from_a_regenerated_sweep_are_discarded_and_rejudged(
    tmp_path, monkeypatch, capsys
):
    """08'in ARŞİVLEYİP BAŞTAN yazdığı bir sweep AYNI `axis_run_id`'yi taşır
    ama TÜM yanıtlar YENİ metindir. `sweep_sha256` bunu yakalar — aynı hücre
    sayıları, aynı `axis_run_id`, ama FARKLI baytlar."""
    data_dir, results_dir = _patch_paths(monkeypatch, tmp_path)
    records_v1 = _make_records(14, [0.0, -0.6], n_roles=2, n_questions=1)
    sweep_path = data_dir / "steering_sweep.jsonl"
    _write_sweep(sweep_path, records_v1)
    _write_meta(
        data_dir / "steering_sweep_meta.json", planned=4, attempted=4, complete=True,
        axis_run_id="axis-same",
    )
    client_1 = FakeClient(default_label="assistant")
    monkeypatch.setattr(ev, "build_default_client", lambda: client_1)
    first_exit = ev.main([])
    assert first_exit in (0, 1)
    assert client_1.calls > 0
    labels_path = data_dir / "steering_labels.json"
    assert labels_path.exists()

    # Sweep ARŞİVLENİP BAŞTAN yazıldı: AYNI axis_run_id (meta DEĞİŞMEDİ),
    # AYNI hücre sayıları, ama do_sample=True yüzünden TAMAMEN farklı metin.
    records_v2 = [dict(r, answer=r["answer"] + " (yeniden üretildi)") for r in records_v1]
    _write_sweep(sweep_path, records_v2)

    client_2 = FakeClient(default_label="assistant")
    monkeypatch.setattr(ev, "build_default_client", lambda: client_2)

    second_exit = ev.main([])

    assert second_exit in (0, 1)
    err = capsys.readouterr().err
    assert "UYARI" in err
    assert "SWEEP DEĞİŞMİŞ" in err
    assert client_2.calls > 0, "durum atıldıysa hakem YENİDEN sorulmalı — 0 çağrı KABUL EDİLEMEZ"
    assert (data_dir / "steering_labels.json.stale").exists(), "eski dosya SİLİNMEDİ, kenara alındı"
    new_payload = json.loads(labels_path.read_text(encoding="utf-8"))
    assert new_payload["sweep_sha256"] == _sweep_sha256(sweep_path)


def test_matching_sweep_fingerprint_reuses_saved_labels_without_recalling_the_judge(
    tmp_path, monkeypatch, capsys
):
    """Sweep DEĞİŞMEDEN aynı komut TEKRAR çalıştırılırsa parmak izi eşleşir,
    durum KULLANILIR, hakem HİÇ çağrılmaz."""
    data_dir, results_dir = _patch_paths(monkeypatch, tmp_path)
    records = _make_records(14, [0.0, -0.6], n_roles=2, n_questions=1)
    sweep_path = data_dir / "steering_sweep.jsonl"
    _write_sweep(sweep_path, records)
    _write_meta(
        data_dir / "steering_sweep_meta.json", planned=4, attempted=4, complete=True,
        axis_run_id="axis-reuse",
    )
    client_1 = FakeClient(default_label="assistant")
    monkeypatch.setattr(ev, "build_default_client", lambda: client_1)
    first_exit = ev.main([])
    assert first_exit in (0, 1)
    assert client_1.calls > 0

    client_2 = FakeClient(default_label="assistant")
    monkeypatch.setattr(ev, "build_default_client", lambda: client_2)
    second_exit = ev.main([])

    assert second_exit == first_exit
    assert client_2.calls == 0, "parmak izi eşleşiyor — hakem TEKRAR sorulmamalı"
    err = capsys.readouterr().err
    assert "UYARI: ESKİ ETİKETLER" not in err
    assert not (data_dir / "steering_labels.json.stale").exists()


def test_labels_discarded_when_cell_record_count_changed(tmp_path, monkeypatch, capsys):
    """`sweep_sha256`/`axis_run_id` doğru olsa bile `record_counts` sweep'in
    GERÇEK hücre sayılarıyla uyuşmuyorsa durum yine ATILIR — `record_counts`
    sha256'dan BAĞIMSIZ, ikinci bir savunma katmanı (madde 4'ün gerekçesi:
    `steering_labels.json` sweep DEĞİŞMEDEN de elle/başka bir yoldan
    bozulmuş olabilir)."""
    data_dir, results_dir = _patch_paths(monkeypatch, tmp_path)
    records = _make_records(14, [0.0, -0.6], n_roles=2, n_questions=1)  # hücre başına 2 kayıt
    sweep_path = data_dir / "steering_sweep.jsonl"
    _write_sweep(sweep_path, records)
    _write_meta(
        data_dir / "steering_sweep_meta.json", planned=4, attempted=4, complete=True,
        axis_run_id="axis-count",
    )
    (data_dir / "steering_labels.json").write_text(json.dumps({
        "labels": {"14|0.0": {"0": "assistant", "1": "assistant"}},
        "unlabelled_positions": {},
        "sweep_sha256": _sweep_sha256(sweep_path),
        "axis_run_id": "axis-count",
        "record_counts": {"14|0.0": 999, "14|-0.6": 2},  # 14|0.0 YANLIŞ: gerçeği 2
    }), encoding="utf-8")
    client = FakeClient(default_label="assistant")
    monkeypatch.setattr(ev, "build_default_client", lambda: client)

    exit_code = ev.main([])

    assert exit_code in (0, 1)
    err = capsys.readouterr().err
    assert "UYARI" in err
    assert "HÜCRE KAYIT SAYILARI UYUŞMUYOR" in err
    assert client.calls > 0, "kayıt sayısı uyuşmazlığında durum atılmalı, hakem tekrar sorulmalı"


def test_old_schema_labels_file_is_discarded_and_rejudged(tmp_path, monkeypatch, capsys):
    """Parmak izi alanlarını HİÇ taşımayan bir `steering_labels.json` (bu
    fix'ten ÖNCEKİ şema) güvenilmez sayılır — durum atılır, hakem yeniden
    sorulur."""
    data_dir, results_dir = _patch_paths(monkeypatch, tmp_path)
    records = _make_records(14, [0.0, -0.6], n_roles=2, n_questions=1)
    sweep_path = data_dir / "steering_sweep.jsonl"
    _write_sweep(sweep_path, records)
    _write_meta(data_dir / "steering_sweep_meta.json", planned=4, attempted=4, complete=True)
    # Bu fix'ten ÖNCEKİ şema: yalnızca "labels"/"unlabelled_positions" — hiç
    # parmak izi alanı yok.
    (data_dir / "steering_labels.json").write_text(json.dumps({
        "labels": {"14|0.0": {"0": "assistant", "1": "assistant"}},
        "unlabelled_positions": {},
    }), encoding="utf-8")
    client = FakeClient(default_label="assistant")
    monkeypatch.setattr(ev, "build_default_client", lambda: client)

    exit_code = ev.main([])

    assert exit_code in (0, 1)
    err = capsys.readouterr().err
    assert "UYARI" in err
    assert "ESKİ ŞEMA" in err
    assert client.calls > 0, "eski şema atıldıysa TÜM hücreler yeniden sorulmalı"
    new_payload = json.loads((data_dir / "steering_labels.json").read_text(encoding="utf-8"))
    assert new_payload["sweep_sha256"] == _sweep_sha256(sweep_path)


def test_stray_out_of_range_position_is_clipped_from_the_rate(tmp_path, monkeypatch):
    """F1 madde 4: fingerprint EŞLEŞSE bile (sha256/axis_run_id/record_counts
    hepsi doğru) bir hücrenin etiketleri `range(len(items_all))` DIŞINDA bir
    pozisyon barındırıyorsa, bu pozisyon orana KARIŞMAMALI. `record_counts`
    yalnızca hücre başına TOPLAM sayıyı doğrular — kaç AYRI POZİSYONUN
    kayıtlı olduğunu değil; bu yüzden bu, sha256/record_counts
    doğrulamasından BAĞIMSIZ bir savunma katmanı."""
    data_dir, results_dir = _patch_paths(monkeypatch, tmp_path)
    records = _make_records(14, [0.0, -0.6], n_roles=2, n_questions=1)  # hücre başına 2 kayıt
    sweep_path = data_dir / "steering_sweep.jsonl"
    _write_sweep(sweep_path, records)
    _write_meta(
        data_dir / "steering_sweep_meta.json", planned=4, attempted=4, complete=True,
        axis_run_id="axis-clip",
    )
    # (14, 0.0) hücresinin GERÇEK pozisyonları 0-1; 99 SIZMIŞ (bayat) bir
    # pozisyon — fingerprint'i BOZMAZ (record_counts yalnızca "bu hücrede 2
    # kayıt var" der, KAÇ POZİSYON kayıtlı olduğunu doğrulamaz).
    ev.save_group_labels(
        data_dir / "steering_labels.json",
        {
            (14, 0.0): {0: "assistant", 1: "assistant", 99: "human_role"},
            (14, -0.6): {0: "human_role", 1: "human_role"},
        },
        {(14, 0.0): set(), (14, -0.6): set()},
        sweep_sha256=_sweep_sha256(sweep_path),
        axis_run_id="axis-clip",
        record_counts={(14, 0.0): 2, (14, -0.6): 2},
    )
    client = FakeClient()  # her iki hücre de zaten "tamam" — hiç çağrılmamalı
    monkeypatch.setattr(ev, "build_default_client", lambda: client)

    exit_code = ev.main([])

    assert client.calls == 0, "kapsam (0,1) zaten dolu — sızmış pozisyon 99 YOKMUŞ gibi ele alınmalı"
    rate_payload = json.loads(
        (results_dir / "steering" / "rate_by_strength.json").read_text(encoding="utf-8")
    )
    # Kırpma YOKSA: 0.0 hücresinin oranı 1/3 olurdu (99'daki "human_role"
    # dahil). Kırpma VARSA: tam 0.0 (yalnızca pozisyon 0 ve 1, ikisi de
    # "assistant").
    assert rate_payload["14"]["0.0"] == pytest.approx(0.0)
    assert rate_payload["14"]["-0.6"] == pytest.approx(1.0)

    criterion_payload = json.loads(
        (results_dir / "steering" / "criterion_b.json").read_text(encoding="utf-8")
    )
    assert criterion_payload["labelled_by_group"]["14|0.0"] == 2, "payda 3 DEĞİL, kırpılmış 2 olmalı"
    assert criterion_payload["layers"]["14"]["passed"] is True
    assert exit_code == 0


# --- F2 (Fix Round 1): bütçe ön kontrolü resume'dan ÖNCE koşuyordu ----------


def test_budget_precheck_uses_pending_positions_not_total_after_resume(
    tmp_path, monkeypatch, capsys
):
    """Bütçe ön kontrolü artık diskte HAZIR olan pozisyonları düşüyor —
    resume'dan sonra kalan bütçe TOPLAM plandan küçük olsa bile koşu
    BAŞLAMALI (yalnızca BEKLEYEN pozisyonlar bütçeye sığdığı sürece).
    Eskiden `planned` TÜM gruplar üzerinden hesaplanıyordu ve
    `load_group_labels`'tan ÖNCE koşuyordu — diskte hazır olan hiçbir şey
    (maliyeti 0 olan cache isabetleri dahil) düşülmüyordu."""
    data_dir, results_dir = _patch_paths(monkeypatch, tmp_path)
    records = _make_records(14, [0.0, -0.6], n_roles=10, n_questions=1)  # hücre başına 10 öğe
    sweep_path = data_dir / "steering_sweep.jsonl"
    _write_sweep(sweep_path, records)
    _write_meta(
        data_dir / "steering_sweep_meta.json", planned=20, attempted=20, complete=True,
        axis_run_id="axis-budget",
    )
    # (14, -0.6) grubu TAMAMEN diskten hazır — bu, bütçe hesabına HİÇ girmemeli.
    ev.save_group_labels(
        data_dir / "steering_labels.json",
        {(14, -0.6): {i: "assistant" for i in range(10)}},
        {(14, -0.6): set()},
        sweep_sha256=_sweep_sha256(sweep_path),
        axis_run_id="axis-budget",
        record_counts={(14, -0.6): 10, (14, 0.0): 10},
    )
    # TOPLAM plan 2 batch (her hücre 1 batch, batch_size=10); BEKLEYEN plan
    # yalnızca 1 batch ((14, 0.0) grubu). stage_remaining=1 TOPLAM'a sığmaz
    # ama BEKLEYEN'e sığar.
    client = FakeClient(default_label="assistant", stage_remaining=1, global_remaining=1000)
    monkeypatch.setattr(ev, "build_default_client", lambda: client)

    exit_code = ev.main([])

    assert exit_code in (0, 1), "kalan bütçe BEKLEYEN plana sığdığı için koşu başlamalı"
    out = capsys.readouterr().out
    assert "toplam planlanan çağrı (üst sınır): 2" in out
    assert "bekleyen (bütçe kontrolü buna göre): 1" in out
    assert client.calls == 1


# --- F3 (Fix Round 1): criterion_b.json hücre başına PAYDAYI yazmıyordu ----


def test_criterion_b_records_the_denominator_per_group(tmp_path, monkeypatch):
    """Aynı orana (1.000) sahip iki FARKLI büyüklükteki hücre
    `rate_by_strength.json`'da bayt bayt AYNI görünür — `criterion_b.json`
    artık her hücrenin PAYDASINI ('labelled_by_group') AYRICA yazıyor,
    böylece '2 öğeden 1.000' ile '5 öğeden 1.000' artefaktın kendisinden
    AYIRT edilebiliyor."""
    data_dir, results_dir = _patch_paths(monkeypatch, tmp_path)
    records = (
        _make_records(14, [0.0, -0.6], n_roles=2, n_questions=1)  # hücre başına 2 öğe
        + _make_records(19, [0.0, -0.6], n_roles=5, n_questions=1)  # hücre başına 5 öğe
    )
    _write_sweep(data_dir / "steering_sweep.jsonl", records)
    _write_meta(data_dir / "steering_sweep_meta.json", planned=14, attempted=14, complete=True)
    client = FakeClient(
        label_rule=lambda block: "human_role" if "s-0.6" in block else "assistant"
    )
    monkeypatch.setattr(ev, "build_default_client", lambda: client)

    exit_code = ev.main([])

    assert exit_code == 0
    rate_payload = json.loads(
        (results_dir / "steering" / "rate_by_strength.json").read_text(encoding="utf-8")
    )
    # İKİ katmanın -0.6 oranı da BİREBİR aynı — 1.000. Bu dosyadan tek
    # başına 2'ye karşı 5 öğe AYIRT EDİLEMEZ.
    assert rate_payload["14"]["-0.6"] == pytest.approx(1.0)
    assert rate_payload["19"]["-0.6"] == pytest.approx(1.0)

    criterion_payload = json.loads(
        (results_dir / "steering" / "criterion_b.json").read_text(encoding="utf-8")
    )
    assert criterion_payload["labelled_by_group"]["14|-0.6"] == 2
    assert criterion_payload["labelled_by_group"]["19|-0.6"] == 5
    assert "labelled_by_group" in criterion_payload["note"]


# --- F4 (Fix Round 1): hakem gönderimlerinin --batch-size'da parçalandığı --
# hiçbir testle sabitlenmemişti -------------------------------------------


def test_judge_sends_are_split_according_to_batch_size(tmp_path, monkeypatch):
    """25 öğelik bir hücre, varsayılan --batch-size (10) ile hakeme
    [10, 10, 5] boyutlarında AYRI çağrılar olarak gönderilmeli — tek bir dev
    çağrı olarak DEĞİL. `pending`'i `args.batch_size` yerine
    `max(len(pending), 1)` adımıyla dilimleyen bir mutasyon bu testten ÖNCE
    27 testin hepsini geçiyordu (hiçbiri 10 öğeden büyük bir hücre
    kullanmıyordu)."""
    data_dir, results_dir = _patch_paths(monkeypatch, tmp_path)
    records = (
        _make_records(14, [-0.6], n_roles=25, n_questions=1)  # 25 öğe
        + _make_records(14, [0.0], n_roles=3, n_questions=1)  # 3 öğe (taban)
    )
    _write_sweep(data_dir / "steering_sweep.jsonl", records)
    _write_meta(data_dir / "steering_sweep_meta.json", planned=28, attempted=28, complete=True)
    client = FakeClient(default_label="assistant")
    monkeypatch.setattr(ev, "build_default_client", lambda: client)

    exit_code = ev.main([])

    assert exit_code in (0, 1)
    # sorted(groups) -> [(14, -0.6), (14, 0.0)]: önce 25 öğelik hücre
    # [10, 10, 5]'e bölünür, sonra 3 öğelik hücre TEK bir [3] çağrısı olur.
    assert client.sizes_seen == [10, 10, 5, 3]


def test_explicit_batch_size_flag_changes_the_actual_send_sizes(tmp_path, monkeypatch):
    """--batch-size YALNIZCA plan sayısını DEĞİL, hakeme GERÇEKTEN
    gönderilen parça boyutlarını da değiştirmeli."""
    data_dir, results_dir = _patch_paths(monkeypatch, tmp_path)
    records = (
        _make_records(14, [-0.6], n_roles=20, n_questions=1)  # 20 öğe
        + _make_records(14, [0.0], n_roles=1, n_questions=1)  # 1 öğe (taban)
    )
    _write_sweep(data_dir / "steering_sweep.jsonl", records)
    _write_meta(data_dir / "steering_sweep_meta.json", planned=21, attempted=21, complete=True)
    client = FakeClient(default_label="assistant")
    monkeypatch.setattr(ev, "build_default_client", lambda: client)

    exit_code = ev.main(["--batch-size", "7"])

    assert exit_code in (0, 1)
    assert client.sizes_seen == [7, 7, 6, 1]


# --- F5 (Fix Round 1): karar üretemeyen koşu karar artefaktlarını EZİYORDU --


def test_zero_record_sweep_does_not_overwrite_existing_decision_artifact(
    tmp_path, monkeypatch, capsys
):
    """Karar üretemeyen bir koşu (burada: sıfır kayıtlı sweep) mevcut
    `criterion_b.json`'ı artık EZMİYOR. Yazım bloğu eskiden KOŞULSUZDU ve
    `overall_exit_code`'dan ÖNCE çalışıyordu — script'in kendi
    konvansiyonunun (TÜM diğer çıkış-2 yolları yazımlardan ÖNCE döner) TEK
    istisnasıydı."""
    data_dir, results_dir = _patch_paths(monkeypatch, tmp_path)
    sweep_path = data_dir / "steering_sweep.jsonl"
    sweep_path.parent.mkdir(parents=True, exist_ok=True)
    sweep_path.write_text("", encoding="utf-8")  # sıfır kayıt
    _write_meta(data_dir / "steering_sweep_meta.json", planned=0, attempted=0, complete=True)
    steering_dir = results_dir / "steering"
    steering_dir.mkdir(parents=True, exist_ok=True)
    previous_payload = {"layers": {"14": {"passed": True}}, "labelled_by_group": {"14|0.0": 250}}
    (steering_dir / "criterion_b.json").write_text(
        json.dumps(previous_payload), encoding="utf-8"
    )
    client = FakeClient()
    monkeypatch.setattr(ev, "build_default_client", lambda: client)

    exit_code = ev.main([])

    assert exit_code == 2
    after_payload = json.loads((steering_dir / "criterion_b.json").read_text(encoding="utf-8"))
    assert after_payload == previous_payload, "boş sweep önceki GERÇEK kararı EZMEMELİ"
    assert client.calls == 0


# --- F8 (Fix Round 1): atomik yazımı hiçbir test sabitlemiyordu ------------


def test_atomic_write_failure_leaves_existing_labels_file_untouched(tmp_path, monkeypatch):
    """Task 4'teki `test_meta_write_failure_leaves_existing_file_untouched`
    (`tests/test_steering_sweep.py`) ile AYNI desen: `save_group_labels`'ın
    ATOMİK yazımı (tempfile + `os.replace`) `os.replace` ORTASINDA patlarsa
    var olan `steering_labels.json` DEĞİŞMEDEN kalmalı ve `.tmp` artığı
    sürünmemeli. Bu bloğu düz bir `path.write_text(...)` ile değiştiren bir
    mutasyon bu testten ÖNCE 27 testin hepsini geçiyordu — gösterilen emsal
    (`tests/test_label_and_train_probe.py:1118-1119`) yalnızca artık `.tmp`
    kalmamasını sabitliyordu, o da AYNI mutasyonu geçiyordu."""
    path = tmp_path / "steering_labels.json"
    path.write_text("ONCEKI-ICERIK", encoding="utf-8")

    def boom_replace(*_args, **_kwargs):
        raise RuntimeError("bilerek — os.replace ortasında patladı")

    monkeypatch.setattr(ev.os, "replace", boom_replace)

    with pytest.raises(RuntimeError):
        ev.save_group_labels(
            path,
            {(14, 0.0): {0: "assistant"}},
            {(14, 0.0): set()},
            sweep_sha256="deadbeef",
            axis_run_id="axis-1",
            record_counts={(14, 0.0): 1},
        )

    assert path.read_text(encoding="utf-8") == "ONCEKI-ICERIK"
    assert [p.name for p in tmp_path.iterdir()] == ["steering_labels.json"]
