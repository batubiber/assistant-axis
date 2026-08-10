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


def _write_meta(path: Path, *, planned: int, attempted: int, complete: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"planned": planned, "attempted": attempted, "complete": complete}),
        encoding="utf-8",
    )


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
    _write_sweep(data_dir / "steering_sweep.jsonl", records)
    _write_meta(data_dir / "steering_sweep_meta.json", planned=8, attempted=8, complete=True)
    ev.save_group_labels(
        data_dir / "steering_labels.json",
        {(14, -0.6): {0: "assistant", 1: "assistant", 2: "assistant", 3: "assistant"}},
        {(14, -0.6): set()},
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
    data_dir, results_dir = _patch_paths(monkeypatch, tmp_path)
    records = _make_records(14, [0.0, -0.6], n_roles=1, n_questions=1)
    _write_sweep(data_dir / "steering_sweep.jsonl", records)
    # meta dosyası KASITLI olarak hiç yazılmadı.

    exit_code = ev.main([])

    assert exit_code == 2
    err = capsys.readouterr().err
    assert "BAŞARISIZ" in err
    assert "Traceback" not in err


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
