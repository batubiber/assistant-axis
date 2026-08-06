"""`scripts/06_label_and_train_probe.py` karar mantığı VE dosya işleme testleri.

Ağa çıkmaz: gerçek `GatewayClient`'a hiç dokunulmaz, `build_default_client`
her testte monkeypatch'lenir. `embed_answers` de her testte sahte bir
fonksiyonla değiştirilir — bu paket hiçbir testte bge-m3'ü (birkaç GB)
yüklemez/indirmez; `tests/conftest.py` zaten soketleri kapatıp
`HF_HUB_OFFLINE=1` set ediyor, burada AYRICA gerçek `embed_answers`'ı hiç
çağırmamak ikinci bir savunma katmanı. Script dosya adı bir rakamla
başladığı için (`06_label_and_train_probe.py`) normal `import` ile içe
aktarılamaz; `importlib` ile dosya yolundan yüklenir (bkz.
`tests/test_smoke_gateway.py`, `tests/test_generate_role_data.py`,
`tests/test_judge_gate.py`).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from aax.gateway import BudgetExceeded, CircuitOpen, GatewayError
from aax.judge import JudgeParseError

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "06_label_and_train_probe.py"
)


def _load_script():
    spec = importlib.util.spec_from_file_location("label_and_train_probe", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ltp = _load_script()


def test_module_is_registered_in_sys_modules():
    """Repo kuralı (bkz. test_smoke_gateway.py, test_generate_role_data.py,
    test_judge_gate.py): rakamla başlayan script'i importlib ile yüklerken
    modülü sys.modules'e de kaydet."""
    assert sys.modules["label_and_train_probe"] is ltp


# --- ortak test yardımcıları --------------------------------------------------


def _patch_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(ltp, "LABELS_PATH", tmp_path / "probe_labels.json")
    monkeypatch.setattr(ltp, "OUT_PATH", tmp_path / "role_expression.json")
    monkeypatch.setattr(ltp.config, "DATA_DIR", tmp_path)


def _make_role_records(roles: list[str], per_role: int = 5) -> list[dict]:
    records = []
    for role in roles:
        for i in range(per_role):
            records.append(
                {
                    "kind": "role",
                    "role": role,
                    "system_prompt": f"You are a {role}.",
                    "question": f"{role} question {i}",
                    "sample_index": i,
                    "answer": f"{role} answer {i}",
                }
            )
    return records


def _write_rollouts(path: Path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )


def _write_roles_catalog(path: Path, roles: list[str]) -> None:
    payload = {
        "complete": True,
        "limit": None,
        "requested": len(roles),
        "catalog_size": len(roles),
        "roles": [
            {
                "role": r,
                "description": f"the role of a {r}",
                "instructions": ["x"],
                "questions": ["q"],
            }
            for r in roles
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _fake_embed_answers(answers, *, model_id: str = "BAAI/bge-m3") -> np.ndarray:
    """Ağsız, deterministik sahte embedding. Değerlerin kendisi çoğu testte
    önemsiz — yalnızca bge-m3'ün asla yüklenmediğini garanti eder."""
    return np.array(
        [[len(a), sum(ord(c) for c in a) % 97, hash(a) % 101] for a in answers],
        dtype=float,
    )


class FakeJudgeClient:
    """Ağsız sahte hakem istemcisi.

    `role_scores`: rol adı -> o role ait TÜM cevaplara verilecek sabit 0-3
    puanı. Prompt içindeki `"the role: {role}."` alt dizesinden hangi rolün
    puanlandığını çözer (bkz. `aax.judge._build_prompt`), böylece
    `stratified_sample`'ın hangi satırları seçtiğini bilmeye gerek kalmaz.
    `exceptions`: verilirse, sırayla `.chat()` çağrılarının ÖNÜNE geçip
    fırlatılır (BudgetExceeded/CircuitOpen/GatewayError senaryoları için).
    """

    def __init__(
        self,
        role_scores: dict[str, int],
        *,
        exceptions: list | None = None,
        cached_prompts: set[str] | None = None,
        stage_remaining: int = 10_000,
        global_remaining: int = 10_000,
    ) -> None:
        self._role_scores = dict(role_scores)
        self._exceptions = list(exceptions or [])
        # `--dry-run` artık gerçek istemcinin API'sini kullanıyor:
        # `would_call()` (cache'te olan çağrıyı saymaz) + `remaining_budget()`
        # (tavan değil, diskteki sayaca göre KALAN).
        self._cached = set(cached_prompts or ())
        self._stage_remaining = stage_remaining
        self._global_remaining = global_remaining
        self.would_call_count = 0
        self.calls = 0
        self.sends_made = 0

    def would_call(self, messages, *, temperature=0.0, max_tokens=1024):
        self.would_call_count += 1
        return messages[0]["content"] not in self._cached

    def remaining_budget(self, stage):
        return self._stage_remaining, self._global_remaining

    def chat(self, messages, *, stage, temperature=0.0, max_tokens=1024):
        self.calls += 1
        self.sends_made += 1
        if self._exceptions:
            outcome = self._exceptions.pop(0)
            raise outcome
        content = messages[0]["content"]
        role = next(r for r in self._role_scores if f"the role: {r}." in content)
        n_items = content.count("[ITEM ")
        return json.dumps([self._role_scores[role]] * n_items)


def _make_fixed_probe_class(*, trustworthy: bool, predict_label: str = "no"):
    """`RoleExpressionProbe` yerine geçen, kapı davranışını istatistiksel
    gürültüden bağımsız test etmeye yarayan sabit bir sahte sınıf."""

    class _FixedProbe:
        def __init__(self, *, seed: int = 0) -> None:
            self.seed = seed
            self.holdout_agreement = 0.99 if trustworthy else 0.5

        def fit(self, embeddings, labels) -> None:
            self._fit_labels = list(labels)

        @property
        def is_trustworthy(self) -> bool:
            return trustworthy

        def predict(self, embeddings):
            return [predict_label] * len(embeddings)

    return _FixedProbe


# --- collapse() (saf, ağsız) ---------------------------------------------------


def test_collapse_maps_scores_to_three_categories():
    assert ltp.collapse(3) == "fully"
    assert ltp.collapse(2) == "somewhat"
    assert ltp.collapse(1) == "no"
    assert ltp.collapse(0) == "no"


def test_collapse_rejects_out_of_range():
    with pytest.raises(ValueError):
        ltp.collapse(4)


# --- --dry-run: bütçe aritmetiği ------------------------------------------------


def test_dry_run_reports_plan_and_exits_zero_when_under_budget(tmp_path, monkeypatch, capsys):
    _patch_paths(monkeypatch, tmp_path)
    roles = ["pirate", "sage"]
    _write_roles_catalog(tmp_path / "roles.json", roles)
    _write_rollouts(tmp_path / "rollouts.jsonl", _make_role_records(roles, per_role=10))
    monkeypatch.setattr(
        ltp, "build_default_client", lambda: FakeJudgeClient({}, stage_remaining=42)
    )

    exit_code = ltp.main(["--dry-run", "--sample-size", "20"])

    assert exit_code == 0
    out = capsys.readouterr().out
    cap = ltp.config.STAGE_BUDGETS[ltp.STAGE]
    assert "Planlanan çağrı:      2" in out
    # C3: KALAN bütçe raporlanmalı, sadece tavan değil.
    assert f"Aşama bütçesi:        {cap} (kalan: 42)" in out
    assert f"Global tavan:         {ltp.config.GLOBAL_BUDGET} (kalan:" in out
    # Birim uyarısı: sayaç HTTP gönderimi sayar, mantıksal çağrı değil.
    assert "gönderim" in out


def test_dry_run_compares_the_plan_against_the_remaining_budget_not_the_cap(
    tmp_path, monkeypatch, capsys
):
    """C3 regresyonu: aşama TAVANI 300 ve plan 2 çağrı — statik kıyaslama
    temiz bir `0` derdi. Ama tavanın 299'u önceki bir koşuda harcanmışsa
    kalan 1'dir ve koşu ortasında kesilirdi. `remaining_budget()` tam bu iş
    için var."""
    _patch_paths(monkeypatch, tmp_path)
    roles = ["pirate", "sage"]
    _write_roles_catalog(tmp_path / "roles.json", roles)
    _write_rollouts(tmp_path / "rollouts.jsonl", _make_role_records(roles, per_role=10))
    monkeypatch.setattr(
        ltp, "build_default_client", lambda: FakeJudgeClient({}, stage_remaining=1)
    )

    exit_code = ltp.main(["--dry-run", "--sample-size", "20"])

    # D8: ön koşul hatası artık 2 — çıkış 1 bu script'te "probe güvenilmez".
    assert exit_code == 2
    err = capsys.readouterr().err
    assert "HATA" in err
    assert "yalnızca 1 kaldı" in err
    assert "--sample-size" in err


def test_dry_run_fails_when_the_global_cap_is_exhausted(tmp_path, monkeypatch, capsys):
    """Aşama bütçesi bol olsa bile 1500'lük global tavan dolmuş olabilir."""
    _patch_paths(monkeypatch, tmp_path)
    roles = ["pirate", "sage"]
    _write_roles_catalog(tmp_path / "roles.json", roles)
    _write_rollouts(tmp_path / "rollouts.jsonl", _make_role_records(roles, per_role=10))
    monkeypatch.setattr(
        ltp,
        "build_default_client",
        lambda: FakeJudgeClient({}, stage_remaining=300, global_remaining=0),
    )

    assert ltp.main(["--dry-run", "--sample-size", "20"]) == 2
    assert "global tavan" in capsys.readouterr().err


def test_dry_run_does_not_count_cached_calls(tmp_path, monkeypatch, capsys):
    """Eski plan `(len(rows) + 9) // 10` idi ve cache'i yok sayıyordu.
    `would_call()` yalnızca GERÇEKTEN harcanacak çağrıyı sayar."""
    from aax.judge import build_role_score_prompts

    _patch_paths(monkeypatch, tmp_path)
    roles = ["pirate", "sage"]
    records = _make_role_records(roles, per_role=10)
    _write_roles_catalog(tmp_path / "roles.json", roles)
    _write_rollouts(tmp_path / "rollouts.jsonl", records)

    # "pirate" rolünün promptunu cache'te sayalım: plan 2 değil 1 olmalı.
    pirate_items = [(r["question"], r["answer"]) for r in records if r["role"] == "pirate"]
    cached = set(
        build_role_score_prompts(
            role="pirate", description="the role of a pirate", items=pirate_items
        )
    )
    monkeypatch.setattr(
        ltp, "build_default_client", lambda: FakeJudgeClient({}, cached_prompts=cached)
    )

    assert ltp.main(["--dry-run", "--sample-size", "20"]) == 0
    out = capsys.readouterr().out
    assert "Planlanan çağrı:      1 (cache'te: 1)" in out


def test_dry_run_plans_the_prompts_that_would_actually_be_sent(tmp_path, monkeypatch):
    """`would_call`'ın anlamlı olabilmesi için ön kontrol, `chat()`'in
    kuracağı payload'ın AYNISINI kurmalı: aynı prompt, aynı sıcaklık, aynı
    max_tokens — cache anahtarı bunlardan türetiliyor."""
    from aax.judge import SCORE_MAX_TOKENS, SCORE_TEMPERATURE

    _patch_paths(monkeypatch, tmp_path)
    roles = ["pirate"]
    _write_roles_catalog(tmp_path / "roles.json", roles)
    _write_rollouts(tmp_path / "rollouts.jsonl", _make_role_records(roles, per_role=5))

    seen = []

    class SpyClient(FakeJudgeClient):
        def would_call(self, messages, *, temperature=0.0, max_tokens=1024):
            seen.append((messages[0]["content"], temperature, max_tokens))
            return True

    monkeypatch.setattr(ltp, "build_default_client", lambda: SpyClient({}))

    assert ltp.main(["--dry-run", "--sample-size", "5"]) == 0

    assert seen, "hiç plan kurulmadı"
    prompt, temperature, max_tokens = seen[0]
    assert temperature == SCORE_TEMPERATURE
    assert max_tokens == SCORE_MAX_TOKENS
    assert "the role: pirate." in prompt  # gerçek hakem promptu


def test_dry_run_sends_no_requests(tmp_path, monkeypatch):
    _patch_paths(monkeypatch, tmp_path)
    roles = ["pirate"]
    _write_roles_catalog(tmp_path / "roles.json", roles)
    _write_rollouts(tmp_path / "rollouts.jsonl", _make_role_records(roles, per_role=5))
    client = FakeJudgeClient({})
    monkeypatch.setattr(ltp, "build_default_client", lambda: client)

    ltp.main(["--dry-run", "--sample-size", "5"])

    assert client.calls == 0, "--dry-run tek bir hakem çağrısı bile atmamalı"
    assert client.would_call_count > 0, "plan `would_call` üzerinden kurulmalı"


# --- Bulgu 1: kurulum aşaması hataları temiz tanıya çevrilir --------------------


def test_main_diagnoses_oversized_sample_size_cleanly(tmp_path, monkeypatch, capsys):
    """Bulgu senaryosu: 90 satırlık smoke veri setine karşı varsayılan 2000
    örneklem boyutu çıplak bir ValueError traceback'i fırlatıyordu."""
    _patch_paths(monkeypatch, tmp_path)
    roles = ["pirate"]
    _write_roles_catalog(tmp_path / "roles.json", roles)
    _write_rollouts(tmp_path / "rollouts.jsonl", _make_role_records(roles, per_role=5))
    build_calls = {"n": 0}

    def fake_build():
        build_calls["n"] += 1
        return FakeJudgeClient({})

    monkeypatch.setattr(ltp, "build_default_client", fake_build)

    exit_code = ltp.main(["--sample-size", "50"])

    assert exit_code == 2
    err = capsys.readouterr().err
    assert "BAŞARISIZ" in err
    assert "50" in err
    assert "5" in err
    assert "--sample-size" in err
    assert "Traceback" not in err
    assert build_calls["n"] == 0, "örnekleme başarısız olduysa istemci hiç kurulmamalı"


def test_main_diagnoses_non_canonical_role_catalog_cleanly(tmp_path, monkeypatch, capsys):
    _patch_paths(monkeypatch, tmp_path)
    _write_rollouts(tmp_path / "rollouts.jsonl", _make_role_records(["pirate"], per_role=5))
    # Kanonik olmayan katalog: complete=False (ör. Aşama 0 yarıda kesilmiş).
    (tmp_path / "roles.json").write_text(
        json.dumps({"complete": False, "limit": None, "requested": 1, "catalog_size": 1, "roles": []}),
        encoding="utf-8",
    )
    build_calls = {"n": 0}

    def fake_build():
        build_calls["n"] += 1
        return FakeJudgeClient({})

    monkeypatch.setattr(ltp, "build_default_client", fake_build)

    exit_code = ltp.main(["--sample-size", "5"])

    assert exit_code == 2
    err = capsys.readouterr().err
    assert "BAŞARISIZ" in err
    assert "kanonik" in err
    assert "Traceback" not in err
    assert build_calls["n"] == 0


def test_main_fails_closed_when_sampled_role_missing_from_catalog(tmp_path, monkeypatch, capsys):
    """Bulgu (Minor): `catalog.get(role, f"the role of a {role}")` sessizce
    jenerik bir açıklama uyduruyordu — bu, hemen üstündeki fail-closed
    yorumla çelişiyordu. Artık örneklenen bir rol katalogda yoksa reddedilir."""
    _patch_paths(monkeypatch, tmp_path)
    _write_roles_catalog(tmp_path / "roles.json", ["pirate"])  # "sage" katalogda YOK
    _write_rollouts(
        tmp_path / "rollouts.jsonl", _make_role_records(["pirate", "sage"], per_role=5)
    )
    build_calls = {"n": 0}

    def fake_build():
        build_calls["n"] += 1
        return FakeJudgeClient({})

    monkeypatch.setattr(ltp, "build_default_client", fake_build)

    exit_code = ltp.main(["--sample-size", "10"])

    assert exit_code == 2
    err = capsys.readouterr().err
    assert "BAŞARISIZ" in err
    assert "sage" in err
    assert "Traceback" not in err
    assert build_calls["n"] == 0, "eksik rol tespit edildiyse istemci hiç kurulmamalı"


def test_main_diagnoses_probe_fit_failure_cleanly(tmp_path, monkeypatch, capsys):
    """Bulgu senaryosu: 2.000 satırlık bir örneklemde nadir 'somewhat'
    kategorisi tek bir üyeye düşerse `train_test_split(..., stratify=...)`
    çıplak bir ValueError fırlatıyordu."""
    _patch_paths(monkeypatch, tmp_path)
    roles = ["pirate"]
    _write_roles_catalog(tmp_path / "roles.json", roles)
    _write_rollouts(tmp_path / "rollouts.jsonl", _make_role_records(roles, per_role=10))
    monkeypatch.setattr(ltp, "embed_answers", _fake_embed_answers)

    # Tek batch, 10 öğe: 9'u "fully" (3), 1'i "somewhat" (2) — stratify için
    # "somewhat" sınıfında yalnızca 1 üye kalır.
    scores = [3] * 9 + [2]

    class SkewedClient:
        sends_made = 0

        def chat(self, messages, *, stage, temperature=0.0, max_tokens=1024):
            self.sends_made += 1
            return json.dumps(scores)

    monkeypatch.setattr(ltp, "build_default_client", lambda: SkewedClient())

    exit_code = ltp.main(["--sample-size", "10"])

    assert exit_code == 2
    err = capsys.readouterr().err
    assert "BAŞARISIZ" in err
    assert "probe eğitilemedi" in err
    assert "Traceback" not in err
    # Hakem etiketleri zaten diske yazılmış olmalı — bu, kaybolan bir
    # koşunun kalıcı, kullanılabilir bir yan ürünü.
    assert ltp.LABELS_PATH.exists()
    assert not ltp.OUT_PATH.exists()


# --- eksik anahtar → temiz tanı, çıkış kodu 2 -----------------------------------


def test_main_missing_api_key_produces_diagnostic(tmp_path, monkeypatch, capsys):
    _patch_paths(monkeypatch, tmp_path)
    roles = ["pirate"]
    _write_roles_catalog(tmp_path / "roles.json", roles)
    _write_rollouts(tmp_path / "rollouts.jsonl", _make_role_records(roles, per_role=5))

    def raise_missing_key():
        raise RuntimeError(
            "APP_KEY_JAILBREAK ortam değişkeni tanımlı değil. "
            "Dağıtım ortamınızın .env dosyasından alıp kabuğunuzda export edin."
        )

    monkeypatch.setattr(ltp, "build_default_client", raise_missing_key)

    exit_code = ltp.main(["--sample-size", "5"])

    assert exit_code == 2
    err = capsys.readouterr().err
    assert "BAŞARISIZ" in err
    assert "gateway istemcisi kurulamadı" in err
    assert "APP_KEY_JAILBREAK" in err
    assert "Traceback" not in err


# --- BudgetExceeded / CircuitOpen / GatewayError / JudgeParseError -------------


@pytest.mark.parametrize(
    "istisna",
    [
        BudgetExceeded("'stage2_probe_labels' aşama bütçesi doldu: 300/300"),
        CircuitOpen("devre kesici açık"),
    ],
)
def test_main_stops_on_budget_or_circuit_exceptions(tmp_path, monkeypatch, capsys, istisna):
    _patch_paths(monkeypatch, tmp_path)
    roles = ["pirate"]
    _write_roles_catalog(tmp_path / "roles.json", roles)
    _write_rollouts(tmp_path / "rollouts.jsonl", _make_role_records(roles, per_role=5))
    client = FakeJudgeClient({"pirate": 3}, exceptions=[istisna])
    monkeypatch.setattr(ltp, "build_default_client", lambda: client)

    exit_code = ltp.main(["--sample-size", "5"])

    assert exit_code == 2
    err = capsys.readouterr().err
    assert "DURDURULDU" in err
    assert not ltp.LABELS_PATH.exists()
    assert not ltp.OUT_PATH.exists()


@pytest.mark.parametrize(
    "istisna",
    [
        GatewayError("HTTP 500"),
        JudgeParseError("bozuk JSON"),
    ],
)
def test_main_reports_gateway_or_judge_parse_failures(tmp_path, monkeypatch, capsys, istisna):
    _patch_paths(monkeypatch, tmp_path)
    roles = ["pirate"]
    _write_roles_catalog(tmp_path / "roles.json", roles)
    _write_rollouts(tmp_path / "rollouts.jsonl", _make_role_records(roles, per_role=5))
    client = FakeJudgeClient({"pirate": 3}, exceptions=[istisna])
    monkeypatch.setattr(ltp, "build_default_client", lambda: client)

    exit_code = ltp.main(["--sample-size", "5"])

    assert exit_code == 2
    err = capsys.readouterr().err
    assert "BAŞARISIZ" in err
    assert not ltp.LABELS_PATH.exists()


# --- is_trustworthy kapısı: güvenilmez probe role_expression.json'a asla yazmaz


def test_untrustworthy_probe_does_not_write_role_expression_and_exits_nonzero(
    tmp_path, monkeypatch, capsys
):
    _patch_paths(monkeypatch, tmp_path)
    roles = ["pirate", "sage"]
    _write_roles_catalog(tmp_path / "roles.json", roles)
    _write_rollouts(tmp_path / "rollouts.jsonl", _make_role_records(roles, per_role=5))
    client = FakeJudgeClient({"pirate": 3, "sage": 0})
    monkeypatch.setattr(ltp, "build_default_client", lambda: client)
    monkeypatch.setattr(ltp, "embed_answers", _fake_embed_answers)
    monkeypatch.setattr(ltp, "RoleExpressionProbe", _make_fixed_probe_class(trustworthy=False))

    exit_code = ltp.main(["--sample-size", "10"])

    assert exit_code == 1
    assert not ltp.OUT_PATH.exists(), "güvenilmez probe role_expression.json yazmamalı"
    assert ltp.LABELS_PATH.exists(), "hakem etiketleri yine de kalıcı olmalı"
    err = capsys.readouterr().err
    assert "GÜVENİLİR DEĞİL" in err


def test_untrustworthy_probe_message_is_actionable(tmp_path, monkeypatch, capsys):
    """Mesaj spec'in geri çekilme kuralını DUYURUYOR ama o kural bu dalda
    UYGULANMADI (kapsam dışı, bilinçli). O hâlde mesaj, operatörün gerçekten
    yapabileceği şeyleri söylemeli: uygulanmadığını, iki seçeneği ve harcanan
    bütçeyi."""
    _patch_paths(monkeypatch, tmp_path)
    roles = ["pirate", "sage"]
    _write_roles_catalog(tmp_path / "roles.json", roles)
    _write_rollouts(tmp_path / "rollouts.jsonl", _make_role_records(roles, per_role=5))
    monkeypatch.setattr(
        ltp,
        "build_default_client",
        lambda: FakeJudgeClient({"pirate": 3, "sage": 0}, stage_remaining=280),
    )
    monkeypatch.setattr(ltp, "embed_answers", _fake_embed_answers)
    monkeypatch.setattr(ltp, "RoleExpressionProbe", _make_fixed_probe_class(trustworthy=False))

    assert ltp.main(["--sample-size", "10"]) == 1

    err = capsys.readouterr().err
    assert "UYGULANMADI" in err  # otomatik geri çekilme yok
    assert "--sample-size" in err  # seçenek 1
    assert "somewhat" in err  # seçenek 2
    assert "gönderim" in err and "kalan: 280" in err  # harcanan/kalan bütçe


# --- B3: role_expression.json koşu kimliği taşımalı --------------------------


def test_role_expression_carries_the_rollout_run_id(tmp_path, monkeypatch):
    """`07_extract_axis.py` bunu `activations_index.json`'ınkiyle karşılaştırıp
    eşit değilse çıkış 2 veriyor. Onsuz, Aşama 1'in aynı satır sayısı ve
    sırasıyla FARKLI bir rol kümesiyle yeniden koşturulması 07'nin sayı ve
    kapsama kontrollerinin İKİSİNİ de geçiyordu."""
    from aax.rollouts import rollouts_run_id

    _patch_paths(monkeypatch, tmp_path)
    roles = ["pirate", "sage"]
    records = _make_role_records(roles, per_role=5)
    _write_roles_catalog(tmp_path / "roles.json", roles)
    _write_rollouts(tmp_path / "rollouts.jsonl", records)
    monkeypatch.setattr(
        ltp, "build_default_client", lambda: FakeJudgeClient({"pirate": 3, "sage": 0})
    )
    monkeypatch.setattr(ltp, "embed_answers", _fake_embed_answers)
    monkeypatch.setattr(ltp, "RoleExpressionProbe", _make_fixed_probe_class(trustworthy=True))

    assert ltp.main(["--sample-size", "10"]) == 0

    payload = json.loads(ltp.OUT_PATH.read_text(encoding="utf-8"))
    assert payload["run_id"] == rollouts_run_id(records)
    assert len(payload["run_id"]) == 16


# --- verbatim etiket önceliği: hakem etiketleri tahminle EZİLMEZ ---------------


def test_trustworthy_probe_preserves_judge_labels_verbatim_and_applies_predictions_elsewhere(
    tmp_path, monkeypatch
):
    """Güvenilir bir probe ile: hakem etiketli satırlar için nihai kategori
    HER ZAMAN hakemin verdiği kategoridir (fully/no), probe "somewhat"
    tahmin etse bile. Hakem etiketi OLMAYAN satırlar için ise nihai kategori
    tahmindir ("somewhat") — iki yön de aynı testte doğrulanır."""
    _patch_paths(monkeypatch, tmp_path)
    roles = ["pirate", "sage"]
    _write_roles_catalog(tmp_path / "roles.json", roles)
    # 20 rol satırı (rol başına 10); yalnızca yarısı hakeme sorulacak.
    _write_rollouts(tmp_path / "rollouts.jsonl", _make_role_records(roles, per_role=10))
    client = FakeJudgeClient({"pirate": 3, "sage": 0})  # pirate->fully, sage->no
    monkeypatch.setattr(ltp, "build_default_client", lambda: client)
    monkeypatch.setattr(ltp, "embed_answers", _fake_embed_answers)
    # Probe HER satır için "somewhat" tahmin eder — ne pirate'ın "fully"
    # etiketiyle ne sage'in "no" etiketiyle eşleşir.
    monkeypatch.setattr(
        ltp, "RoleExpressionProbe", _make_fixed_probe_class(trustworthy=True, predict_label="somewhat")
    )

    exit_code = ltp.main(["--sample-size", "10"])  # 20 satırın yalnızca 10'u hakeme sorulur

    assert exit_code == 0
    labels_payload = json.loads(ltp.LABELS_PATH.read_text(encoding="utf-8"))
    judge_labels = labels_payload["labels"]
    expression = json.loads(ltp.OUT_PATH.read_text(encoding="utf-8"))["expression"]

    assert len(judge_labels) == 10
    assert len(expression) == 20

    for row, category in judge_labels.items():
        assert category in ("fully", "no")
        assert expression[row] == category, (
            "hakem etiketi tahminle EZİLMEMELİ — labels.get(row, pred) önceliği bozuldu"
        )
        assert expression[row] != "somewhat"

    probe_only_rows = set(expression) - set(judge_labels)
    assert len(probe_only_rows) == 10
    for row in probe_only_rows:
        assert expression[row] == "somewhat", "hakem etiketi olmayan satır tahminden gelmeli"


# --- Bulgu 2: tek geçişli embedding — etiketli satırlar doğru embedding'e eşlenir


def test_labelled_rows_are_indexed_to_the_correct_embeddings_after_single_pass(
    tmp_path, monkeypatch
):
    """`embed_answers` artık TÜM rol yanıtları için TEK SEFERDE çağrılır ve
    hakem etiketli alt küme bu tek dizinin içinden indekslenir. Bu test,
    indekslemenin `role_rows`'taki KONUMA göre yapıldığını (global `row`
    numarasına göre DEĞİL) doğrudan kanıtlar: kayıtların başına "role"
    olmayan satırlar eklenerek konum != global satır numarası hale getirilir
    — yanlış bir indeksleme (`row` ile indekslemek gibi) burada ya yanlış
    embedding eşlerdi ya da dizi sınırlarını aşardı.
    """
    _patch_paths(monkeypatch, tmp_path)
    roles = ["pirate", "sage", "engineer"]
    _write_roles_catalog(tmp_path / "roles.json", roles)

    # Baştaki 5 satır "default" türünde — role_rows'ta yer almaz, böylece
    # her rol satırının `records` içindeki global konumu, `role_rows`
    # içindeki konumundan FARKLI olur.
    default_records = [
        {
            "kind": "default",
            "role": None,
            "system_prompt": None,
            "question": f"default question {i}",
            "sample_index": i,
            "answer": f"default answer {i}",
        }
        for i in range(5)
    ]
    role_records = _make_role_records(roles, per_role=10)  # 30 rol satırı
    _write_rollouts(tmp_path / "rollouts.jsonl", default_records + role_records)

    client = FakeJudgeClient({"pirate": 3, "sage": 2, "engineer": 0})
    monkeypatch.setattr(ltp, "build_default_client", lambda: client)

    captured: dict[str, object] = {}

    def spy_embed_answers(answers, *, model_id: str = "BAAI/bge-m3"):
        captured["embed_call_answers"] = list(answers)
        # k'ıncı cevabın embedding'i skaler k'dır — bu, ÇAĞRI İÇİNDEKİ
        # konumu doğrudan kodlar, indeksleme kaymasını ortaya çıkarır.
        return np.arange(len(answers), dtype=float).reshape(-1, 1)

    class SpyProbe:
        def __init__(self, *, seed: int = 0) -> None:
            self.holdout_agreement = 0.99

        def fit(self, embeddings, labels) -> None:
            captured["fit_embeddings"] = np.asarray(embeddings).reshape(-1).tolist()
            captured["fit_labels"] = list(labels)

        @property
        def is_trustworthy(self) -> bool:
            return True

        def predict(self, embeddings):
            captured["predict_embeddings"] = np.asarray(embeddings).reshape(-1).tolist()
            return ["no"] * len(embeddings)

    monkeypatch.setattr(ltp, "embed_answers", spy_embed_answers)
    monkeypatch.setattr(ltp, "RoleExpressionProbe", SpyProbe)

    # 9 örnek (rol başına 3) — 30 rol satırının yalnızca bir kısmı hakeme
    # sorulur, geri kalanı yalnızca `embed_answers`'ın TEK çağrısından gelir.
    exit_code = ltp.main(["--sample-size", "9"])
    assert exit_code == 0

    # `embed_answers` TEK sefer çağrılmış olmalı (Bulgu 2'nin özü).
    assert captured["embed_call_answers"] == [r["answer"] for r in role_records], (
        "embed_answers rol satırları için (ve YALNIZCA onlar için) TEK çağrıda, "
        "role_rows sırasıyla çağrılmalı"
    )

    # role_rows == records[5:] (ilk 5 satır "default"), bu yüzden k'ıncı rol
    # satırının role_rows'taki konumu tam olarak k'dır — spy_embed_answers'ın
    # döndürdüğü embedding de tam olarak k'dır. Etiketli her satır için
    # fit()'e giden embedding'in bu beklenen konum değeriyle eşleştiğini
    # doğrudan doğrula.
    labels_payload = json.loads(ltp.LABELS_PATH.read_text(encoding="utf-8"))
    judge_rows = sorted(int(k) for k in labels_payload["labels"])
    expected_positions = [row - len(default_records) for row in judge_rows]
    assert captured["fit_embeddings"] == expected_positions, (
        "etiketli satırlar YANLIŞ embedding'e eşlenmiş — tek geçişli "
        "indeksleme (role_row_position) bozuk"
    )
    # predict() TÜM 30 rol satırı için, role_rows sırasıyla çağrılmalı.
    assert captured["predict_embeddings"] == list(range(30))
