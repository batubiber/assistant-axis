"""Aşama 0 script'inin dosya adlandırma / tamlık mantığı testleri.

Ağa çıkmaz: `write_artifacts()`, `run_generation_loop()` ve yardımcılarını
doğrudan `tmp_path`'e / sahte bir istemciye karşı çağırır, gerçek
`GatewayClient`'a hiç dokunmaz. Script dosya adı bir rakamla başladığı için
(`00_generate_role_data.py`) normal `import` ile içe aktarılamaz; `importlib`
ile dosya yolundan yüklenir.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest

from aax.gateway import (
    BudgetCorrupted,
    BudgetExceeded,
    CircuitOpen,
    GatewayClient,
    GatewayConfig,
    GatewayError,
)

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "00_generate_role_data.py"
)


def _load_script():
    spec = importlib.util.spec_from_file_location("generate_role_data", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


grd = _load_script()


def make_record(role: str, n_questions: int = 5) -> dict:
    return {
        "role": role,
        "description": f"a {role}",
        "instructions": ["a", "b", "c"],
        "questions": [f"{role}-q{i}" for i in range(n_questions)],
    }


def raw_response_for(role: str, n_questions: int = 40) -> str:
    """`parse_generation_response`'un kabul edeceği ham (JSON) model yanıtı."""
    payload = {
        "description": f"a {role}",
        "instructions": ["a", "b", "c"],
        "questions": [f"{role}-q{i}" for i in range(n_questions)],
    }
    return json.dumps(payload)


class StubClient:
    """`run_generation_loop` için ağsız sahte gateway istemcisi.

    `.chat()` çağrılarını sırayla `responses`'tan karşılar: öğe bir
    `BaseException` ise fırlatılır, değilse ham içerik string'i olarak
    döner. Liste tükenirse son öğe tekrar edilir. Gerçek `GatewayClient`'a
    hiç dokunulmaz.
    """

    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self.calls = 0
        self.sends_made = 0  # gerçek GatewayClient'ın tanı sayacı

    def chat(self, messages, *, stage, temperature=0.0, max_tokens=1024):
        self.calls += 1
        self.sends_made += 1
        index = min(self.calls - 1, len(self._responses) - 1)
        outcome = self._responses[index]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def main_with(argv: list[str]) -> int:
    """`main()`'i sys.argv'ye dokunmadan çağır."""
    return grd.main(argv)


def make_gateway_client(tmp_path, response_text: str):
    """Sahte transport'lu GERÇEK `GatewayClient` — cache ve bütçe gerçek.

    `would_call`/`chat` cache anahtarı eşdeğerliğini ancak gerçek istemci
    doğrulayabilir; `StubClient`'ın cache'i yok.
    """
    calls: list[dict] = []

    def transport(payload):
        calls.append(payload)
        return 200, {
            "choices": [{"message": {"content": response_text}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    cfg = GatewayConfig(
        base_url="https://example.invalid/Jailbreak",
        model="hakem-llm",
        api_key="test-key",
        stage_budgets={grd.STAGE: 145},
        global_budget=1500,
    )
    client = GatewayClient(
        cfg,
        cache_dir=tmp_path / "cache",
        budget_path=tmp_path / "budget.json",
        log_path=tmp_path / "calls.jsonl",
        transport=transport,
        monotonic=lambda: 0.0,
        sleep=lambda _seconds: None,
    )
    return client, calls


# --- resolve_artifact_paths (pure, no I/O) ----------------------------------


def test_resolve_paths_uses_canonical_names_when_complete(tmp_path):
    roles_path, questions_path = grd.resolve_artifact_paths(
        tmp_path, complete=True, allow_partial=False
    )
    assert roles_path == tmp_path / "roles.json"
    assert questions_path == tmp_path / "questions.json"


def test_resolve_paths_uses_partial_names_when_incomplete_and_not_allowed(tmp_path):
    roles_path, questions_path = grd.resolve_artifact_paths(
        tmp_path, complete=False, allow_partial=False
    )
    assert roles_path == tmp_path / "roles.partial.json"
    assert questions_path == tmp_path / "questions.partial.json"


def test_resolve_paths_promotes_to_canonical_names_when_allow_partial(tmp_path):
    roles_path, questions_path = grd.resolve_artifact_paths(
        tmp_path, complete=False, allow_partial=True
    )
    assert roles_path == tmp_path / "roles.json"
    assert questions_path == tmp_path / "questions.json"


# --- build_roles_payload / build_questions_payload --------------------------


def test_roles_payload_is_complete_when_produced_equals_requested():
    records = [make_record("pirate"), make_record("sage")]
    payload = grd.build_roles_payload(
        records, failures=[], requested=2, attempted=2, not_attempted=[]
    )
    assert payload == {
        "run_id": grd.compute_run_id(records),
        "complete": True,
        "requested": 2,
        "attempted": 2,
        "produced": 2,
        "not_attempted": [],
        "failed": [],
        "roles": records,
    }


def test_roles_payload_records_failures_and_marks_incomplete():
    records = [make_record("pirate")]
    failures = [("sage", "boom"), ("ghost", "kaboom")]
    payload = grd.build_roles_payload(
        records, failures, requested=3, attempted=3, not_attempted=[]
    )
    assert payload["complete"] is False
    assert payload["requested"] == 3
    assert payload["attempted"] == 3
    assert payload["produced"] == 1
    assert payload["failed"] == [
        {"role": "sage", "reason": "boom"},
        {"role": "ghost", "reason": "kaboom"},
    ]


def test_roles_payload_attempted_can_be_less_than_requested():
    # Bütçe/devre kesici erken durdu: 5 istendi, döngü yalnızca 2'ye ulaştı
    # (1 başarı + 1 tetikleyici hata), kalan 3 hiç denenmedi.
    records = [make_record("pirate")]
    failures = [("sage", "DURDURULDU — koşuyu durduran ... tetikleyicisi: boom")]
    not_attempted = ["ghost", "ancient", "oracle"]
    payload = grd.build_roles_payload(
        records, failures, requested=5, attempted=2, not_attempted=not_attempted
    )
    assert payload["requested"] == 5
    assert payload["attempted"] == 2
    assert payload["attempted"] != payload["requested"]
    assert payload["not_attempted"] == not_attempted
    assert payload["complete"] is False


def test_questions_payload_tracks_completeness():
    records = [make_record("pirate"), make_record("sage")]
    payload = grd.build_questions_payload(records, requested=2, attempted=2)
    assert payload["complete"] is True
    assert payload["requested"] == 2
    assert payload["attempted"] == 2
    assert payload["produced"] == 2
    assert len(payload["shared_questions"]) == 10  # 2 roles * 5 questions each


# --- write_artifacts (the full behavior under test) --------------------------


def test_complete_run_writes_canonical_filenames(tmp_path):
    roles = ("pirate", "sage")
    records = [make_record("pirate"), make_record("sage")]
    exit_code, roles_path, questions_path, roles_payload, questions_payload = (
        grd.write_artifacts(
            tmp_path, roles, records, failures=[], attempted=2, allow_partial=False
        )
    )

    assert exit_code == 0
    assert roles_path == tmp_path / "roles.json"
    assert questions_path == tmp_path / "questions.json"
    assert roles_path.exists()
    assert questions_path.exists()
    assert not (tmp_path / "roles.partial.json").exists()
    assert not (tmp_path / "questions.partial.json").exists()

    on_disk = json.loads(roles_path.read_text(encoding="utf-8"))
    assert on_disk["complete"] is True
    assert on_disk["produced"] == 2
    assert on_disk["requested"] == 2
    assert on_disk["attempted"] == 2
    assert on_disk["not_attempted"] == []
    assert on_disk["failed"] == []
    assert len(on_disk["roles"]) == 2

    q_on_disk = json.loads(questions_path.read_text(encoding="utf-8"))
    assert q_on_disk["complete"] is True
    assert q_on_disk == questions_payload


def test_truncated_run_writes_only_partial_filenames_and_exits_nonzero(tmp_path):
    roles = ("pirate", "sage")
    records = [make_record("pirate")]
    failures = [("sage", "budget exceeded mid-run")]
    exit_code, roles_path, questions_path, roles_payload, _ = grd.write_artifacts(
        tmp_path, roles, records, failures, attempted=2, allow_partial=False
    )

    assert exit_code != 0
    assert roles_path == tmp_path / "roles.partial.json"
    assert questions_path == tmp_path / "questions.partial.json"
    assert roles_path.exists()
    assert questions_path.exists()
    assert not (tmp_path / "roles.json").exists(), "eksik koşu kanonik dosyaya yazmamalı"
    assert not (tmp_path / "questions.json").exists()

    on_disk = json.loads(roles_path.read_text(encoding="utf-8"))
    assert on_disk["complete"] is False
    assert on_disk["produced"] == 1
    assert on_disk["requested"] == 2
    assert on_disk["attempted"] == 2
    assert on_disk["not_attempted"] == []
    assert on_disk["failed"] == [{"role": "sage", "reason": "budget exceeded mid-run"}]


def test_truncated_run_not_attempted_lists_roles_never_reached_in_catalog_order(tmp_path):
    from aax.roles import ROLE_NAMES

    roles = ROLE_NAMES[:5]
    records = [make_record(roles[0]), make_record(roles[1])]
    failures = [
        (roles[2], "DURDURULDU — koşuyu durduran bütçe/devre kesici tetikleyicisi: boom")
    ]
    exit_code, _, _, roles_payload, _ = grd.write_artifacts(
        tmp_path, roles, records, failures, attempted=3, allow_partial=False
    )

    assert exit_code != 0
    assert roles_payload["requested"] == 5
    assert roles_payload["attempted"] == 3
    assert roles_payload["attempted"] != roles_payload["requested"], (
        "attempted döngünün ulaştığı rol sayısını yansıtmalı, batch büyüklüğünü değil"
    )
    assert roles_payload["not_attempted"] == [roles[3], roles[4]]
    assert roles_payload["not_attempted"] == list(roles[3:]), "katalog sırası korunmalı"


def test_truncated_run_leaves_existing_complete_artifacts_untouched(tmp_path):
    # Önceki tam koşudan kalan kanonik dosyalar zaten var.
    existing_roles = tmp_path / "roles.json"
    existing_questions = tmp_path / "questions.json"
    existing_roles.write_text('{"complete": true, "sentinel": "keep-me"}', encoding="utf-8")
    existing_questions.write_text(
        '{"complete": true, "sentinel": "keep-me"}', encoding="utf-8"
    )

    roles = ("pirate", "sage")
    records = [make_record("pirate")]
    failures = [("sage", "boom")]
    exit_code, roles_path, questions_path, _, _ = grd.write_artifacts(
        tmp_path, roles, records, failures, attempted=2, allow_partial=False
    )

    assert exit_code != 0
    assert roles_path.name == "roles.partial.json"
    assert json.loads(existing_roles.read_text(encoding="utf-8")) == {
        "complete": True,
        "sentinel": "keep-me",
    }
    assert json.loads(existing_questions.read_text(encoding="utf-8")) == {
        "complete": True,
        "sentinel": "keep-me",
    }


def test_allow_partial_promotes_truncated_run_to_canonical_filenames(tmp_path):
    roles = ("pirate", "sage")
    records = [make_record("pirate")]
    failures = [("sage", "boom")]
    exit_code, roles_path, questions_path, roles_payload, _ = grd.write_artifacts(
        tmp_path, roles, records, failures, attempted=2, allow_partial=True
    )

    assert exit_code == 0, "--allow-partial verildiyse pipeline durmamalı"
    assert roles_path == tmp_path / "roles.json"
    assert questions_path == tmp_path / "questions.json"
    assert roles_path.exists()
    assert questions_path.exists()
    assert not (tmp_path / "roles.partial.json").exists()

    on_disk = json.loads(roles_path.read_text(encoding="utf-8"))
    assert on_disk["complete"] is False, "promote edilse de zarf hâlâ eksik olduğunu söylemeli"
    assert on_disk["produced"] == 1
    assert on_disk["requested"] == 2
    assert on_disk["attempted"] == 2


def test_allow_partial_promotion_emits_stderr_warning(tmp_path, capsys):
    roles = ("pirate", "sage", "ghost")
    records = [make_record("pirate")]
    failures = [("sage", "boom")]
    grd.write_artifacts(tmp_path, roles, records, failures, attempted=2, allow_partial=True)

    err = capsys.readouterr().err
    assert "UYARI" in err
    assert "1/3" in err  # produced/requested
    assert "roles.json" in err


def test_complete_run_does_not_emit_promotion_warning(tmp_path, capsys):
    roles = ("pirate", "sage")
    records = [make_record("pirate"), make_record("sage")]
    grd.write_artifacts(
        tmp_path, roles, records, failures=[], attempted=2, allow_partial=False
    )

    err = capsys.readouterr().err
    assert "UYARI" not in err


def test_complete_run_with_allow_partial_does_not_emit_promotion_warning(tmp_path, capsys):
    # complete=True ise --allow-partial hiçbir şeyi "terfi ettirmiyor" —
    # zaten kanonik ada yazılıyor, bu yüzden uyarı yanlış olurdu.
    roles = ("pirate", "sage")
    records = [make_record("pirate"), make_record("sage")]
    grd.write_artifacts(
        tmp_path, roles, records, failures=[], attempted=2, allow_partial=True
    )

    err = capsys.readouterr().err
    assert "UYARI" not in err


def test_write_artifacts_creates_data_dir_if_missing(tmp_path):
    data_dir = tmp_path / "nested" / "data"
    assert not data_dir.exists()
    exit_code, roles_path, _, _, _ = grd.write_artifacts(
        data_dir,
        ("pirate",),
        [make_record("pirate")],
        failures=[],
        attempted=1,
        allow_partial=False,
    )
    assert exit_code == 0
    assert roles_path.exists()


# --- run_generation_loop (network-free via StubClient) -----------------------


def test_run_generation_loop_processes_all_roles_when_no_failures():
    roles = ("pirate", "sage")
    client = StubClient([raw_response_for("pirate"), raw_response_for("sage")])
    records, failures, attempted, stop_reason = grd.run_generation_loop(roles, client)

    assert attempted == 2
    assert failures == []
    assert [r["role"] for r in records] == ["pirate", "sage"]


def test_run_generation_loop_attempted_reflects_roles_reached_not_batch_size():
    # 3 istendi ama 2. çağrı bütçeyi patlatıyor — döngü 3.'ye hiç ulaşmıyor.
    roles = ("pirate", "sage", "ghost")
    trigger = BudgetExceeded("'stage0_roles' aşama bütçesi doldu: 130/130")
    client = StubClient([raw_response_for("pirate"), trigger])
    records, failures, attempted, stop_reason = grd.run_generation_loop(roles, client)

    assert attempted == 2
    assert attempted != len(roles), "attempted batch büyüklüğüyle eşit olmamalı"


def test_run_generation_loop_records_triggering_role_in_failed_with_stop_reason():
    roles = ("pirate", "sage", "ghost")
    trigger = BudgetExceeded("'stage0_roles' aşama bütçesi doldu: 130/130")
    client = StubClient([raw_response_for("pirate"), trigger])
    records, failures, attempted, stop_reason = grd.run_generation_loop(roles, client)

    assert attempted == 2
    assert [r["role"] for r in records] == ["pirate"]
    assert len(failures) == 1
    failed_role, reason = failures[0]
    assert failed_role == "sage"
    # Neden, bir ayrıştırma/gateway hatasından ayırt edilebilir şekilde
    # tetikleyici olduğunu açıkça söylemeli.
    assert "tetikleyici" in reason.lower()
    assert "130/130" in reason  # orijinal istisna mesajı da korunmalı


def test_run_generation_loop_stops_on_circuit_open_too():
    roles = ("pirate", "sage")
    trigger = CircuitOpen("devre kesici açık")
    client = StubClient([trigger])
    records, failures, attempted, stop_reason = grd.run_generation_loop(roles, client)

    assert attempted == 1
    assert records == []
    assert len(failures) == 1
    assert failures[0][0] == "pirate"
    assert "tetikleyici" in failures[0][1].lower()


def test_run_generation_loop_records_judge_parse_error_and_continues():
    roles = ("pirate", "sage")
    client = StubClient(["not json at all", raw_response_for("sage")])
    records, failures, attempted, stop_reason = grd.run_generation_loop(roles, client)

    assert attempted == 2
    assert [r["role"] for r in records] == ["sage"]
    assert len(failures) == 1
    assert failures[0][0] == "pirate"


def test_run_generation_loop_records_gateway_error_and_continues():
    roles = ("pirate", "sage")
    client = StubClient([GatewayError("HTTP 500"), raw_response_for("sage")])
    records, failures, attempted, stop_reason = grd.run_generation_loop(roles, client)

    assert attempted == 2
    assert [r["role"] for r in records] == ["sage"]
    assert failures == [("pirate", "HTTP 500")]


def test_run_generation_loop_with_no_roles_returns_zero_attempted():
    client = StubClient([])
    records, failures, attempted, stop_reason = grd.run_generation_loop((), client)

    assert records == []
    assert failures == []
    assert attempted == 0


# --- --limit 0 no longer means "no limit" (finding 4) ------------------------


def test_select_roles_with_limit_zero_returns_empty():
    from aax.roles import ROLE_NAMES

    assert grd.select_roles(0) == ()
    assert grd.select_roles(0) != ROLE_NAMES


def test_select_roles_with_limit_none_returns_all():
    from aax.roles import ROLE_NAMES

    assert grd.select_roles(None) == ROLE_NAMES


def test_select_roles_with_positive_limit_returns_prefix():
    from aax.roles import ROLE_NAMES

    assert grd.select_roles(5) == ROLE_NAMES[:5]
    assert len(grd.select_roles(5)) == 5


# --- argparse wiring ----------------------------------------------------------


def test_argument_parser_accepts_allow_partial_flag():
    parser = grd.build_arg_parser()
    args = parser.parse_args(["--allow-partial"])
    assert args.allow_partial is True
    args_default = parser.parse_args([])
    assert args_default.allow_partial is False


def test_argument_parser_limit_zero_is_distinct_from_no_limit():
    parser = grd.build_arg_parser()
    assert parser.parse_args(["--limit", "0"]).limit == 0
    assert parser.parse_args([]).limit is None


def test_argument_parser_exposes_max_parse_failures():
    parser = grd.build_arg_parser()
    assert parser.parse_args([]).max_parse_failures == 10
    assert parser.parse_args(["--max-parse-failures", "3"]).max_parse_failures == 3


# --- I1: döngüden sızan hiçbir istisna üretilmiş işi çöpe atmamalı ------------


def test_loop_survives_budget_corrupted_and_keeps_records():
    """`BudgetCorrupted` `RuntimeError`'dan türer ama yakalanan dört sınıfın
    hiçbirinden türemez — eski döngü onu hiç yakalamıyordu."""
    roles = ("pirate", "sage", "ghost")
    client = StubClient(
        [
            raw_response_for("pirate"),
            raw_response_for("sage"),
            BudgetCorrupted("bütçe dosyası ayrıştırılamadı"),
        ]
    )
    records, failures, attempted, stop_reason = grd.run_generation_loop(roles, client)

    assert [r["role"] for r in records] == ["pirate", "sage"], (
        "tamamlanan kayıtlar sızan istisnayla birlikte çöpe gitmemeli"
    )
    assert attempted == 3
    assert stop_reason is not None
    assert "BudgetCorrupted" in stop_reason
    assert failures == [(roles[2], stop_reason)]


def test_loop_survives_unknown_stage_value_error():
    """Bilinmeyen aşama `ValueError`'ı da hiçbir istisna sınıfından türemiyordu."""
    roles = ("pirate", "sage")
    client = StubClient([raw_response_for("pirate"), ValueError("Bilinmeyen aşama adı")])
    records, failures, attempted, stop_reason = grd.run_generation_loop(roles, client)

    assert [r["role"] for r in records] == ["pirate"]
    assert stop_reason is not None
    assert "ValueError" in stop_reason


def test_loop_survives_keyboard_interrupt():
    """120 rollük bir koşuda Ctrl-C tüm işi çöpe atıyordu."""
    roles = ("pirate", "sage", "ghost")
    client = StubClient([raw_response_for("pirate"), KeyboardInterrupt()])
    records, failures, attempted, stop_reason = grd.run_generation_loop(roles, client)

    assert [r["role"] for r in records] == ["pirate"]
    assert attempted == 2
    assert stop_reason is not None
    assert "KeyboardInterrupt" in stop_reason
    assert len(failures) == 1


def test_main_writes_partial_artifact_when_unexpected_error_escapes(
    tmp_path, monkeypatch, capsys
):
    """Uçtan uca: sızan hata sonrası `main()` yine de artifact yazmalı.

    Bu, bulgunun ampirik doğrulamasının testleşmiş hâli: iki kez başaran,
    sonra `BudgetCorrupted` fırlatan bir istemci eskiden İKİ tamamlanmış
    kaydı da çöpe atıyor ve HİÇ artifact yazmıyordu.
    """
    client = StubClient(
        [
            raw_response_for(role)
            for role in ("bohemian", "engineer")
        ]
        + [BudgetCorrupted("bütçe dosyası ayrıştırılamadı")]
    )
    monkeypatch.setattr(grd, "build_default_client", lambda: client)
    monkeypatch.setattr(grd.config, "DATA_DIR", tmp_path)

    exit_code = main_with(["--limit", "3"])

    assert exit_code != 0
    partial = tmp_path / "roles.partial.json"
    assert partial.exists(), "tamamlanan kayıtlar diske yazılmalıydı"
    on_disk = json.loads(partial.read_text(encoding="utf-8"))
    assert on_disk["produced"] == 2
    assert on_disk["complete"] is False
    assert "BudgetCorrupted" in on_disk["failed"][-1]["reason"]
    assert (tmp_path / "questions.partial.json").exists()

    err = capsys.readouterr().err
    assert "BudgetCorrupted" in err


def test_main_aborted_run_exits_nonzero_even_with_allow_partial(
    tmp_path, monkeypatch, capsys
):
    """Yarıda kesilmiş koşu asla "başarı" değildir — Ctrl-C bile olsa."""
    client = StubClient([raw_response_for("bohemian"), KeyboardInterrupt()])
    monkeypatch.setattr(grd, "build_default_client", lambda: client)
    monkeypatch.setattr(grd.config, "DATA_DIR", tmp_path)

    exit_code = main_with(["--limit", "3", "--allow-partial"])

    assert exit_code != 0
    # `--allow-partial` dosya adlarını yine de terfi ettirir (kapı korunuyor).
    assert (tmp_path / "roles.json").exists()
    on_disk = json.loads((tmp_path / "roles.json").read_text(encoding="utf-8"))
    assert on_disk["produced"] == 1
    assert on_disk["complete"] is False


# --- I2: ardışık ayrıştırma hatalarında kapalı yönde dur ---------------------


def test_loop_stops_after_consecutive_parse_failures():
    """İçerik hatası devre kesiciyi SIFIRLAR — kapı script düzeyinde olmalı."""
    roles = grd.ROLE_NAMES[:20]
    client = StubClient(["bu json degil"])
    records, failures, attempted, stop_reason = grd.run_generation_loop(
        roles, client, max_consecutive_parse_failures=10
    )

    assert records == []
    assert attempted == 10, "10. ayrıştırma hatasında durmalı, 20'ye kadar gitmemeli"
    assert client.calls == 10, "aşama bütçesinin tamamı sıfır kayıt için yakılmamalı"
    assert stop_reason is not None
    assert "ayrıştırma hatası" in stop_reason
    assert len(failures) == 10, "durdurma anında ikinci bir hata satırı eklenmemeli"


def test_parse_failure_counter_resets_on_success():
    """Sayaç ARDIŞIK hataları sayar — araya giren başarı sıfırlamalı."""
    roles = grd.ROLE_NAMES[:6]
    client = StubClient(
        [
            "bozuk",
            "bozuk",
            raw_response_for(roles[2]),
            "bozuk",
            "bozuk",
            raw_response_for(roles[5]),
        ]
    )
    records, failures, attempted, stop_reason = grd.run_generation_loop(
        roles, client, max_consecutive_parse_failures=3
    )

    assert stop_reason is None, "hiçbir noktada 3 ardışık hata olmadı"
    assert attempted == 6
    assert len(records) == 2
    assert len(failures) == 4


def test_parse_failure_threshold_is_configurable():
    roles = grd.ROLE_NAMES[:10]
    client = StubClient(["bozuk"])
    _, failures, attempted, stop_reason = grd.run_generation_loop(
        roles, client, max_consecutive_parse_failures=3
    )
    assert attempted == 3
    assert len(failures) == 3
    assert stop_reason is not None


def test_gateway_errors_do_not_trip_the_parse_guard():
    """Taşıma hatası ayrıştırma sayacını artırmamalı — onu devre kesici görür."""
    roles = grd.ROLE_NAMES[:5]
    client = StubClient([GatewayError("HTTP 500")])
    records, failures, attempted, stop_reason = grd.run_generation_loop(
        roles, client, max_consecutive_parse_failures=2
    )

    assert attempted == 5, "GatewayError ayrıştırma kapısını tetiklememeli"
    assert stop_reason is None
    assert len(failures) == 5


def test_main_passes_parse_failure_flag_through(tmp_path, monkeypatch):
    client = StubClient(["bu json degil"])
    monkeypatch.setattr(grd, "build_default_client", lambda: client)
    monkeypatch.setattr(grd.config, "DATA_DIR", tmp_path)

    exit_code = main_with(["--limit", "20", "--max-parse-failures", "4"])

    assert exit_code != 0
    assert client.calls == 4


# --- şema değişmezi -----------------------------------------------------------


def test_payload_counter_invariant_holds_with_all_three_components_nonzero(tmp_path):
    """`requested == produced + len(failed) + len(not_attempted)`.

    Üç bileşenin de AYNI ANDA sıfırdan farklı olduğu bir koşu kurulur:
    2 başarı, 2 hata (biri tetikleyici), 3 hiç denenmemiş rol.
    """
    roles = grd.ROLE_NAMES[:7]
    client = StubClient(
        [
            raw_response_for(roles[0]),
            "bu json degil",
            raw_response_for(roles[2]),
            BudgetExceeded("'stage0_roles' aşama bütçesi doldu: 145/145"),
        ]
    )
    records, failures, attempted, stop_reason = grd.run_generation_loop(roles, client)

    _, _, _, roles_payload, _ = grd.write_artifacts(
        tmp_path, roles, records, failures, attempted, allow_partial=True
    )

    assert roles_payload["produced"] == 2
    assert len(roles_payload["failed"]) == 2
    assert len(roles_payload["not_attempted"]) == 3
    assert roles_payload["requested"] == (
        roles_payload["produced"]
        + len(roles_payload["failed"])
        + len(roles_payload["not_attempted"])
    )
    assert roles_payload["attempted"] == 4


def test_counter_invariant_holds_on_disk(tmp_path):
    """Aynı değişmez, diske yazılan zarfta da geçerli olmalı."""
    roles = grd.ROLE_NAMES[:7]
    client = StubClient(
        [
            raw_response_for(roles[0]),
            "bu json degil",
            raw_response_for(roles[2]),
            CircuitOpen("devre kesici açık"),
        ]
    )
    records, failures, attempted, _ = grd.run_generation_loop(roles, client)
    _, roles_path, _, _, _ = grd.write_artifacts(
        tmp_path, roles, records, failures, attempted, allow_partial=False
    )

    on_disk = json.loads(roles_path.read_text(encoding="utf-8"))
    assert on_disk["produced"] > 0
    assert len(on_disk["failed"]) > 0
    assert len(on_disk["not_attempted"]) > 0
    assert on_disk["requested"] == (
        on_disk["produced"] + len(on_disk["failed"]) + len(on_disk["not_attempted"])
    )


# --- I10: soru kümesi takasını görünür kıl -----------------------------------


def test_run_id_is_deterministic_from_role_names_in_catalog_order():
    records = [make_record("pirate"), make_record("sage")]
    assert grd.compute_run_id(records) == grd.compute_run_id(
        [make_record("pirate"), make_record("sage")]
    )
    assert len(grd.compute_run_id(records)) == 16


def test_run_id_changes_when_the_produced_role_set_changes():
    kismi = [make_record("pirate"), make_record("sage")]
    tam = [make_record("pirate"), make_record("sage"), make_record("ghost")]
    assert grd.compute_run_id(kismi) != grd.compute_run_id(tam)


def test_run_id_is_not_clock_based():
    """Kimlik içerikten türetilir — aynı içerik her zaman aynı kimlik."""
    records = [make_record("pirate")]
    ilk = grd.compute_run_id(records)
    time.sleep(0.01)
    assert grd.compute_run_id(records) == ilk


def test_roles_and_questions_share_the_same_run_id(tmp_path):
    roles = ("pirate", "sage")
    records = [make_record("pirate"), make_record("sage")]
    _, roles_path, questions_path, _, _ = grd.write_artifacts(
        tmp_path, roles, records, failures=[], attempted=2, allow_partial=False
    )
    r = json.loads(roles_path.read_text(encoding="utf-8"))
    q = json.loads(questions_path.read_text(encoding="utf-8"))
    assert r["run_id"] == q["run_id"]


def test_partial_and_complete_runs_are_distinguishable_by_run_id(tmp_path):
    """Bulgu senaryosu: kısmi koşu terfi ediyor, sonra tam koşu üzerine yazıyor.

    İki koşunun `shared_questions`'ı FARKLI. Eskiden bunu diskten anlamanın
    yolu yoktu; artık `run_id` ve örnekleme girdileri takası görünür kılıyor.
    """
    roles = ("pirate", "sage", "ghost")

    kismi = [make_record("pirate"), make_record("sage")]
    grd.write_artifacts(tmp_path, roles, kismi, [("ghost", "boom")], 3, allow_partial=True)
    kismi_q = json.loads((tmp_path / "questions.json").read_text(encoding="utf-8"))

    tam = [make_record("pirate"), make_record("sage"), make_record("ghost")]
    grd.write_artifacts(tmp_path, roles, tam, [], 3, allow_partial=False)
    tam_q = json.loads((tmp_path / "questions.json").read_text(encoding="utf-8"))

    assert kismi_q["shared_questions"] != tam_q["shared_questions"], (
        "kurulum geçersiz: iki koşu zaten aynı soruları seçmiş"
    )
    assert kismi_q["run_id"] != tam_q["run_id"], "takas diskten görülebilmeli"
    assert (kismi_q["role_count"], kismi_q["pool_size"]) == (2, 10)
    assert (tam_q["role_count"], tam_q["pool_size"]) == (3, 15)
    assert kismi_q["seed"] == tam_q["seed"] == grd.SEED, (
        "tohum sabit — determinizmi bozan tohum değil, rol kümesi"
    )


def test_allow_partial_warning_mentions_run_id(tmp_path, capsys):
    roles = ("pirate", "sage")
    records = [make_record("pirate")]
    grd.write_artifacts(tmp_path, roles, records, [("sage", "boom")], 2, allow_partial=True)
    err = capsys.readouterr().err
    assert "run_id" in err


# --- artifact atomikliği ------------------------------------------------------


def test_artifacts_are_written_atomically_without_leftovers(tmp_path):
    roles = ("pirate", "sage")
    records = [make_record("pirate"), make_record("sage")]
    grd.write_artifacts(tmp_path, roles, records, [], 2, allow_partial=False)

    artiklar = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert artiklar == [], f"geçici dosya artığı kaldı: {artiklar}"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["questions.json", "roles.json"]


def test_neither_artifact_is_published_when_serialization_fails(tmp_path, monkeypatch):
    """İkinci dosya diske yazılamıyorsa BİRİNCİSİ de yayımlanmamalı.

    Eski kod iki ayrı `write_text` çağrısıydı: ilki başarılı, ikincisi
    başarısız olduğunda `roles.json` yeni, `questions.json` eski kalıyordu.
    """
    (tmp_path / "roles.json").write_text('{"nobet": "eski"}', encoding="utf-8")
    (tmp_path / "questions.json").write_text('{"nobet": "eski"}', encoding="utf-8")

    gercek = grd._stage_temp_file
    cagrilar = {"n": 0}

    def patlayan(path, text):
        cagrilar["n"] += 1
        if cagrilar["n"] == 2:  # questions dosyası
            raise OSError("disk doldu")
        return gercek(path, text)

    monkeypatch.setattr(grd, "_stage_temp_file", patlayan)

    roles = ("pirate", "sage")
    records = [make_record("pirate"), make_record("sage")]
    with pytest.raises(OSError):
        grd.write_artifacts(tmp_path, roles, records, [], 2, allow_partial=False)

    assert json.loads((tmp_path / "roles.json").read_text(encoding="utf-8")) == {
        "nobet": "eski"
    }, "ikinci dosya yazılamadıysa birincisi de yerine konmamalı"
    artiklar = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert artiklar == [], f"geçici dosya artığı kaldı: {artiklar}"


# --- I3 + I4: main() ve --dry-run ön kontrolü --------------------------------


def test_main_dry_run_sends_nothing_and_reports_plan(tmp_path, monkeypatch, capsys):
    client, calls = make_gateway_client(tmp_path, raw_response_for("bohemian"))
    monkeypatch.setattr(grd, "build_default_client", lambda: client)
    monkeypatch.setattr(grd.config, "DATA_DIR", tmp_path)

    exit_code = main_with(["--dry-run", "--limit", "5"])

    assert exit_code == 0
    assert calls == [], "--dry-run tek bir istek bile atmamalı"
    assert client.sends_made == 0
    out = capsys.readouterr().out
    assert "Planlanan çağrı:      5" in out
    assert "kalan: 145" in out
    assert "kalan: 1500" in out


def test_main_dry_run_fails_when_stage_budget_already_spent(
    tmp_path, monkeypatch, capsys
):
    """Bulgu senaryosu: 145'in 130'unu harcamış operatör temiz bir 0 görüyordu.

    Eski kod planı `config.STAGE_BUDGETS[STAGE]` TAVANIYLA kıyaslıyor,
    `gateway_budget.json`'ı hiç okumuyordu.
    """
    client, _ = make_gateway_client(tmp_path, raw_response_for("bohemian"))
    (tmp_path / "budget.json").write_text('{"stage0_roles": 130}', encoding="utf-8")
    monkeypatch.setattr(grd, "build_default_client", lambda: client)
    monkeypatch.setattr(grd.config, "DATA_DIR", tmp_path)

    exit_code = main_with(["--dry-run", "--limit", "100"])

    assert exit_code != 0
    captured = capsys.readouterr()
    assert "kalan: 15" in captured.out
    assert "aşama bütçesi" in captured.err
    assert "100 planlandı" in captured.err


def test_main_dry_run_fails_when_global_cap_nearly_spent(tmp_path, monkeypatch, capsys):
    """Aşama bütçesi bol ama global tavan dolmak üzere."""
    client, _ = make_gateway_client(tmp_path, raw_response_for("bohemian"))
    (tmp_path / "budget.json").write_text('{"baska": 1495}', encoding="utf-8")
    monkeypatch.setattr(grd, "build_default_client", lambda: client)
    monkeypatch.setattr(grd.config, "DATA_DIR", tmp_path)

    exit_code = main_with(["--dry-run", "--limit", "50"])

    assert exit_code != 0
    err = capsys.readouterr().err
    assert "global tavan" in err
    assert "yalnızca 5 kaldı" in err


def test_main_dry_run_passes_when_both_budgets_suffice(tmp_path, monkeypatch, capsys):
    client, _ = make_gateway_client(tmp_path, raw_response_for("bohemian"))
    (tmp_path / "budget.json").write_text('{"stage0_roles": 100}', encoding="utf-8")
    monkeypatch.setattr(grd, "build_default_client", lambda: client)
    monkeypatch.setattr(grd.config, "DATA_DIR", tmp_path)

    exit_code = main_with(["--dry-run", "--limit", "40"])

    assert exit_code == 0
    assert "HATA" not in capsys.readouterr().err


def test_dry_run_and_chat_share_the_same_cache_key(tmp_path, monkeypatch, capsys):
    """`would_call` ile `chat` AYNI cache anahtarını üretmeli.

    `temperature`/`max_tokens` iki yolda ayrışırsa `--dry-run` cache'te hazır
    duran kayıtları göremez ve her şeyi "planlanan" sayar. Kod bunu kırılgan
    diye işaretlemişti ama hiçbir şey birbirine bağlamıyordu.
    """
    client, calls = make_gateway_client(tmp_path, raw_response_for("bohemian"))
    monkeypatch.setattr(grd, "build_default_client", lambda: client)
    monkeypatch.setattr(grd.config, "DATA_DIR", tmp_path)

    # Önce gerçek koşu: 3 rol cache'e girer.
    assert main_with(["--limit", "3"]) == 0
    assert len(calls) == 3
    capsys.readouterr()

    # Sonra dry-run: hepsi cache'te olmalı, planlanan 0.
    assert main_with(["--dry-run", "--limit", "3"]) == 0
    out = capsys.readouterr().out
    assert "Planlanan çağrı:      0 (cache'te: 3)" in out, out
    assert len(calls) == 3, "dry-run yeni istek atmamalı"


def test_main_complete_run_writes_canonical_files_and_exits_zero(
    tmp_path, monkeypatch, capsys
):
    client, calls = make_gateway_client(tmp_path, raw_response_for("bohemian"))
    monkeypatch.setattr(grd, "build_default_client", lambda: client)
    monkeypatch.setattr(grd.config, "DATA_DIR", tmp_path)

    exit_code = main_with(["--limit", "3"])

    assert exit_code == 0
    assert len(calls) == 3
    on_disk = json.loads((tmp_path / "roles.json").read_text(encoding="utf-8"))
    assert on_disk["complete"] is True
    assert on_disk["produced"] == 3
    assert not (tmp_path / "roles.partial.json").exists()

    out = capsys.readouterr().out
    assert "complete=True" in out
    assert "run_id=" in out
    assert "Gönderilen istek: 3" in out


def test_main_second_run_is_free_from_cache(tmp_path, monkeypatch, capsys):
    client, calls = make_gateway_client(tmp_path, raw_response_for("bohemian"))
    monkeypatch.setattr(grd, "build_default_client", lambda: client)
    monkeypatch.setattr(grd.config, "DATA_DIR", tmp_path)

    assert main_with(["--limit", "3"]) == 0
    assert main_with(["--limit", "3"]) == 0
    assert len(calls) == 3, "ikinci koşu tamamen cache'ten dönmeli"
    assert "Gönderilen istek: 3" in capsys.readouterr().out


def test_main_incomplete_run_writes_partial_and_exits_nonzero(
    tmp_path, monkeypatch, capsys
):
    client = StubClient([raw_response_for("bohemian"), "bu json degil"])
    monkeypatch.setattr(grd, "build_default_client", lambda: client)
    monkeypatch.setattr(grd.config, "DATA_DIR", tmp_path)

    exit_code = main_with(["--limit", "2"])

    assert exit_code == 1
    assert (tmp_path / "roles.partial.json").exists()
    assert not (tmp_path / "roles.json").exists(), "kapı korunmalı"
    assert "koşu eksik kaldı" in capsys.readouterr().err


def test_main_allow_partial_promotes_and_exits_zero_when_not_aborted(
    tmp_path, monkeypatch, capsys
):
    """Kapı davranışı değişmedi: kesinti YOKSA --allow-partial 0 döndürür."""
    client = StubClient([raw_response_for("bohemian"), "bu json degil"])
    monkeypatch.setattr(grd, "build_default_client", lambda: client)
    monkeypatch.setattr(grd.config, "DATA_DIR", tmp_path)

    exit_code = main_with(["--limit", "2", "--allow-partial"])

    assert exit_code == 0
    assert (tmp_path / "roles.json").exists()
    assert "UYARI" in capsys.readouterr().err


def test_main_rejects_unknown_stage_in_dry_run(tmp_path, monkeypatch):
    """Yazım hatası yapılmış aşama adı `--dry-run`'da temiz 0 dönmemeli."""
    client, _ = make_gateway_client(tmp_path, raw_response_for("bohemian"))
    monkeypatch.setattr(grd, "build_default_client", lambda: client)
    monkeypatch.setattr(grd, "STAGE", "stage0_rolez")

    with pytest.raises(ValueError, match="Bilinmeyen aşama"):
        main_with(["--dry-run", "--limit", "2"])


# --- I5: tanı sarmalayıcısı (iyi bicimde hata mesajları) ----------------------


def test_main_missing_api_key_produces_diagnostic(monkeypatch, capsys):
    """APP_KEY_JAILBREAK yoksa traceback yerine tanı çıkmalı."""

    def raise_missing_key():
        raise RuntimeError(
            "APP_KEY_JAILBREAK ortam değişkeni tanımlı değil. "
            "Dağıtım ortamınızın .env dosyasından alıp kabuğunuzda export edin."
        )

    monkeypatch.setattr(grd, "build_default_client", raise_missing_key)

    exit_code = main_with([])

    assert exit_code == 2, "eksik anahtar EXIT_KOSULAMADI (2) dönmeli"
    err = capsys.readouterr().err
    assert "BAŞARISIZ" in err, "hata iletisi Türkçe tanı içermeli"
    assert "gateway istemcisi kurulamadı" in err
    assert "APP_KEY_JAILBREAK" in err
    assert "RuntimeError" not in err, "traceback içermemeli"


def test_main_budget_corrupted_during_dry_run_produces_diagnostic(
    tmp_path, monkeypatch, capsys
):
    """BudgetCorrupted --dry-run sırasında tanısal mesaj ile EXIT_KOSULAMADI dönmeli."""

    class MockClientBudgetCorrupted:
        def remaining_budget(self, stage):
            raise BudgetCorrupted("bütçe dosyası ayrıştırılamadı")

        def would_call(self, *args, **kwargs):
            return True

    monkeypatch.setattr(grd, "build_default_client", lambda: MockClientBudgetCorrupted())

    exit_code = main_with(["--dry-run", "--limit", "1"])

    assert exit_code == 2, "gateway hatası EXIT_KOSULAMADI (2) dönmeli"
    err = capsys.readouterr().err
    assert "BAŞARISIZ" in err
    assert "bütçe dosyası" in err
    assert "BudgetCorrupted" not in err, "sınıf adını göstermemeli"


def test_main_budget_exceeded_during_dry_run_produces_diagnostic(
    tmp_path, monkeypatch, capsys
):
    """BudgetExceeded --dry-run sırasında tanısal mesaj ile EXIT_KOSULAMADI dönmeli."""

    class MockClientBudgetExceeded:
        def remaining_budget(self, stage):
            raise BudgetExceeded("'stage0_roles' aşama bütçesi doldu: 145/145")

        def would_call(self, *args, **kwargs):
            return True

    monkeypatch.setattr(grd, "build_default_client", lambda: MockClientBudgetExceeded())

    exit_code = main_with(["--dry-run", "--limit", "1"])

    assert exit_code == 2
    err = capsys.readouterr().err
    assert "BAŞARISIZ" in err
    assert "bütçe" in err
    assert "dolu" in err
    assert "BudgetExceeded" not in err, "sınıf adını göstermemeli"


def test_main_circuit_open_during_dry_run_produces_diagnostic(
    tmp_path, monkeypatch, capsys
):
    """CircuitOpen --dry-run sırasında tanısal mesaj ile EXIT_KOSULAMADI dönmeli."""

    class MockClientCircuitOpen:
        def remaining_budget(self, stage):
            raise CircuitOpen("devre kesici açık")

        def would_call(self, *args, **kwargs):
            return True

    monkeypatch.setattr(grd, "build_default_client", lambda: MockClientCircuitOpen())

    exit_code = main_with(["--dry-run", "--limit", "1"])

    assert exit_code == 2
    err = capsys.readouterr().err
    assert "BAŞARISIZ" in err
    assert "devre kesici" in err
    assert "CircuitOpen" not in err, "sınıf adını göstermemeli"


def test_main_gateway_error_during_dry_run_produces_diagnostic(
    tmp_path, monkeypatch, capsys
):
    """GatewayError --dry-run sırasında tanısal mesaj ile EXIT_KOSULAMADI dönmeli."""

    class MockClientGatewayError:
        def remaining_budget(self, stage):
            raise GatewayError("HTTP 500 Sunucu Hatası")

        def would_call(self, *args, **kwargs):
            return True

    monkeypatch.setattr(grd, "build_default_client", lambda: MockClientGatewayError())

    exit_code = main_with(["--dry-run", "--limit", "1"])

    assert exit_code == 2
    err = capsys.readouterr().err
    assert "BAŞARISIZ" in err
    assert "gateway çağrısı" in err
    assert "GatewayError" not in err, "sınıf adını göstermemeli"
