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
from pathlib import Path

import pytest

from aax.gateway import BudgetExceeded, CircuitOpen, GatewayError

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
    döner. Gerçek `GatewayClient`'a hiç dokunulmaz.
    """

    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self.calls = 0

    def chat(self, messages, *, stage, max_tokens):
        self.calls += 1
        outcome = self._responses[self.calls - 1]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


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
    records, failures, attempted = grd.run_generation_loop(roles, client)

    assert attempted == 2
    assert failures == []
    assert [r["role"] for r in records] == ["pirate", "sage"]


def test_run_generation_loop_attempted_reflects_roles_reached_not_batch_size():
    # 3 istendi ama 2. çağrı bütçeyi patlatıyor — döngü 3.'ye hiç ulaşmıyor.
    roles = ("pirate", "sage", "ghost")
    trigger = BudgetExceeded("'stage0_roles' aşama bütçesi doldu: 130/130")
    client = StubClient([raw_response_for("pirate"), trigger])
    records, failures, attempted = grd.run_generation_loop(roles, client)

    assert attempted == 2
    assert attempted != len(roles), "attempted batch büyüklüğüyle eşit olmamalı"


def test_run_generation_loop_records_triggering_role_in_failed_with_stop_reason():
    roles = ("pirate", "sage", "ghost")
    trigger = BudgetExceeded("'stage0_roles' aşama bütçesi doldu: 130/130")
    client = StubClient([raw_response_for("pirate"), trigger])
    records, failures, attempted = grd.run_generation_loop(roles, client)

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
    records, failures, attempted = grd.run_generation_loop(roles, client)

    assert attempted == 1
    assert records == []
    assert len(failures) == 1
    assert failures[0][0] == "pirate"
    assert "tetikleyici" in failures[0][1].lower()


def test_run_generation_loop_records_judge_parse_error_and_continues():
    roles = ("pirate", "sage")
    client = StubClient(["not json at all", raw_response_for("sage")])
    records, failures, attempted = grd.run_generation_loop(roles, client)

    assert attempted == 2
    assert [r["role"] for r in records] == ["sage"]
    assert len(failures) == 1
    assert failures[0][0] == "pirate"


def test_run_generation_loop_records_gateway_error_and_continues():
    roles = ("pirate", "sage")
    client = StubClient([GatewayError("HTTP 500"), raw_response_for("sage")])
    records, failures, attempted = grd.run_generation_loop(roles, client)

    assert attempted == 2
    assert [r["role"] for r in records] == ["sage"]
    assert failures == [("pirate", "HTTP 500")]


def test_run_generation_loop_with_no_roles_returns_zero_attempted():
    client = StubClient([])
    records, failures, attempted = grd.run_generation_loop((), client)

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
