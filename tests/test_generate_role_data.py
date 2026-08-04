"""Aşama 0 script'inin dosya adlandırma / tamlık mantığı testleri.

Ağa çıkmaz: `write_artifacts()` ve yardımcılarını doğrudan `tmp_path`'e
karşı çağırır, `GatewayClient`'a hiç dokunmaz. Script dosya adı bir rakamla
başladığı için (`00_generate_role_data.py`) normal `import` ile
içe aktarılamaz; `importlib` ile dosya yolundan yüklenir.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

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


def test_roles_payload_is_complete_when_produced_equals_attempted():
    records = [make_record("pirate"), make_record("sage")]
    payload = grd.build_roles_payload(records, failures=[], attempted=2)
    assert payload == {
        "complete": True,
        "attempted": 2,
        "produced": 2,
        "failed": [],
        "roles": records,
    }


def test_roles_payload_records_failures_and_marks_incomplete():
    records = [make_record("pirate")]
    failures = [("sage", "boom"), ("ghost", "kaboom")]
    payload = grd.build_roles_payload(records, failures, attempted=3)
    assert payload["complete"] is False
    assert payload["attempted"] == 3
    assert payload["produced"] == 1
    assert payload["failed"] == [
        {"role": "sage", "reason": "boom"},
        {"role": "ghost", "reason": "kaboom"},
    ]


def test_questions_payload_tracks_completeness():
    records = [make_record("pirate"), make_record("sage")]
    payload = grd.build_questions_payload(records, attempted=2)
    assert payload["complete"] is True
    assert payload["attempted"] == 2
    assert payload["produced"] == 2
    assert len(payload["shared_questions"]) == 10  # 2 roles * 5 questions each


# --- write_artifacts (the full behavior under test) --------------------------


def test_complete_run_writes_canonical_filenames(tmp_path):
    records = [make_record("pirate"), make_record("sage")]
    exit_code, roles_path, questions_path, roles_payload, questions_payload = (
        grd.write_artifacts(tmp_path, records, failures=[], attempted=2, allow_partial=False)
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
    assert on_disk["attempted"] == 2
    assert on_disk["failed"] == []
    assert len(on_disk["roles"]) == 2

    q_on_disk = json.loads(questions_path.read_text(encoding="utf-8"))
    assert q_on_disk["complete"] is True
    assert q_on_disk == questions_payload


def test_truncated_run_writes_only_partial_filenames_and_exits_nonzero(tmp_path):
    records = [make_record("pirate")]
    failures = [("sage", "budget exceeded mid-run")]
    exit_code, roles_path, questions_path, roles_payload, _ = grd.write_artifacts(
        tmp_path, records, failures, attempted=2, allow_partial=False
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
    assert on_disk["attempted"] == 2
    assert on_disk["failed"] == [{"role": "sage", "reason": "budget exceeded mid-run"}]


def test_truncated_run_leaves_existing_complete_artifacts_untouched(tmp_path):
    # Önceki tam koşudan kalan kanonik dosyalar zaten var.
    existing_roles = tmp_path / "roles.json"
    existing_questions = tmp_path / "questions.json"
    existing_roles.write_text('{"complete": true, "sentinel": "keep-me"}', encoding="utf-8")
    existing_questions.write_text(
        '{"complete": true, "sentinel": "keep-me"}', encoding="utf-8"
    )

    records = [make_record("pirate")]
    failures = [("sage", "boom")]
    exit_code, roles_path, questions_path, _, _ = grd.write_artifacts(
        tmp_path, records, failures, attempted=2, allow_partial=False
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
    records = [make_record("pirate")]
    failures = [("sage", "boom")]
    exit_code, roles_path, questions_path, roles_payload, _ = grd.write_artifacts(
        tmp_path, records, failures, attempted=2, allow_partial=True
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
    assert on_disk["attempted"] == 2


def test_write_artifacts_creates_data_dir_if_missing(tmp_path):
    data_dir = tmp_path / "nested" / "data"
    assert not data_dir.exists()
    exit_code, roles_path, _, _, _ = grd.write_artifacts(
        data_dir, [make_record("pirate")], failures=[], attempted=1, allow_partial=False
    )
    assert exit_code == 0
    assert roles_path.exists()


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
