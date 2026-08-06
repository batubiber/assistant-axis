import json

import pytest

from aax.prompts import (
    DEFAULT_SYSTEM_PROMPTS,
    build_default_specs,
    build_role_specs,
    load_role_catalog,
    to_chat_messages,
)


def make_catalog(n_roles=3, n_instructions=3, n_questions=40):
    return {
        "complete": True,
        "limit": None,
        "requested": n_roles,
        "produced": n_roles,
        "catalog_size": n_roles,
        "failed": [],
        "run_id": "test",
        "roles": [
            {
                "role": f"rol{i}",
                "description": f"aciklama {i}",
                "instructions": [f"You are rol{i}, variant {j}." for j in range(n_instructions)],
                "questions": [f"soru {i}-{j}" for j in range(n_questions)],
            }
            for i in range(n_roles)
        ],
    }


def write_catalog(tmp_path, payload):
    path = tmp_path / "roles.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_role_catalog_accepts_canonical_artifact(tmp_path):
    path = write_catalog(tmp_path, make_catalog())
    roles = load_role_catalog(path)
    assert len(roles) == 3


def test_load_role_catalog_rejects_incomplete_artifact(tmp_path):
    payload = make_catalog()
    payload["complete"] = False
    path = write_catalog(tmp_path, payload)
    with pytest.raises(ValueError, match="complete"):
        load_role_catalog(path)


def test_load_role_catalog_rejects_pilot_artifact(tmp_path):
    payload = make_catalog()
    payload["limit"] = 3
    path = write_catalog(tmp_path, payload)
    with pytest.raises(ValueError, match="pilot"):
        load_role_catalog(path)


def test_load_role_catalog_rejects_partial_catalog(tmp_path):
    payload = make_catalog(n_roles=3)
    payload["catalog_size"] = 120
    path = write_catalog(tmp_path, payload)
    with pytest.raises(ValueError, match="katalog"):
        load_role_catalog(path)


def test_load_role_catalog_rejects_a_missing_roles_key_as_valueerror(tmp_path):
    """D1: bu satır `payload["roles"]` ile doğrudan indeksliyordu ve bozuk bir
    dosyada ÇIPLAK bir `KeyError` fırlatıyordu. `KeyError` bir `ValueError`
    DEĞİLDİR: `06_label_and_train_probe.py`'nin `except ValueError`
    sarmalayıcısını atlayıp yorumlayıcıyı çıkış 1 ile döndürüyordu — o
    script'te 1 "probe güvenilmez" demek. Yani BOZUK BİR KATALOG, PROBE
    HAKKINDA BİR BULGU olarak raporlanıyordu."""
    payload = make_catalog()
    del payload["roles"]
    path = write_catalog(tmp_path, payload)
    with pytest.raises(ValueError, match="roles"):
        load_role_catalog(path)


def test_load_role_catalog_rejects_a_non_list_roles_value(tmp_path):
    payload = make_catalog()
    payload["roles"] = {"rol0": "bozuk"}
    path = write_catalog(tmp_path, payload)
    with pytest.raises(ValueError, match="roles"):
        load_role_catalog(path)


def test_role_specs_are_roles_times_prompts_times_questions():
    catalog = make_catalog(n_roles=5, n_instructions=3)["roles"]
    questions = [f"ortak {i}" for i in range(40)]
    specs = build_role_specs(catalog, questions)
    assert len(specs) == 5 * 3 * 40
    assert all(s.kind == "role" for s in specs)


def test_role_specs_use_shared_questions_not_per_role_questions():
    """Makale tüm roller için AYNI soru setini kullanır (Bölüm 2.1.1)."""
    catalog = make_catalog(n_roles=2)["roles"]
    questions = ["ortak-A", "ortak-B"]
    specs = build_role_specs(catalog, questions)
    assert {s.question for s in specs} == {"ortak-A", "ortak-B"}


def test_default_specs_count():
    questions = [f"ortak {i}" for i in range(40)]
    specs = build_default_specs(questions, samples_per_prompt=10)
    assert len(specs) == len(DEFAULT_SYSTEM_PROMPTS) * 40 * 10
    assert all(s.kind == "default" for s in specs)
    assert all(s.role is None for s in specs)


def test_default_prompts_include_a_bare_no_system_variant():
    assert None in DEFAULT_SYSTEM_PROMPTS


def test_chat_messages_omit_system_when_prompt_is_none():
    specs = build_default_specs(["tek soru"], samples_per_prompt=1)
    bare = [s for s in specs if s.system_prompt is None][0]
    messages = to_chat_messages(bare)
    assert [m["role"] for m in messages] == ["user"]
    assert messages[0]["content"] == "tek soru"


def test_chat_messages_include_system_when_present():
    catalog = make_catalog(n_roles=1)["roles"]
    specs = build_role_specs(catalog, ["tek soru"])
    messages = to_chat_messages(specs[0])
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == specs[0].system_prompt


def test_sample_index_distinguishes_repeated_default_rollouts():
    specs = build_default_specs(["tek soru"], samples_per_prompt=3)
    bare = [s for s in specs if s.system_prompt is None]
    assert sorted(s.sample_index for s in bare) == [0, 1, 2]
