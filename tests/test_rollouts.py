import json

import pytest

from aax.prompts import RolloutSpec
from aax.rollouts import read_rollouts, rollout_record, write_rollouts


def make_spec(role="pirate", question="soru?"):
    return RolloutSpec(
        kind="role",
        role=role,
        system_prompt="You are a pirate.",
        question=question,
        sample_index=0,
    )


def test_record_carries_every_field_needed_downstream():
    record = rollout_record(make_spec(), "Arrr!")
    assert record["kind"] == "role"
    assert record["role"] == "pirate"
    assert record["system_prompt"] == "You are a pirate."
    assert record["question"] == "soru?"
    assert record["answer"] == "Arrr!"
    assert record["sample_index"] == 0


def test_write_then_read_roundtrips(tmp_path):
    records = [rollout_record(make_spec(question=f"s{i}"), f"a{i}") for i in range(5)]
    path = tmp_path / "rollouts.jsonl"
    write_rollouts(path, records)
    assert read_rollouts(path) == records


def test_write_is_atomic_no_temp_left_behind(tmp_path):
    path = tmp_path / "rollouts.jsonl"
    write_rollouts(path, [rollout_record(make_spec(), "x")])
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != "rollouts.jsonl"]
    assert leftovers == []


def test_read_rejects_truncated_file(tmp_path):
    path = tmp_path / "rollouts.jsonl"
    path.write_text('{"kind": "role"}\n{"kind": "ro', encoding="utf-8")
    with pytest.raises(ValueError, match="satır"):
        read_rollouts(path)


def test_empty_answer_is_rejected():
    with pytest.raises(ValueError, match="boş"):
        rollout_record(make_spec(), "   ")
