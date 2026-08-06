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


class _BoomPartway:
    """N kayıttan sonra patlayan sahte kayıt akışı.

    Yarıda kesilen bir yazımı simüle eder: `write_rollouts` bunu iterate
    ederken bazı kayıtlar diske gitmiş olur, sonra `RuntimeError` fırlar —
    tıpkı 16.000 kayıtlık gerçek koşuda ortasında kesilen bir işlem gibi.
    """

    def __init__(self, n_ok: int):
        self.n_ok = n_ok

    def __iter__(self):
        for i in range(self.n_ok):
            yield rollout_record(make_spec(question=f"s{i}"), f"a{i}")
        raise RuntimeError("yazım ortasında simüle edilmiş çökme")


def test_write_failure_partway_leaves_no_temp_and_preserves_existing_target(tmp_path):
    """Gerçek atomiklik garantisi: yarıda kesilen yazım ne temp dosya bırakır
    ne de var olan hedefi bozar. Sahte, tempfile'sız bir
    `path.write_text(...)` bu testi geçemez (hedefi kısmi içerikle üzerine
    yazardı); `test_write_is_atomic_no_temp_left_behind` bunu ayırt edemezdi
    çünkü yalnızca başarılı bir yazımdan sonra bakıyordu."""
    path = tmp_path / "rollouts.jsonl"
    original = json.dumps(rollout_record(make_spec(), "mevcut"), ensure_ascii=False) + "\n"
    path.write_text(original, encoding="utf-8")

    with pytest.raises(RuntimeError, match="yazım ortasında"):
        write_rollouts(path, _BoomPartway(n_ok=2))

    leftovers = [p.name for p in tmp_path.iterdir() if p.name != "rollouts.jsonl"]
    assert leftovers == []
    assert path.read_text(encoding="utf-8") == original


def test_read_rejects_truncated_file(tmp_path):
    path = tmp_path / "rollouts.jsonl"
    path.write_text('{"kind": "role"}\n{"kind": "ro', encoding="utf-8")
    with pytest.raises(ValueError, match="satır"):
        read_rollouts(path)


def test_empty_answer_is_rejected():
    with pytest.raises(ValueError, match="boş"):
        rollout_record(make_spec(), "   ")
