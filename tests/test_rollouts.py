import json

import pytest

from aax.prompts import RolloutSpec
from aax.rollouts import (
    load_rollouts_meta,
    read_rollouts,
    rollout_record,
    rollouts_meta_payload,
    rollouts_run_id,
    write_rollouts,
    write_rollouts_meta,
)


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


# --- C5: rollouts.jsonl'ın künyesi (pilot artefaktı kanonik sanılmasın) ------


def _records(n: int, role: str = "pirate") -> list[dict]:
    return [rollout_record(make_spec(role=role, question=f"s{i}"), f"a{i}") for i in range(n)]


def test_run_id_is_content_derived_and_stable():
    assert rollouts_run_id(_records(3)) == rollouts_run_id(_records(3))
    assert len(rollouts_run_id(_records(3))) == 16
    assert rollouts_run_id(_records(3)) != rollouts_run_id(_records(3, role="sage"))


def test_run_id_ignores_the_answer_text():
    """Kimlik "hangi rollout kümesi" sorusunu yanıtlar, "model o gün ne
    üretti" sorusunu değil: aynı spec'ler farklı yanıtlarla aynı kimliği
    vermeli, yoksa `05` ile `06` aynı dosyayı okusa bile ayrışabilirdi."""
    a = [rollout_record(make_spec(question=f"s{i}"), f"cevap-{i}") for i in range(3)]
    b = [rollout_record(make_spec(question=f"s{i}"), f"BAMBAŞKA-{i}") for i in range(3)]
    assert rollouts_run_id(a) == rollouts_run_id(b)


def test_meta_payload_carries_limit_n_and_run_id():
    records = _records(4)
    assert rollouts_meta_payload(records, None) == {
        "n": 4,
        "limit": None,
        "run_id": rollouts_run_id(records),
    }
    assert rollouts_meta_payload(records, 100)["limit"] == 100


def test_load_meta_accepts_a_matching_canonical_artifact(tmp_path):
    records = _records(4)
    path = tmp_path / "rollouts_meta.json"
    write_rollouts_meta(path, records, None)
    assert load_rollouts_meta(path, records)["n"] == 4


def test_load_meta_rejects_a_pilot_artifact_unless_allowed(tmp_path):
    """Aşama 0'ın `load_role_catalog` deseninin aynısı: `--limit` ile
    üretilmiş bir dosya kanonik yolda dursa bile kanonik DEĞİLDİR."""
    records = _records(4)
    path = tmp_path / "rollouts_meta.json"
    write_rollouts_meta(path, records, 100)

    with pytest.raises(ValueError, match="PİLOT"):
        load_rollouts_meta(path, records)
    assert load_rollouts_meta(path, records, allow_pilot=True)["limit"] == 100


def test_load_meta_rejects_a_missing_file(tmp_path):
    with pytest.raises(ValueError, match="yok"):
        load_rollouts_meta(tmp_path / "rollouts_meta.json", _records(2))


def test_load_meta_rejects_a_stale_file_from_another_run(tmp_path):
    path = tmp_path / "rollouts_meta.json"
    write_rollouts_meta(path, _records(4), None)

    with pytest.raises(ValueError, match="tarif etmiyor"):
        load_rollouts_meta(path, _records(4, role="sage"))  # aynı sayı, farklı içerik
    with pytest.raises(ValueError, match="tarif etmiyor"):
        load_rollouts_meta(path, _records(3))  # farklı sayı


def test_load_meta_rejects_a_corrupt_file(tmp_path):
    path = tmp_path / "rollouts_meta.json"
    path.write_text("{bozuk", encoding="utf-8")
    with pytest.raises(ValueError, match="bozuk"):
        load_rollouts_meta(path, _records(2))
