import importlib.util
import json
import sys
from pathlib import Path

import pytest

_P = Path(__file__).resolve().parents[1] / "scripts" / "08_steering_sweep.py"


def _load():
    spec = importlib.util.spec_from_file_location("steering_sweep", _P)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ss = _load()


def test_module_is_registered_in_sys_modules():
    assert "steering_sweep" in sys.modules


def test_plan_counts_layers_times_strengths_times_roles_times_questions():
    n = ss.planned_generation_count(n_layers=2, n_strengths=7, n_roles=50, n_questions=5)
    assert n == 2 * 7 * 50 * 5


def test_plan_rejects_zero_dimensions():
    with pytest.raises(ValueError, match="sıfır"):
        ss.planned_generation_count(n_layers=0, n_strengths=7, n_roles=50, n_questions=5)


def test_record_carries_every_field_downstream_needs():
    r = ss.sweep_record(layer=14, strength=-0.4, role="analyst",
                        question="Who are you?", answer="I am Alex.")
    assert r == {"layer": 14, "strength": -0.4, "role": "analyst",
                 "question": "Who are you?", "answer": "I am Alex."}


def test_record_rejects_blank_answer():
    with pytest.raises(ValueError, match="boş"):
        ss.sweep_record(layer=14, strength=0.0, role="analyst",
                        question="Who are you?", answer="   ")


def test_write_is_atomic_and_leaves_no_temp(tmp_path):
    path = tmp_path / "sweep.jsonl"
    ss.write_sweep(path, [ss.sweep_record(layer=14, strength=0.0, role="r",
                                          question="q", answer="a")])
    assert [p.name for p in tmp_path.iterdir()] == ["sweep.jsonl"]


def test_write_failure_leaves_existing_file_untouched(tmp_path):
    path = tmp_path / "sweep.jsonl"
    path.write_text("ONCEKI", encoding="utf-8")

    class Boom(list):
        def __iter__(self):
            yield ss.sweep_record(layer=1, strength=0.0, role="r", question="q", answer="a")
            raise RuntimeError("bilerek")

    with pytest.raises(RuntimeError):
        ss.write_sweep(path, Boom())
    assert path.read_text(encoding="utf-8") == "ONCEKI"
    assert [p.name for p in tmp_path.iterdir()] == ["sweep.jsonl"]


def test_read_rejects_a_truncated_file(tmp_path):
    path = tmp_path / "sweep.jsonl"
    path.write_text('{"layer": 14}\n{"layer": 1', encoding="utf-8")
    with pytest.raises(ValueError, match="satır"):
        ss.read_sweep(path)
