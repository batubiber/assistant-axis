import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "judge_gate", Path(__file__).resolve().parents[1] / "scripts" / "03_judge_gate.py"
)
judge_gate = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(judge_gate)


def test_collapse_maps_paper_rubric_to_three_categories():
    assert judge_gate.collapse_to_category(3) == "fully"
    assert judge_gate.collapse_to_category(2) == "somewhat"
    assert judge_gate.collapse_to_category(1) == "no"
    assert judge_gate.collapse_to_category(0) == "no"


def test_collapse_rejects_out_of_range():
    with pytest.raises(ValueError):
        judge_gate.collapse_to_category(4)


def test_agreement_is_computed_on_collapsed_categories():
    """0 ve 1 aynı kategoriye düştüğü için bu çift UYUŞUR."""
    assert judge_gate.agreement_rate([0], [1]) == 1.0


def test_agreement_counts_category_mismatch():
    assert judge_gate.agreement_rate([3, 3], [3, 2]) == 0.5


def test_agreement_rejects_length_mismatch():
    with pytest.raises(ValueError, match="uzunluk"):
        judge_gate.agreement_rate([1, 2], [1])


def test_agreement_rejects_empty_input():
    with pytest.raises(ValueError, match="boş"):
        judge_gate.agreement_rate([], [])


def test_gate_passes_at_threshold_exactly():
    assert judge_gate.gate_passed(0.75) is True
    assert judge_gate.gate_passed(0.7499) is False
