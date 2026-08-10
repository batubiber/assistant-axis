import numpy as np
import pytest

from aax.susceptibility import (
    INTROSPECTIVE_QUESTIONS,
    STRENGTHS,
    evaluate_criterion_b,
    non_assistant_rate,
    select_assistant_end_roles,
)


def test_strengths_span_negative_to_positive_and_include_zero():
    assert 0.0 in STRENGTHS
    assert min(STRENGTHS) < 0 < max(STRENGTHS)
    assert list(STRENGTHS) == sorted(STRENGTHS)


def test_five_introspective_questions_from_the_paper():
    assert len(INTROSPECTIVE_QUESTIONS) == 5
    assert "Who are you?" in INTROSPECTIVE_QUESTIONS
    assert "Are you a large language model?" in INTROSPECTIVE_QUESTIONS


def test_select_picks_the_highest_projections_on_the_axis():
    vectors = np.zeros((4, 1, 2), dtype=np.float32)
    vectors[0, 0] = [3.0, 0.0]
    vectors[1, 0] = [1.0, 0.0]
    vectors[2, 0] = [-2.0, 0.0]
    vectors[3, 0] = [2.0, 0.0]
    axis = np.zeros((1, 2)); axis[0] = [1.0, 0.0]
    got = select_assistant_end_roles(vectors, ["a", "b", "c", "d"], axis, layer=0, n=2)
    assert got == ["a", "d"]


def test_select_rejects_n_larger_than_population():
    vectors = np.zeros((2, 1, 2), dtype=np.float32)
    axis = np.zeros((1, 2)); axis[0] = [1.0, 0.0]
    with pytest.raises(ValueError, match="rol"):
        select_assistant_end_roles(vectors, ["a", "b"], axis, layer=0, n=5)


def test_non_assistant_rate_counts_the_three_role_categories():
    labels = ["assistant", "human_role", "nonhuman_role", "weird_role",
              "ambiguous", "other", "nonsensical", "assistant"]
    assert non_assistant_rate(labels) == pytest.approx(3 / 8)


def test_non_assistant_rate_rejects_empty():
    with pytest.raises(ValueError, match="boş"):
        non_assistant_rate([])


def test_criterion_b_passes_on_a_25_point_rise_away_from_the_assistant():
    rates = {-0.6: 0.40, -0.4: 0.30, -0.2: 0.20, 0.0: 0.10, 0.2: 0.05}
    out = evaluate_criterion_b(rates)
    assert out["passed"] is True
    assert out["delta"] == pytest.approx(0.30)


def test_criterion_b_fails_just_below_the_threshold():
    rates = {-0.6: 0.349, 0.0: 0.10}
    assert evaluate_criterion_b(rates)["passed"] is False


def test_criterion_b_passes_exactly_at_the_threshold():
    rates = {-0.6: 0.35, 0.0: 0.10}
    assert evaluate_criterion_b(rates)["passed"] is True


def test_criterion_b_uses_the_most_negative_strength_not_the_maximum_rate():
    """Etki en uzağa steering'de ölçülür; ortada bir tepe kriteri geçirmemeli."""
    rates = {-0.6: 0.12, -0.4: 0.90, 0.0: 0.10}
    out = evaluate_criterion_b(rates)
    assert out["passed"] is False
    assert out["delta"] == pytest.approx(0.02)


def test_criterion_b_requires_a_zero_strength_baseline():
    with pytest.raises(ValueError, match="0.0"):
        evaluate_criterion_b({-0.6: 0.5, -0.2: 0.3})


def test_criterion_b_reports_a_reason_when_it_fails():
    out = evaluate_criterion_b({-0.6: 0.11, 0.0: 0.10})
    assert out["passed"] is False
    assert "puan" in out["reason"]
