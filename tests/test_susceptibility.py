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


# --- Fix Round 1 ---------------------------------------------------------


def test_select_rejects_non_positive_n():
    vectors = np.zeros((3, 1, 2), dtype=np.float32)
    axis = np.zeros((1, 2)); axis[0] = [1.0, 0.0]
    with pytest.raises(ValueError, match="pozitif"):
        select_assistant_end_roles(vectors, ["a", "b", "c"], axis, layer=0, n=0)
    with pytest.raises(ValueError, match="pozitif"):
        select_assistant_end_roles(vectors, ["a", "b", "c"], axis, layer=0, n=-1)


def test_select_rejects_name_vector_length_mismatch():
    axis = np.zeros((1, 2)); axis[0] = [1.0, 0.0]
    vectors = np.zeros((3, 1, 2), dtype=np.float32)
    # isim sayısı vektör sayısından FAZLA
    with pytest.raises(ValueError, match="uyuşmuyor"):
        select_assistant_end_roles(vectors, ["a", "b", "c", "d"], axis, layer=0, n=1)
    # isim sayısı vektör sayısından AZ
    with pytest.raises(ValueError, match="uyuşmuyor"):
        select_assistant_end_roles(vectors, ["a", "b"], axis, layer=0, n=1)


def test_select_rejects_layer_out_of_range_for_vectors():
    vectors = np.zeros((2, 1, 2), dtype=np.float32)
    axis = np.zeros((1, 2)); axis[0] = [1.0, 0.0]
    with pytest.raises(ValueError, match="katman aralık dışı"):
        select_assistant_end_roles(vectors, ["a", "b"], axis, layer=5, n=1)


def test_select_rejects_layer_out_of_range_for_axis():
    # vectors'ün katman boyutu layer=1'i barındırır ama axis'inki barındırmaz
    vectors = np.zeros((2, 2, 2), dtype=np.float32)
    axis = np.zeros((1, 2)); axis[0] = [1.0, 0.0]
    with pytest.raises(ValueError, match="katman aralık dışı"):
        select_assistant_end_roles(vectors, ["a", "b"], axis, layer=1, n=1)


def test_criterion_b_rejects_when_no_negative_strength_is_measured():
    with pytest.raises(ValueError, match="negatif"):
        evaluate_criterion_b({0.0: 0.1, 0.2: 0.9})


def test_criterion_b_rejects_non_finite_rates():
    with pytest.raises(ValueError, match="sonlu"):
        evaluate_criterion_b({0.0: float("nan"), -0.6: 0.5})


def test_epsilon_does_not_swallow_a_realistic_sub_threshold_delta():
    """1e-9'luk epsilon, hücre başı N~250 ile üretilebilecek gerçek bir
    eşik-altı delta'yı YUTMAMALI — sadece float64 çıkarma gürültüsünü
    emmeli."""
    n = 250
    rates = {-0.6: 62 / n, 0.0: 0 / n}  # delta = 0.248, eşiğin ~0.002 altı
    out = evaluate_criterion_b(rates)
    assert out["passed"] is False


def test_epsilon_lets_the_papers_own_threshold_float_noise_pass():
    """`0.35 - 0.10` float64'te tam 0.25 değil, 0.24999999999999997 çıkar;
    plan bunun GEÇMESİNİ varsayıyordu (bkz.
    test_criterion_b_passes_exactly_at_the_threshold)."""
    assert 0.35 - 0.10 != 0.25  # float64 gürültüsünün varlığını doğrula
    out = evaluate_criterion_b({-0.6: 0.35, 0.0: 0.10})
    assert out["passed"] is True
    assert out["threshold_eps"] == pytest.approx(1e-9)


def test_criterion_b_returns_the_full_documented_schema():
    rates = {-0.6: 0.40, 0.0: 0.10}
    out = evaluate_criterion_b(rates)
    assert set(out.keys()) == {
        "baseline_strength",
        "baseline_rate",
        "far_strength",
        "far_rate",
        "delta",
        "threshold",
        "threshold_eps",
        "passed",
        "reason",
    }
    assert out["baseline_strength"] == 0.0
    assert out["baseline_rate"] == pytest.approx(0.10)
    assert out["far_strength"] == -0.6
    assert out["far_rate"] == pytest.approx(0.40)
    assert out["delta"] == pytest.approx(0.30)
    assert out["threshold"] == pytest.approx(0.25)
    assert out["threshold_eps"] == pytest.approx(1e-9)
    assert out["passed"] is True
    assert out["reason"] == "eşik aşıldı"
