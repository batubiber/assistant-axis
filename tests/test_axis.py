import numpy as np
import pytest

from aax.axis import (
    contrast_axis,
    cosine,
    evaluate_criterion_a,
    pca_components,
    projection_percentile,
    role_vectors,
)


def test_cosine_of_identical_vectors_is_one():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine(v, v) == pytest.approx(1.0)


def test_cosine_of_opposite_vectors_is_minus_one():
    v = np.array([1.0, 0.0])
    assert cosine(v, -v) == pytest.approx(-1.0)


def test_cosine_rejects_zero_vector():
    with pytest.raises(ValueError, match="sıfır"):
        cosine(np.zeros(3), np.ones(3))


def test_contrast_axis_is_unit_norm():
    axis = contrast_axis(np.array([5.0, 0.0]), np.array([1.0, 0.0]))
    assert np.linalg.norm(axis) == pytest.approx(1.0)


def test_contrast_axis_points_from_roles_toward_default():
    axis = contrast_axis(np.array([10.0, 0.0]), np.array([2.0, 0.0]))
    assert axis[0] > 0


def test_pca_recovers_a_planted_direction():
    """Ekilmiş bir yön varsa PC1 onu bulmalı."""
    rng = np.random.default_rng(0)
    planted = np.array([1.0, 0.0, 0.0, 0.0])
    scores = rng.normal(scale=5.0, size=200)
    noise = rng.normal(scale=0.1, size=(200, 4))
    vectors = scores[:, None] * planted[None, :] + noise

    components, ratios = pca_components(vectors, n_components=2)
    assert abs(cosine(components[0], planted)) > 0.99
    assert ratios[0] > 0.9


def test_role_vectors_skip_categories_below_minimum():
    acts = np.ones((12, 2, 3), dtype=np.float32)
    roles = ["a"] * 12
    cats = ["fully"] * 9 + ["no"] * 3
    vectors, names = role_vectors(acts, roles, cats, min_responses=10)
    assert names == []


def test_role_vectors_averages_qualifying_rows():
    acts = np.zeros((10, 1, 2), dtype=np.float32)
    acts[:, 0, 0] = 4.0
    roles = ["pirate"] * 10
    cats = ["fully"] * 10
    vectors, names = role_vectors(acts, roles, cats, min_responses=10)
    assert names == ["pirate"]
    assert vectors.shape == (1, 1, 2)
    assert vectors[0, 0, 0] == pytest.approx(4.0)


def test_role_vectors_keeps_fully_and_somewhat_separate():
    acts = np.zeros((20, 1, 2), dtype=np.float32)
    acts[:10, 0, 0] = 1.0
    acts[10:, 0, 0] = 9.0
    roles = ["bard"] * 20
    cats = ["fully"] * 10 + ["somewhat"] * 10
    vectors, names = role_vectors(acts, roles, cats, min_responses=10)
    assert sorted(names) == ["bard::fully", "bard::somewhat"]


def test_projection_percentile_at_extremes():
    dist = np.arange(100.0)
    assert projection_percentile(-5.0, dist) == pytest.approx(0.0)
    assert projection_percentile(200.0, dist) == pytest.approx(1.0)


def test_criterion_a_passes_when_both_conditions_hold():
    result = evaluate_criterion_a(cos_pc1_axis=0.72, default_percentile=0.95)
    assert result["passed"] is True


def test_criterion_a_fails_on_low_cosine():
    result = evaluate_criterion_a(cos_pc1_axis=0.41, default_percentile=0.98)
    assert result["passed"] is False
    assert "cos" in result["reason"]


def test_criterion_a_fails_when_default_not_in_top_decile():
    result = evaluate_criterion_a(cos_pc1_axis=0.80, default_percentile=0.55)
    assert result["passed"] is False
    assert "desil" in result["reason"]


def test_criterion_a_accepts_negative_cosine_by_magnitude():
    """PC1'in işareti keyfîdir; önemli olan büyüklük."""
    result = evaluate_criterion_a(cos_pc1_axis=-0.75, default_percentile=0.02)
    assert result["passed"] is True
