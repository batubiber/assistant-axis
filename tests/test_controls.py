import numpy as np
import pytest

from aax.controls import CONTROL_KINDS, control_direction, direction_fingerprint


def _axis(d=64, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(d)
    return v / np.linalg.norm(v)


def _roles(n=20, d=64, seed=1):
    return np.random.default_rng(seed).standard_normal((n, d))


def test_kinds_are_the_three_preregistered_controls():
    assert CONTROL_KINDS == ("gaussian", "shuffled", "rolespan")


@pytest.mark.parametrize("kind", ["gaussian", "shuffled", "rolespan"])
def test_every_direction_is_unit_norm(kind):
    v = control_direction(kind, axis_layer=_axis(), role_vectors_layer=_roles(), seed=7)
    assert np.linalg.norm(v) == pytest.approx(1.0)


@pytest.mark.parametrize("kind", ["gaussian", "shuffled", "rolespan"])
def test_same_seed_same_vector(kind):
    kw = dict(axis_layer=_axis(), role_vectors_layer=_roles())
    a = control_direction(kind, seed=3, **kw)
    b = control_direction(kind, seed=3, **kw)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("kind", ["gaussian", "shuffled", "rolespan"])
def test_different_seed_different_vector(kind):
    kw = dict(axis_layer=_axis(), role_vectors_layer=_roles())
    assert not np.array_equal(
        control_direction(kind, seed=3, **kw), control_direction(kind, seed=4, **kw)
    )


def test_shuffled_preserves_the_coordinate_multiset():
    """Ağır kuyruklu büyüklük profili AYNEN korunmalı — kontrolün varlık sebebi bu."""
    v = _axis()
    out = control_direction("shuffled", axis_layer=v, role_vectors_layer=_roles(), seed=5)
    # İşaretler korunmalı: sadece büyüklükleri değil, imzalı değerleri de
    assert np.allclose(np.sort(out), np.sort(v))


def test_shuffled_actually_changes_the_direction():
    v = _axis()
    out = control_direction("shuffled", axis_layer=v, role_vectors_layer=_roles(), seed=5)
    assert abs(float(out @ v)) < 0.5


def test_rolespan_lies_in_the_span_of_the_role_vectors():
    """Span dışına düşen bir vektör bu kontrolü anlamsız kılar."""
    v, R = _axis(), _roles()
    out = control_direction("rolespan", axis_layer=v, role_vectors_layer=R, seed=9)
    # R'nin satır uzayına izdüşüm, vektörün kendisini geri vermeli
    proj = R.T @ np.linalg.lstsq(R.T, out, rcond=None)[0]
    assert np.allclose(proj, out, atol=1e-8)


def test_rolespan_is_orthogonal_to_the_axis():
    v, R = _axis(), _roles()
    out = control_direction("rolespan", axis_layer=v, role_vectors_layer=R, seed=9)
    assert abs(float(out @ v)) < 1e-8


def test_gaussian_is_not_aligned_with_the_axis():
    v = _axis()
    out = control_direction("gaussian", axis_layer=v, role_vectors_layer=_roles(), seed=11)
    assert abs(float(out @ v)) < 0.5


def test_unknown_kind_raises():
    with pytest.raises(ValueError, match="bilinmeyen"):
        control_direction("pirate", axis_layer=_axis(), role_vectors_layer=_roles(), seed=1)


def test_non_finite_axis_raises():
    v = _axis(); v[0] = np.nan
    with pytest.raises(ValueError, match="sonlu"):
        control_direction("shuffled", axis_layer=v, role_vectors_layer=_roles(), seed=1)


def test_dimension_mismatch_raises():
    with pytest.raises(ValueError, match="d_model"):
        control_direction(
            "rolespan", axis_layer=_axis(d=64), role_vectors_layer=_roles(d=32), seed=1
        )


def test_non_1d_axis_raises():
    with pytest.raises(ValueError, match="1 boyutlu"):
        control_direction(
            "gaussian", axis_layer=np.zeros((2, 64)), role_vectors_layer=_roles(), seed=1
        )


def test_fingerprint_is_stable_and_discriminating():
    a = control_direction("gaussian", axis_layer=_axis(), role_vectors_layer=_roles(), seed=1)
    b = control_direction("gaussian", axis_layer=_axis(), role_vectors_layer=_roles(), seed=2)
    assert direction_fingerprint(a) == direction_fingerprint(a)
    assert direction_fingerprint(a) != direction_fingerprint(b)
    assert len(direction_fingerprint(a)) == 16


@pytest.mark.parametrize("kind", ["gaussian", "shuffled", "rolespan"])
def test_returns_float64_dtype(kind):
    """Çıktı dtype'ı her zaman float64 olmalı — parmak iz istikrarı için."""
    # Girdi float32 olsa bile çıktı float64 olmalı
    v_float32 = _axis().astype(np.float32)
    roles_float32 = _roles().astype(np.float32)
    result = control_direction(kind, axis_layer=v_float32, role_vectors_layer=roles_float32, seed=42)
    assert result.dtype == np.float64, f"{kind} döndürüyor: {result.dtype}"

    # Girdi float64 olduğunda da float64 olmalı
    v_float64 = _axis()
    roles_float64 = _roles()
    result = control_direction(kind, axis_layer=v_float64, role_vectors_layer=roles_float64, seed=42)
    assert result.dtype == np.float64


def test_rolespan_raises_on_degenerate_axis():
    """Eksenin rol span'indeki izdüşümü sıfıra yakın olursa ValueError fırlatmalı.

    Dejenere durum: rol vektörleri matrisi tek sıra = eksenin kendisi.
    Span tek 1D, eksenin tamamen içinde değilse (sayısal olarak sıfır),
    ortogonalleştirme bir gürültü vektörü tutturmaz ve norm eşiğini geçer.
    Ama eksen TAMAMEN span'e eşitse (örn. eksen tek rol vektörü),
    çıkartma sonucu sıfır olur.
    """
    # Rol vektörü = eksen kendisi (tekil matris)
    v = _axis()
    # Rol vektörleri: sadece v'nin kendisini içeren 1×64 matris
    R_degenerate = v.reshape(1, -1)

    with pytest.raises(ValueError, match="sıfıra çok yakın"):
        control_direction("rolespan", axis_layer=v, role_vectors_layer=R_degenerate, seed=99)
