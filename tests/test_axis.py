import numpy as np
import pytest

from aax.axis import (
    contrast_axis,
    cosine,
    evaluate_criterion_a,
    n_components_for_variance,
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


def test_cosine_rejects_non_finite_input():
    """NaN yayılmamalı: `na == 0` kontrolü NaN için ateşlemez, o yüzden ayrı
    bir sonluluk kontrolü şart — yoksa boş dilim ortalaması sessizce geçer."""
    nan_vector = np.array([np.nan, 0.0, 0.0])
    with pytest.raises(ValueError, match="sonlu olmayan"):
        cosine(nan_vector, np.ones(3))
    with pytest.raises(ValueError, match="sonlu olmayan"):
        cosine(np.ones(3), nan_vector)
    with pytest.raises(ValueError, match="sonlu olmayan"):
        cosine(np.array([np.inf, 0.0, 0.0]), np.ones(3))


def test_contrast_axis_is_unit_norm():
    axis = contrast_axis(np.array([5.0, 0.0]), np.array([1.0, 0.0]))
    assert np.linalg.norm(axis) == pytest.approx(1.0)


def test_contrast_axis_points_from_roles_toward_default():
    axis = contrast_axis(np.array([10.0, 0.0]), np.array([2.0, 0.0]))
    assert axis[0] > 0


def test_contrast_axis_rejects_non_finite_input():
    """Boş bir dilimin ortalaması NaN'dır; `norm == 0` bunu yakalamaz."""
    nan_mean = np.full(3, np.nan)
    with pytest.raises(ValueError, match="sonlu olmayan"):
        contrast_axis(np.ones(3), nan_mean)
    with pytest.raises(ValueError, match="sonlu olmayan"):
        contrast_axis(nan_mean, np.ones(3))


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


def test_pca_centring_survives_a_large_shared_offset():
    """Ortalama çıkarma (centring) regresyon koruması.

    `test_pca_recovers_a_planted_direction`'daki sabit tohumlu örneklemin
    örnek ortalaması zaten ~0 olduğu için centring silinse bile geçiyor.
    Gerçek aktivasyonlarda ise tüm rollerin paylaştığı BÜYÜK bir ortak
    ortalama var — centring tam da o zaman kritik. Burada ekilmiş yöne dik,
    normu varyanstan çok daha büyük bir ofset ekleniyor: centring olmadan
    SVD'nin ilk sağ tekil vektörü ekilmiş yönü değil ofsetin yönünü bulur ve
    aşağıdaki kosinüs çöker.
    """
    rng = np.random.default_rng(1)
    planted = np.array([0.0, 1.0, 0.0, 0.0])
    offset = np.array([50.0, 0.0, 0.0, 0.0])
    scores = rng.normal(scale=1.0, size=200)
    noise = rng.normal(scale=0.05, size=(200, 4))
    vectors = offset[None, :] + scores[:, None] * planted[None, :] + noise

    components, ratios = pca_components(vectors, n_components=2)
    assert abs(cosine(components[0], planted)) > 0.99
    assert ratios[0] > 0.9


def test_role_vectors_skip_categories_below_minimum():
    acts = np.ones((12, 2, 3), dtype=np.float32)
    roles = ["a"] * 12
    cats = ["fully"] * 9 + ["no"] * 3
    vectors, names, categories = role_vectors(acts, roles, cats, min_responses=10)
    assert names == []
    assert categories == []


def test_role_vectors_averages_qualifying_rows():
    acts = np.zeros((10, 1, 2), dtype=np.float32)
    acts[:, 0, 0] = 4.0
    roles = ["pirate"] * 10
    cats = ["fully"] * 10
    vectors, names, categories = role_vectors(acts, roles, cats, min_responses=10)
    assert names == ["pirate"]
    assert categories == ["fully"]
    assert vectors.shape == (1, 1, 2)
    assert vectors[0, 0, 0] == pytest.approx(4.0)


def test_role_vectors_keeps_fully_and_somewhat_separate():
    acts = np.zeros((20, 1, 2), dtype=np.float32)
    acts[:10, 0, 0] = 1.0
    acts[10:, 0, 0] = 9.0
    roles = ["bard"] * 20
    cats = ["fully"] * 10 + ["somewhat"] * 10
    vectors, names, categories = role_vectors(acts, roles, cats, min_responses=10)
    assert sorted(names) == ["bard::fully", "bard::somewhat"]
    assert sorted(categories) == ["fully", "somewhat"]


def test_role_vectors_reports_category_even_when_name_is_ambiguous():
    """Gösterim ismi tek başına kategoriyi ayırt etmez.

    Yalnızca `somewhat`'ı eşiği geçen bir rol "rol" adını alır — yalnızca
    `fully`'si geçen bir rolle aynı. Eksen sadece `fully` vektörlerinden
    hesaplandığı için çağıranın kategoriyi isimden tahmin etmesi imkânsız;
    bu yüzden ayrı olarak dönmeli.
    """
    acts = np.zeros((30, 1, 2), dtype=np.float32)
    acts[:10, 0, 0] = 1.0  # sadece_somewhat, 10 satır somewhat
    acts[10:20, 0, 0] = 2.0  # sadece_fully, 10 satır fully
    acts[20:, 0, 0] = 3.0  # sadece_fully, 10 satır no (elenir)
    roles = ["sadece_somewhat"] * 10 + ["sadece_fully"] * 10 + ["sadece_fully"] * 10
    cats = ["somewhat"] * 10 + ["fully"] * 10 + ["no"] * 10

    vectors, names, categories = role_vectors(acts, roles, cats, min_responses=10)

    assert names == ["sadece_fully", "sadece_somewhat"]
    assert categories == ["fully", "somewhat"]
    assert len(categories) == len(names) == vectors.shape[0]
    # kategori üzerinden seçim: eksen yalnızca bu vektörden hesaplanmalı
    fully_only = vectors[[i for i, c in enumerate(categories) if c == "fully"]]
    assert fully_only.shape[0] == 1
    assert fully_only[0, 0, 0] == pytest.approx(2.0)


def test_n_components_for_variance_counts_against_the_full_spectrum():
    """Kesilmiş spektrumdan kesin sayı okunamaz.

    20 eşit bileşende %70'e ulaşmak 14 bileşen ister. İlk 10 oranla
    `np.searchsorted(cumsum, 0.70)` doyuma ulaşıp 11 derdi — gerçekten
    desteklenmeyen, yanıltıcı biçimde DÜŞÜK bir sayı.
    """
    full = np.full(20, 1 / 20)
    assert n_components_for_variance(full, 0.70) == 14
    assert n_components_for_variance(full[:10], 0.70) is None
    assert n_components_for_variance(np.array([0.8, 0.2]), 0.70) == 1
    assert n_components_for_variance(np.array([]), 0.70) is None


def test_projection_percentile_at_extremes():
    dist = np.arange(100.0)
    assert projection_percentile(-5.0, dist) == pytest.approx(0.0)
    assert projection_percentile(200.0, dist) == pytest.approx(1.0)


def test_projection_percentile_rejects_non_finite_value():
    """NaN sessizce 0.0'a çözülürse `evaluate_criterion_a` bunu alt desilin
    İÇİNDE sayıp yanlış yönde (GEÇTİ'ye doğru) bir sonuç üretir."""
    dist = np.arange(10.0)
    with pytest.raises(ValueError, match="sonlu olmayan"):
        projection_percentile(float("nan"), dist)
    with pytest.raises(ValueError, match="sonlu olmayan"):
        projection_percentile(float("inf"), dist)


def test_projection_percentile_rejects_non_finite_distribution():
    dist = np.array([1.0, np.nan, 3.0])
    with pytest.raises(ValueError, match="sonlu olmayan"):
        projection_percentile(1.0, dist)


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


def test_criterion_a_accepts_negative_cosine_when_default_is_in_the_bottom_decile():
    """PC1'in işareti keyfîdir (SVD seçer) — ama işaret hangi desilin
    BEKLENDİĞİNİ belirler. Negatif kosinüs + alt desil, pozitif kosinüs +
    üst desilin ayna görüntüsüdür ve aynı fiziksel duruma karşılık gelir."""
    result = evaluate_criterion_a(cos_pc1_axis=-0.75, default_percentile=0.02)
    assert result["passed"] is True


# --- A1: işaret × desil eşleşmesi (coupled) ---------------------------------
#
# Eskiden iki koşul BAĞIMSIZDI: `|cos| > 0.6` VE `(persentil >= 0.9 VEYA
# persentil <= 0.1)`. Bu, geçme bölgesini ikiye katlıyordu ve bölgenin YARISI
# hipotezin ALEYHİNE delildi. Aşağıdaki dört test, işaret × desil
# kombinasyonlarının tamamını sabitler.


def test_criterion_a_positive_cosine_requires_the_top_decile():
    result = evaluate_criterion_a(cos_pc1_axis=0.95, default_percentile=0.97)
    assert result["passed"] is True
    assert result["required_decile"] == "top"


def test_criterion_a_positive_cosine_with_bottom_decile_fails():
    """A1'in başlık regresyonu.

    Düzeltme öncesi bu tam olarak `passed: True` dönüyordu (doğrulandı).
    Eksen rollerden default'a doğru bakar; PC1 onunla AYNI yöndeyken default
    projeksiyonunun HER rol vektörünün ALTINDA kalması, ölçülen kriterin tam
    tersidir — geçme değil, hipoteze karşı delildir.
    """
    result = evaluate_criterion_a(cos_pc1_axis=0.95, default_percentile=0.0)
    assert result["passed"] is False
    assert result["required_decile"] == "top"
    assert "POZİTİF" in result["reason"]
    assert "ÜST desilde" in result["reason"]


def test_criterion_a_negative_cosine_requires_the_bottom_decile():
    result = evaluate_criterion_a(cos_pc1_axis=-0.95, default_percentile=0.03)
    assert result["passed"] is True
    assert result["required_decile"] == "bottom"


def test_criterion_a_negative_cosine_with_top_decile_fails():
    """Aynalı regresyon: bağımsız testlerde bu da `passed: True` derdi."""
    result = evaluate_criterion_a(cos_pc1_axis=-0.95, default_percentile=1.0)
    assert result["passed"] is False
    assert result["required_decile"] == "bottom"
    assert "NEGATİF" in result["reason"]
    assert "ALT desilde" in result["reason"]


def test_criterion_a_zero_cosine_has_no_required_decile_and_fails():
    """`cos == 0`: işaret yok, dolayısıyla istenen desil de tanımsız.
    `|cos| <= 0.6` zaten düşürüyor, ama gerekçe her iki koşulu da adlandırmalı."""
    result = evaluate_criterion_a(cos_pc1_axis=0.0, default_percentile=1.0)
    assert result["passed"] is False
    assert result["required_decile"] is None
    assert "işaret" in result["reason"]


def test_criterion_a_never_passes_on_a_nan_cosine():
    """Sahte GEÇTİ yolu.

    NaN ile yapılan her karşılaştırma False döner: `abs(nan) <= 0.6` False
    olduğu için eski kod hiçbir gerekçe eklemez ve `passed` True çıkardı —
    yani tanımsız veriden "A KRİTERİ: GEÇTİ".
    """
    result = evaluate_criterion_a(cos_pc1_axis=float("nan"), default_percentile=0.95)
    assert result["passed"] is False
    assert "sonlu değil" in result["reason"]


def test_criterion_a_never_passes_on_a_nan_percentile():
    result = evaluate_criterion_a(cos_pc1_axis=0.9, default_percentile=float("nan"))
    assert result["passed"] is False
    assert "sonlu değil" in result["reason"]


def test_criterion_a_rejects_infinite_values():
    for cos_value, percentile in ((float("inf"), 0.95), (0.9, float("-inf"))):
        result = evaluate_criterion_a(cos_pc1_axis=cos_value, default_percentile=percentile)
        assert result["passed"] is False
        assert "sonlu değil" in result["reason"]


def test_criterion_a_boundary_bottom_decile_exactly_0_1_passes():
    """`1 - TOP_DECILE` ikili kayan noktada `0.09999999999999998`'tir —
    tam `0.1` persentili (n 10'un katıysa `k/n` ile ATTAINABLE, beklenen
    ölçekte rutin) bu ifadeyle KAÇARDI. `BOTTOM_DECILE = 0.1` sabiti bunu
    düzeltir; sınır ULP'siz, ayna simetrik olmalı.

    A1 sonrası kosinüs NEGATİF: alt desil yalnızca negatif işaretle
    istenir. Sınamanın konusu değişmedi — sınırın tam `0.1` olması."""
    result = evaluate_criterion_a(cos_pc1_axis=-0.9, default_percentile=0.1)
    assert result["passed"] is True


def test_criterion_a_boundary_top_decile_exactly_0_9_passes():
    """Aynalı üst sınır — regresyon: bu her zaman geçiyordu, alt sınırla
    aynı davranması gerektiğini doğrulamak için burada."""
    result = evaluate_criterion_a(cos_pc1_axis=0.9, default_percentile=0.9)
    assert result["passed"] is True
