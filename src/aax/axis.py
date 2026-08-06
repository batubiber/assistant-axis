"""Persona uzayı analizi — saf numpy.

Bu modül model, GPU veya ağ bilmez. Girdi vektör matrisleri, çıktı eksen ve
PCA sonuçları. Bu sayede ekilmiş bir yönle sentetik veride tam test edilebilir
(spec Bölüm 4.3, Bölüm 10).
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

COS_THRESHOLD = 0.6
TOP_DECILE = 0.9
# `1 - TOP_DECILE` GÖRÜNÜŞTE aynı şeydir ama değildir: ikili kayan noktada
# `1 - 0.9 == 0.09999999999999998`, yani tam `0.1` persentili (`percentile =
# k/n` roller üzerinde, n 10'un katıysa ulaşılabilir — beklenen ölçekte
# rutin) alt desil testini KAÇIRIR, oysa aynalı `0.9` üst desil testini
# geçer. A kriteri ön kaydedilmiş: sınır tam olmalı, bir ULP'lik asimetri
# olmamalı. `BOTTOM_DECILE` bu yüzden `1 - TOP_DECILE` olarak DEĞİL, açıkça
# `0.1` olarak tanımlanır.
BOTTOM_DECILE = 0.1


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """İki vektörün kosinüs benzerliği.

    NaN/inf sessizce yayılmaz: sonlu olmayan girdi ya da sonlu olmayan sonuç
    `ValueError` ile reddedilir. Aksi hâlde boş bir dilimin ortalamasından
    doğan bir NaN, `evaluate_criterion_a`'ya kadar gidip sahte bir "GEÇTİ"
    üretir (NaN karşılaştırmaları hep False'tur).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError(
            "kosinüs sonlu olmayan (NaN/inf) değer içeren vektörle tanımsız — "
            "girdi büyük olasılıkla boş bir dilimin ortalamasından geliyor"
        )
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        raise ValueError("sıfır vektörün kosinüsü tanımsız")
    value = float(np.dot(a, b) / (na * nb))
    if not math.isfinite(value):
        raise ValueError("kosinüs sonlu olmayan bir değere çözüldü")
    return value


def contrast_axis(default_mean: np.ndarray, role_mean: np.ndarray) -> np.ndarray:
    """Assistant Axis = mean(default) − mean(rol vektörleri), L2 normalize.

    Makale bunu PC1'e tercih ediyor: PC1'in her modelde aynı anlamı taşıyacağı
    garanti değil (Ek G.5).

    `cosine` ile aynı gerekçe: sonlu olmayan girdi/çıktı sessizce geçmez.
    `norm == 0` kontrolü NaN için ateşlemez, bu yüzden ayrı bir sonluluk
    kontrolü şart.
    """
    default_mean = np.asarray(default_mean, dtype=np.float64)
    role_mean = np.asarray(role_mean, dtype=np.float64)
    if not np.isfinite(default_mean).all() or not np.isfinite(role_mean).all():
        raise ValueError(
            "kontrast vektörü sonlu olmayan (NaN/inf) girdiyle tanımsız — "
            "default veya rol ortalaması büyük olasılıkla boş bir dilimden geliyor"
        )
    axis = default_mean - role_mean
    norm = np.linalg.norm(axis)
    if norm == 0:
        raise ValueError("kontrast vektörü sıfır — default ve rol ortalamaları aynı")
    axis = axis / norm
    if not np.isfinite(axis).all():
        raise ValueError("kontrast vektörü sonlu olmayan bir değere çözüldü")
    return axis


def role_vectors(
    activations: np.ndarray,
    row_roles: list[str],
    row_categories: list[str],
    *,
    min_responses: int = 10,
) -> tuple[np.ndarray, list[str], list[str]]:
    """(rol, kategori) başına ortalama aktivasyon.

    Makalenin kuralı: bir kategori en az `min_responses` yanıt içermiyorsa
    o vektör hesaplanmaz. fully ve somewhat ayrı vektörler üretir.

    Dönüş: ([n_vectors, n_layers, d_model], isimler, kategoriler).

    `isimler` gösterim içindir: "rol::kategori", ya da o rolden tek kategori
    kaldıysa sadece "rol". Bu kural isimleri tek başına ayırt edici KILMAZ —
    yalnızca `somewhat`'ı kalan bir rol de sadece "rol" adını alır ve yalnızca
    `fully`'si kalan bir rolden ayırt edilemez. `kategoriler` bu nedenle ayrı
    döner: Assistant Axis yalnızca `fully` rol vektörlerinden hesaplandığı için
    çağıranın kategoriyi isimden tahmin etmesi değil, doğrudan bilmesi gerekir.
    `kategoriler[i]` her zaman `isimler[i]` ile aynı vektöre aittir.
    """
    buckets: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, (role, category) in enumerate(zip(row_roles, row_categories)):
        if category in ("fully", "somewhat"):
            buckets[(role, category)].append(index)

    kept = {k: rows for k, rows in buckets.items() if len(rows) >= min_responses}
    if not kept:
        return np.empty((0,) + activations.shape[1:], dtype=np.float32), [], []

    roles_with_both = defaultdict(set)
    for role, category in kept:
        roles_with_both[role].add(category)

    names: list[str] = []
    categories: list[str] = []
    vectors: list[np.ndarray] = []
    for (role, category), rows in sorted(kept.items()):
        name = f"{role}::{category}" if len(roles_with_both[role]) > 1 else role
        names.append(name)
        categories.append(category)
        vectors.append(activations[rows].astype(np.float64).mean(axis=0))

    return np.stack(vectors).astype(np.float32), names, categories


def pca_components(
    vectors: np.ndarray, n_components: int
) -> tuple[np.ndarray, np.ndarray]:
    """Roller arası ortalamayı çıkarıp PCA koş.

    Ortalamayı çıkarmak (centring) isteğe bağlı bir süsleme değil: gerçek
    aktivasyonlarda tüm rollerin paylaştığı büyük bir ortalama vektör vardır ve
    centring olmadan SVD'nin ilk sağ tekil vektörü varyans yönünü değil bu ortak
    ortalamanın yönünü bulur. Regresyon koruması:
    `test_pca_centring_survives_a_large_shared_offset`.

    Dönüş: (bileşenler [k, d], açıklanan varyans oranı [k]).
    """
    centered = np.asarray(vectors, dtype=np.float64)
    centered = centered - centered.mean(axis=0, keepdims=True)
    _u, s, vt = np.linalg.svd(centered, full_matrices=False)
    variance = s**2
    ratios = variance / variance.sum()
    k = min(n_components, vt.shape[0])
    return vt[:k], ratios[:k]


def n_components_for_variance(
    explained_variance_ratio: np.ndarray, threshold: float = 0.70
) -> int | None:
    """Kümülatif varyansın `threshold`'u ilk kez aştığı bileşen sayısı.

    Verilen spektrum eşiğe hiç ulaşmıyorsa `None` döner. Bu ayrım önemli:
    yalnızca ilk 10 bileşen istenmişse `np.searchsorted(cumsum, 0.70)` doyuma
    ulaşıp her zaman 11 der ve gerçek cevap 10'dan büyük olduğunda "persona
    uzayı düşük boyutlu" iddiasını destekleyecek şekilde YANILTICI biçimde
    küçük bir sayı raporlar. `None` dönen çağıran taraf ya tam spektrumla
    yeniden hesaplamalı ya da ">k" gibi bir alt sınır yazmalıdır.
    """
    ratios = np.asarray(explained_variance_ratio, dtype=np.float64)
    if ratios.size == 0:
        return None
    if not np.isfinite(ratios).all():
        raise ValueError("açıklanan varyans oranı sonlu olmayan değer içeriyor")
    reached = np.nonzero(np.cumsum(ratios) >= threshold)[0]
    if reached.size == 0:
        return None
    return int(reached[0]) + 1


def projection_percentile(value: float, distribution: np.ndarray) -> float:
    """`value`'nun dağılım içindeki konumu, 0-1 arası.

    `cosine`/`contrast_axis` ile aynı gerekçe, ama burada YÖN daha kritik:
    sonlu olmayan (NaN/inf) bir `value` kontrolsüz bırakılırsa
    `(dist <= nan).sum()` her zaman `0` verir, yani persentil sessizce
    `0.0`'a çözülür — ve `evaluate_criterion_a(0.9, 0.0)` bunu ALT desilin
    İÇİNDE sayıp `passed: True` üretir. Modülün geri kalanındaki her NaN
    koruması BAŞARISIZLIĞA doğru yanılır; bu satır korumasız kalırsa tam
    tersi yönde, GEÇTİ'ye doğru yanılırdı — ön kaydedilmiş bir kriter için
    yanlış yön. Script'te bugün erişilemez (her iki girdi de yukarı akışta
    sonlu olduğu doğrulanır), ama modüldeki son korumasız NaN geçişi
    buydu.
    """
    value = float(value)
    dist = np.asarray(distribution, dtype=np.float64)
    if not math.isfinite(value):
        raise ValueError(
            "persentil sonlu olmayan (NaN/inf) bir `value` için tanımsız — "
            "girdi büyük olasılıkla boş bir dilimin ortalamasından geliyor"
        )
    if not np.isfinite(dist).all():
        raise ValueError(
            "persentil sonlu olmayan (NaN/inf) değer içeren bir dağılımla tanımsız"
        )
    return float((dist <= value).sum() / len(dist))


def evaluate_criterion_a(cos_pc1_axis: float, default_percentile: float) -> dict:
    """Spec Bölüm 7, A kriteri.

    Geçer: orta katmanda |cos(PC1, kontrast vektörü)| > 0.6 VE default
    Assistant projeksiyonu PC1'in en üst (veya en alt) desilinde.

    PC1'in işareti SVD'nin keyfî bir seçimidir; hem kosinüs hem desil
    büyüklük üzerinden değerlendirilir.

    Sonlu olmayan girdi (NaN/inf) SERT BAŞARISIZLIKTIR. NaN ile yapılan her
    karşılaştırma False döndüğü için sessiz bir NaN, hiçbir gerekçe
    eklenmeden `passed: True` üretirdi — yani tanımsız veriden "GEÇTİ".
    """
    cos_value = float(cos_pc1_axis)
    percentile_value = float(default_percentile)
    cos_is_finite = math.isfinite(cos_value)
    percentile_is_finite = math.isfinite(percentile_value)

    magnitude = abs(cos_value)
    in_extreme_decile = percentile_is_finite and (
        percentile_value >= TOP_DECILE or percentile_value <= BOTTOM_DECILE
    )

    reasons = []
    if not cos_is_finite:
        reasons.append(
            f"cos(PC1, eksen) sonlu değil ({cos_value}) — tanımsız değerden karar çıkarılamaz"
        )
    elif magnitude <= COS_THRESHOLD:
        reasons.append(f"|cos| {magnitude:.3f} <= {COS_THRESHOLD}")
    if not percentile_is_finite:
        reasons.append(
            f"default persentili sonlu değil ({percentile_value}) — "
            "tanımsız değerden karar çıkarılamaz"
        )
    elif not in_extreme_decile:
        reasons.append(
            f"default projeksiyonu uç desilde değil (persentil {percentile_value:.3f})"
        )

    return {
        "cos_pc1_axis": cos_value,
        "cos_magnitude": magnitude,
        "default_percentile": percentile_value,
        "passed": not reasons,
        "reason": "; ".join(reasons) if reasons else "her iki koşul da sağlandı",
    }
