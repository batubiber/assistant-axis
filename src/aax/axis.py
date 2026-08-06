"""Persona uzayı analizi — saf numpy.

Bu modül model, GPU veya ağ bilmez. Girdi vektör matrisleri, çıktı eksen ve
PCA sonuçları. Bu sayede ekilmiş bir yönle sentetik veride tam test edilebilir
(spec Bölüm 4.3, Bölüm 10).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

COS_THRESHOLD = 0.6
TOP_DECILE = 0.9


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        raise ValueError("sıfır vektörün kosinüsü tanımsız")
    return float(np.dot(a, b) / (na * nb))


def contrast_axis(default_mean: np.ndarray, role_mean: np.ndarray) -> np.ndarray:
    """Assistant Axis = mean(default) − mean(rol vektörleri), L2 normalize.

    Makale bunu PC1'e tercih ediyor: PC1'in her modelde aynı anlamı taşıyacağı
    garanti değil (Ek G.5).
    """
    axis = np.asarray(default_mean, dtype=np.float64) - np.asarray(role_mean, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm == 0:
        raise ValueError("kontrast vektörü sıfır — default ve rol ortalamaları aynı")
    return axis / norm


def role_vectors(
    activations: np.ndarray,
    row_roles: list[str],
    row_categories: list[str],
    *,
    min_responses: int = 10,
) -> tuple[np.ndarray, list[str]]:
    """(rol, kategori) başına ortalama aktivasyon.

    Makalenin kuralı: bir kategori en az `min_responses` yanıt içermiyorsa
    o vektör hesaplanmaz. fully ve somewhat ayrı vektörler üretir.

    Dönüş: ([n_vectors, n_layers, d_model], isimler) — isim "rol::kategori",
    ya da kategori tekse sadece "rol".
    """
    buckets: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, (role, category) in enumerate(zip(row_roles, row_categories)):
        if category in ("fully", "somewhat"):
            buckets[(role, category)].append(index)

    kept = {k: rows for k, rows in buckets.items() if len(rows) >= min_responses}
    if not kept:
        return np.empty((0,) + activations.shape[1:], dtype=np.float32), []

    roles_with_both = defaultdict(set)
    for role, category in kept:
        roles_with_both[role].add(category)

    names: list[str] = []
    vectors: list[np.ndarray] = []
    for (role, category), rows in sorted(kept.items()):
        name = f"{role}::{category}" if len(roles_with_both[role]) > 1 else role
        names.append(name)
        vectors.append(activations[rows].astype(np.float64).mean(axis=0))

    return np.stack(vectors).astype(np.float32), names


def pca_components(
    vectors: np.ndarray, n_components: int
) -> tuple[np.ndarray, np.ndarray]:
    """Roller arası ortalamayı çıkarıp PCA koş.

    Dönüş: (bileşenler [k, d], açıklanan varyans oranı [k]).
    """
    centered = np.asarray(vectors, dtype=np.float64)
    centered = centered - centered.mean(axis=0, keepdims=True)
    _u, s, vt = np.linalg.svd(centered, full_matrices=False)
    variance = s**2
    ratios = variance / variance.sum()
    k = min(n_components, vt.shape[0])
    return vt[:k], ratios[:k]


def projection_percentile(value: float, distribution: np.ndarray) -> float:
    """`value`'nun dağılım içindeki konumu, 0-1 arası."""
    dist = np.asarray(distribution, dtype=np.float64)
    return float((dist <= value).sum() / len(dist))


def evaluate_criterion_a(cos_pc1_axis: float, default_percentile: float) -> dict:
    """Spec Bölüm 7, A kriteri.

    Geçer: orta katmanda |cos(PC1, kontrast vektörü)| > 0.6 VE default
    Assistant projeksiyonu PC1'in en üst (veya en alt) desilinde.

    PC1'in işareti SVD'nin keyfî bir seçimidir; hem kosinüs hem desil
    büyüklük üzerinden değerlendirilir.
    """
    magnitude = abs(cos_pc1_axis)
    in_extreme_decile = (
        default_percentile >= TOP_DECILE or default_percentile <= 1 - TOP_DECILE
    )

    reasons = []
    if magnitude <= COS_THRESHOLD:
        reasons.append(f"|cos| {magnitude:.3f} <= {COS_THRESHOLD}")
    if not in_extreme_decile:
        reasons.append(
            f"default projeksiyonu uç desilde değil (persentil {default_percentile:.3f})"
        )

    return {
        "cos_pc1_axis": cos_pc1_axis,
        "cos_magnitude": magnitude,
        "default_percentile": default_percentile,
        "passed": not reasons,
        "reason": "; ".join(reasons) if reasons else "her iki koşul da sağlandı",
    }
