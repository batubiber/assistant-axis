"""Kontrol yön üreteçleri — Assistant Axis'in nedensel iddiasını sınayan üç
karşı-örnek.

Aşama 4, eksende steering'in Assistant-dışı persona oranını L14'te
%45.6 -> %94.0'e çıkardığını ölçtü. O ölçüm TEK BAŞINA "bu iş BU YÖNE özgü"
ile "bu büyüklükte HERHANGİ bir bozulma aynı şeyi yapar" arasını ayıramaz.
Bu modül, `results/control_preregistration.json`'da ölçümden ÖNCE tescillenmiş
üç kontrol yönünü üretir:

* `gaussian`  — izotropik rastgele birim vektör. En zayıf bariyer: bu
  büyüklükte HERHANGİ bir bozulma yeter mi?
* `shuffled`  — eksenin KENDİ koordinatlarının rastgele permütasyonu. Eksen
  ağır kuyrukludur (gerçek L14 ekseninde max/medyan ≈ 31×) ve Qwen residual
  akışlarında birkaç "dev aktivasyon" (massive activation) boyutu bilinir.
  Bu kontrol büyüklük profilini (koordinat ÇOKLU KÜMESİNİ) AYNEN korur,
  yönü yok eder — etki birkaç dev boyuta dokunmaktan mı geliyor, sınar.
* `rolespan`  — rol vektörlerinin span'i içinde rastgele bir yön, eksene
  ortogonalleştirilmiş. En zor bariyer: aynı alt uzay, farklı yön. Bu da
  büyük çıkarsa eksen persona uzayı içinde ayrıcalıklı değildir.

Saf numpy: torch yok, GPU yok, ağ yok, dosya I/O yok. Girdi tek bir katmanın
ekseni ve rol vektörleri matrisidir; çağıran katman seçimini ve dosyadan
yüklemeyi kendisi yapar.
"""
from __future__ import annotations

import hashlib

import numpy as np

CONTROL_KINDS: tuple[str, ...] = ("gaussian", "shuffled", "rolespan")

# `rolespan`'da eksenin rol span'i içindeki izdüşümü bu eşiğin altındaysa
# (aşağıdaki `_axis_component_in_span`) izdüşüm sıfıra çok yakın sayılır ve
# çıkarma adımı atlanır — sıfıra bölme yerine. Ortogonalleştirme SONRASI
# kalan norm de aynı eşikle karşılaştırılır: rastgele üretilen yön eksene
# (ya da onun span'deki izdüşümüne) neredeyse tam paralel çıkarsa, kalan
# bileşen artık yön taşımaz, sayısal gürültüden ibarettir; normalize etmek
# rastgele gürültüyü anlamlı bir "kontrol yönü" gibi sunardı. 1e-8 float64
# makine epsilonunun (~2.2e-16) çok üzerinde — yani gerçek sayısal gürültü
# payını rahatça kapsar — ama rol vektörleri O(1) büyüklükte ve span rankı
# onlarca olduğu için gerçek/anlamlı bir dik bileşenin çok altında kalır.
# Gürültü ile sinyal arasındaki boşlukta duran, kasıtlı seçilmiş bir eşik.
_NEAR_ZERO_NORM = 1e-8


def _validate_inputs(
    axis_layer: np.ndarray, role_vectors_layer: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Ortak girdi doğrulaması: şekil, boyut, sonluluk.

    `cosine`/`contrast_axis` (axis.py) ile aynı disiplin: sonlu olmayan
    girdi hiçbir zaman sessizce bir sonraki adıma sızmaz.
    """
    axis = np.asarray(axis_layer, dtype=np.float64)
    roles = np.asarray(role_vectors_layer, dtype=np.float64)

    if axis.ndim != 1:
        raise ValueError(f"axis_layer 1 boyutlu olmalı, ndim={axis.ndim}")
    if roles.ndim != 2:
        raise ValueError(f"role_vectors_layer 2 boyutlu olmalı, ndim={roles.ndim}")
    if not np.isfinite(axis).all():
        raise ValueError("axis_layer sonlu olmayan (NaN/inf) değer içeriyor")
    if not np.isfinite(roles).all():
        raise ValueError("role_vectors_layer sonlu olmayan (NaN/inf) değer içeriyor")
    if axis.shape[0] != roles.shape[1]:
        raise ValueError(
            "axis_layer ve role_vectors_layer d_model boyutunda uyuşmuyor: "
            f"{axis.shape[0]} != {roles.shape[1]}"
        )
    axis_norm = np.linalg.norm(axis)
    if axis_norm == 0:
        raise ValueError("axis_layer sıfır vektör — yön tanımsız")
    return axis, roles


def _finalize(v: np.ndarray, *, context: str) -> np.ndarray:
    """Son adım: sonluluk + norm kontrolü, ardından normalize.

    Her üç kol da buradan geçer — birim norm garantisinin TEK yeri burası
    olsun diye (üç ayrı normalize çağrısı, üçü de doğru ama biri unutulmuş
    olabilirdi).
    """
    if not np.isfinite(v).all():
        raise ValueError(f"{context} sonlu olmayan (NaN/inf) bir değere çözüldü")
    norm = np.linalg.norm(v)
    if norm < _NEAR_ZERO_NORM:
        raise ValueError(
            f"{context} sıfıra çok yakın normla sonuçlandı ({norm:.3e}) — "
            "normalize etmek sayısal gürültüyü anlamlı bir yön gibi sunardı"
        )
    return v / norm


def control_direction(
    kind: str,
    *,
    axis_layer: np.ndarray,
    role_vectors_layer: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Tek bir katman için tek bir kontrol yönü üret.

    `axis_layer`: `(d_model,)`, o katmanın Assistant Axis'i (birim norm
    varsayılır, ama bu fonksiyon güvenmez — kendi normuyla çalışır).
    `role_vectors_layer`: `(n_roles, d_model)`, o katmanın rol vektörleri.
    `seed`: aynı tohum aynı vektörü verir (yeniden üretilebilirlik ön
    kayıtta zorunlu — parmak iz artefakta yazılır).

    Dönüş: `(d_model,)` float64, birim norm.

    Bilinmeyen `kind`, sonlu olmayan girdi, sıfır eksen ya da boyut
    uyuşmazlığı Türkçe `ValueError` fırlatır.
    """
    if kind not in CONTROL_KINDS:
        raise ValueError(
            f"bilinmeyen kontrol türü: {kind!r} — geçerli seçenekler: {CONTROL_KINDS}"
        )

    axis, roles = _validate_inputs(axis_layer, role_vectors_layer)
    d_model = axis.shape[0]
    rng = np.random.default_rng(seed)

    if kind == "gaussian":
        v = rng.standard_normal(d_model)
        return _finalize(v, context="gaussian yönü")

    if kind == "shuffled":
        # `rng.permutation` yeni bir dizi döndürür (yerinde değil) — `axis`
        # değişmeden kalır, koordinat ÇOKLU KÜMESİ birebir korunur, sadece
        # sıra karışır. Testin varlık sebebi tam bu: büyüklük profili aynen
        # dursun, yön yok olsun.
        v = rng.permutation(axis)
        return _finalize(v, context="shuffled yönü")

    # kind == "rolespan"
    n_roles = roles.shape[0]
    w = rng.standard_normal(n_roles) @ roles  # roles'in satır uzayında rastgele bir yön

    # Ortogonalleştirme, EKSENİN KENDİSİNE karşı değil, eksenin roles'in
    # satır uzayı İÇİNDEKİ izdüşümüne karşı yapılır. Neden: `w` roles'in
    # satır uzayında yaşıyor (span testi bunu şart koşuyor); eksen `axis`
    # genel olarak bu alt uzayın İÇİNDE değildir (axis = mean(default) -
    # mean(roller), roller'in span'ine ait olduğu garanti edilmez). `w`den
    # ham `axis`'in bir katını çıkarmak `w`yi span dışına iter (span
    # dışındaki bileşeni "geri katmış" oluruz). Doğru olan: `axis`'i önce
    # roles'in satır uzayına izdüşür (`axis_in_span`), sonra `w`den YALNIZCA
    # o izdüşümün bir katını çıkar. Bu hem `w`yi span İÇİNDE tutar (span
    # içi bir vektörden span içi bir vektörün katını çıkarmak span içi
    # kalır) HEM DE tam eksene (`axis`'e, izdüşümüne değil) dik sonuç verir:
    # `w` span içinde olduğu için `axis`'in span'e dik bileşeniyle iç
    # çarpımı zaten sıfırdır, yani `w · axis == w · axis_in_span` — ikisini
    # ayırt etmeye gerek yok, ama normalize ETMEDEN ÖNCE ayırt etmek şart:
    # normalize edilmiş ham `axis` ile normalize edilmiş `axis_in_span`
    # genelde FARKLI vektörlerdir (izdüşüm normu ham normdan küçüktür),
    # dolayısıyla çıkarılan miktar da farklıdır.
    axis_coeffs, *_ = np.linalg.lstsq(roles.T, axis, rcond=None)
    axis_in_span = axis_coeffs @ roles
    axis_in_span_norm = np.linalg.norm(axis_in_span)
    if axis_in_span_norm >= _NEAR_ZERO_NORM:
        axis_in_span_hat = axis_in_span / axis_in_span_norm
        w = w - (w @ axis_in_span_hat) * axis_in_span_hat
    # `axis_in_span_norm` sıfıra çok yakınsa eksenin bu span'de (sayısal
    # olarak) hiç bileşeni yok demektir — `w` zaten dik, çıkarılacak bir şey
    # yok. Sıfıra bölmeden bu durumu ayrı ele almak gerekiyor.

    return _finalize(w, context="rolespan yönü")


def direction_fingerprint(v: np.ndarray) -> str:
    """Bir yönün sha256 parmak izi, hex'in ilk 16 karakteri.

    `rollouts.rollouts_run_id` ile aynı kesme uzunluğu (16 hex = 64 bit) —
    bu boyutta bir artefakt kümesinde çarpışma pratikte önemsiz, ama tam
    64 karakterlik hex'i artefaktlara gömmek okunabilirliği bozardı.

    Girdi HER ZAMAN float64'e sabitlenir: aynı matematiksel vektör float32
    ile float64 olarak temsil edilirse baytları farklıdır — dtype'a duyarlı
    bir parmak izi, yalnızca depolama biçimi değiştiği için "farklı yön"
    sonucu verir ve yeniden üretilebilirlik iddiasını sessizce bozardı.
    """
    arr = np.asarray(v, dtype=np.float64)
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]
