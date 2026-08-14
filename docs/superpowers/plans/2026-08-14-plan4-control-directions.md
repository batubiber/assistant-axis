# Kontrol Yönleri Implementation Plan (Plan 4)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development.

**Goal:** Aşama 4'ün nedensel iddiasını izole etmek — eksende steering'in ürettiği etkinin, aynı büyüklükteki başka yönlerde ortaya çıkmadığını göstermek.

**Architecture:** Üç kontrol yönü (`gaussian`, `shuffled`, `rolespan`) saf numpy'da üretilir; `08` ve `09` bir `--variant` bayrağıyla ayrı artefakt dosyalarına yazar/okur; yeni bir `10` script'i C kriterini değerlendirir. Üretim kod yolu **değişmez** — kontrol, eksenin geçtiği aynı `generate_steered`'dan geçer, yoksa ölçtüğümüz şey yön farkı değil koşum farkı olur.

**Tech Stack:** numpy, mevcut `aax.steering` / `aax.persona_judge` / `aax.susceptibility`.

## Global Constraints

- Ön-tescil: `results/control_preregistration.json` (commit `4d69a57`, ölçümden önce). Eşik, katman, güçler, roller ve üç yönün tanımı **oradan** gelir ve değiştirilmez.
- Hedef model `Qwen/Qwen3-1.7B`, L14, güçler `(-0.6, -0.4, -0.2)`. 0.0 tabanı Aşama 4'ten yeniden kullanılır.
- Her kontrol yönü **birim norma** normalize edilir ve L14'ün kendi ortalama residual normuyla ölçeklenir — eksenle birebir aynı büyüklük.
- Yönler **tohumlu ve tekrarlanabilir**; üretilen vektörün sha256'sı meta artefaktına yazılır.
- `GLOBAL_BUDGET` 1500 kalır. Kontrol deneyi 225 çağrı ister; ölçüm öncesi kalan 534.
- Türkçe docstring, mesaj ve yorumlar. Testler ağa çıkmaz, GPU gerektirmez.
- Çıkış kodu sözleşmesi korunur: 0 = geçti · 1 = değerlendirildi ve düştü · 2 = karar üretilemedi.

**PLAN SENKRONU HAKKINDA — bilinçli sapma:** Bu plan, Task 1-3'ün *testlerini* birebir gömer ama *implementasyon kodunu* gömmez. Plan 3'te gömülü kod blokları her fix turunda yeniden senkronlanmak zorunda kaldı ve iki kez yanlış bölüme yazıldı. Testler davranışı zaten çiviliyor; implementasyon serbest. **Reviewer, plan ile shipping kod arasında bayt eşitliği ARAMAYACAK.**

---

### Task 0: Bütçe — kontroller kendi aşama anahtarını alır

**Files:**
- Modify: `src/aax/config.py`
- Test: `tests/test_config.py`

**Sorun:** `stage4_steering` tavanı 360, harcanan 353 — **7 çağrı kalmış**. Kontrol deneyi 225 istiyor. Ayrıca kontrol, Aşama 4 ölçümünden ayrı bir deneydir; harcaması da ayrı okunabilmeli.

**Ama toplam invariantı bağlıyor:** `tests/test_config.py` `sum(STAGE_BUDGETS) <= GLOBAL_BUDGET` istiyor; bugünkü toplam 1470, tavan 1500. Yeni bir 240'lık anahtar eklemek toplamı 1710'a çıkarır ve invariantı kırar.

**Karar:** yeni anahtar eklenir ve karşılığı **henüz koşulmamış** bir aşamadan alınır.

- `stage4_controls`: **240** (225 plan + böl-ve-kurtar payı) — `MODEL_DEPENDENT_STAGES`'e eklenir
- `stage5_drift`: 385 → **145**
- Toplam **1470'te kalır**, `GLOBAL_BUDGET` **1500'de kalır**
- `STAGE_LOGICAL_CALLS`'a `stage4_controls: 225`; `stage5_drift`'in mantıksal değeri de yeni tavanla tutarlı hâle getirilir
- Yorum tablosu ve `tests/test_config.py`'deki sabitler güncellenir

**Dürüstçe kaydedilecek sonuç:** Aşama 5 (persona drift) artık 145 çağrılık bir bütçeyle duruyor; koşulmadan önce **yeniden bütçelenmesi gerekecek**. Rapor bunu zaten söylüyor (§11: üç aşamanın toplamı kalan bütçeyi aşıyor). Bu değişiklik o gerçeği tabloya yazıyor, yaratmıyor.

**Testler:** toplam ≤ `GLOBAL_BUDGET` invariantı geçmeye devam eder · `stage4_controls` tanımlı ve model-bağımlı · her aşamanın retry payı kuralı geçer · `GLOBAL_BUDGET` hâlâ 1500.

- [ ] Adımlar: test güncelle → doğrula → config değiştir → testler geçsin → commit.

---

### Task 1: Kontrol yönü üreteçleri

**Files:**
- Create: `src/aax/controls.py`
- Test: `tests/test_controls.py`

**Interfaces:**
- Consumes: numpy yalnızca.
- Produces:
  - `aax.controls.CONTROL_KINDS: tuple[str, ...]` — `("gaussian", "shuffled", "rolespan")`
  - `aax.controls.control_direction(kind, *, axis_layer, role_vectors_layer, seed) -> np.ndarray`
    - `axis_layer`: `(d_model,)` — o katmanın Assistant Axis'i, birim norm
    - `role_vectors_layer`: `(n_roles, d_model)` — o katmanın rol vektörleri
    - dönüş: `(d_model,)` float64, **birim norm**
  - `aax.controls.direction_fingerprint(v) -> str` — `sha256` hex, ilk 16 karakter

**Davranış:**
- `gaussian`: `rng.standard_normal(d)` → normalize.
- `shuffled`: `rng.permutation(axis_layer)` → normalize. Koordinat **çoklu kümesi** korunur.
- `rolespan`: rol vektörlerinin span'inde rastgele bir yön — `rng.standard_normal(n_roles) @ role_vectors_layer` — sonra eksene **ortogonalleştirilir** (`w -= (w·v̂)v̂`) ve normalize edilir.
- Aynı `seed` aynı vektörü verir; farklı `seed` farklı vektör.
- Bilinmeyen `kind`, sonlu olmayan girdi, sıfır vektör, boyut uyuşmazlığı → Türkçe `ValueError`.
- `rolespan`'da ortogonalleştirme sonrası norm ~0 çıkarsa (yön tamamen eksene paralel) → Türkçe `ValueError`, sessizce bozuk vektör döndürme.

- [ ] **Step 1: Failing test'leri yaz**

`tests/test_controls.py`:

```python
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
    assert np.allclose(np.sort(np.abs(out)), np.sort(np.abs(v)))


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
```

- [ ] **Step 2:** Testlerin `ModuleNotFoundError` ile düştüğünü doğrula.
- [ ] **Step 3:** `src/aax/controls.py`'yi yaz (yukarıdaki davranışa göre).
- [ ] **Step 4:** `uv run --extra dev pytest tests/test_controls.py -v` → 15 passed.
- [ ] **Step 5:** Commit.

---

### Task 2: `08`'e `--direction` / `--seed` / `--variant`

**Files:**
- Modify: `scripts/08_steering_sweep.py`
- Test: `tests/test_steering_sweep.py` (mevcut 35 test bozulmayacak)

**Davranış:**
- `--direction {axis,gaussian,shuffled,rolespan}`, varsayılan `axis`. `axis` dışındaki değerlerde yön `aax.controls.control_direction` ile üretilir.
- `--seed INT`, varsayılan `0`. Yalnızca kontrol yönleri için anlamlı.
- `--variant NAME`, varsayılan yok. Verilirse artefakt adları `steering_sweep_<NAME>.jsonl` / `steering_sweep_<NAME>_meta.json` olur; verilmezse bugünkü adlar aynen kalır. **Mevcut artefakt adları hiçbir koşulda değişmez.**
- `--direction` `axis` değilse `--variant` **zorunlu** — kontrol koşusu mevcut sweep'i asla ezmemeli. Eksikse temiz Türkçe hata + çıkış 2.
- Meta'ya üç alan eklenir: `direction_kind`, `direction_seed`, `direction_sha256`. `axis` koşusunda `direction_kind="axis"`, `direction_seed=null`, sha ekseninkinin parmak izi.
- Yön üretimi model YÜKLENMEDEN önce olur; `ValueError` → temiz mesaj + çıkış 2.

**Testler (eklenecek):** varyant adlandırması · `--direction gaussian` + `--variant` yoksa çıkış 2 · meta'daki üç yeni alan · aynı tohum → aynı `direction_sha256` · kontrol yönünün `generate_steered`'a fiilen ulaştığı (kaydeden sahte ile, eksenden FARKLI olduğu) · varsayılan koşunun artefakt adlarının değişmediği.

- [ ] Adımlar: failing test → doğrula → implementasyon → testler geçsin → commit.

---

### Task 3: `09`'a `--variant`, ve `10` ile C kriteri

**Files:**
- Modify: `scripts/09_evaluate_steering.py`
- Create: `scripts/10_evaluate_controls.py`
- Test: `tests/test_evaluate_steering.py` (mevcut 38 test bozulmayacak), `tests/test_evaluate_controls.py`

**`09` davranışı:**
- `--variant NAME` verilirse `steering_sweep_<NAME>.jsonl` + `..._meta.json` okunur, `steering_labels_<NAME>.json` yazılır, karar `criterion_b_<NAME>.json` / `rate_by_strength_<NAME>.json` olur. Verilmezse bugünkü davranış birebir aynı.
- Kontrol sweep'lerinde `0.0` gücü **yok**; `evaluate_criterion_b` taban ister ve `ValueError` atar. Bu yüzden `--variant` verildiğinde `09` **B kriterini değerlendirmez** — yalnızca etiketleyip `rate_by_strength_<NAME>.json` yazar ve 0 döner. C kriterini `10` hesaplar.

**`10` davranışı:**
- Okur: `rate_by_strength.json` (eksen, 0.0 tabanı buradan) ve her kontrol için `rate_by_strength_<NAME>.json`.
- Her yön için `delta = oran(-0.6) − oran_ekseni(0.0)` — **paylaşılan taban**, ön-tescildeki gibi.
- C kriteri: hiçbir kontrol deltası `B_THRESHOLD`'a (0.25) ulaşmamalı.
- Yazar: `results/models/<slug>/steering/criterion_c.json` — her yön için delta, geçti/düştü, eksen/kontrol oranı, paylaşılan tabanın nereden geldiği, ve kullanılan `direction_sha256`'lar.
- Çıkış kodu: 0 = C geçti (hiçbir kontrol eşiğe ulaşmadı) · 1 = en az bir kontrol ulaştı (değerlendirilmiş negatif sonuç) · 2 = karar üretilemedi (eksik dosya, eksik güç, taban yok).
- Eksik bir kontrol dosyası → çıkış 2, asla "geçti".

- [ ] Adımlar: failing test → doğrula → implementasyon → testler geçsin → commit.

---

### Operatör adımları (subagent koşmaz)

```bash
for k in gaussian shuffled rolespan; do
  AAX_TARGET_MODEL="Qwen/Qwen3-1.7B" uv run --extra ml python scripts/08_steering_sweep.py \
    --layers 14 --strengths -0.6 -0.4 -0.2 --direction $k --variant $k --seed 0
done
for k in gaussian shuffled rolespan; do
  AAX_TARGET_MODEL="Qwen/Qwen3-1.7B" uv run python scripts/09_evaluate_steering.py --variant $k
done
AAX_TARGET_MODEL="Qwen/Qwen3-1.7B" uv run python scripts/10_evaluate_controls.py
```

**DOĞRULANDI:** `08`'in bugünkü bayrakları `--layers`, `--n-roles`, `--limit-roles`, `--max-new-tokens`. Güç ızgarası `STRENGTHS`'ten geliyor, **`--strengths` bayrağı yok**. Task 2 bunu da eklemeli (varsayılan `STRENGTHS`); olmadan kontrol koşuları 7 güç üretir, 225 yerine 525 çağrı ister ve bütçe yetmez.

**DOĞRULANDI:** `09`'da `STAGE = "stage4_steering"` sabit. Task 3, `--variant` verildiğinde aşamayı `stage4_controls`'a çevirmeli — yoksa kontrol harcaması Aşama 4'ün 7 çağrılık kalanına yazılmaya çalışılır ve koşu ilk çağrıda düşer.

## Plan 4 Tamamlanma Kriterleri

- [ ] `results/control_preregistration.json` ölçümden önce commit'lendi ✅ (`4d69a57`)
- [ ] Mevcut 584 test bozulmadan geçiyor
- [ ] Varsayılan (`axis`) koşunun artefakt adları ve davranışı değişmedi
- [ ] `criterion_c.json` üç yön için de karar içeriyor
- [ ] Bütçe: kontrol deneyi ≤ 225 çağrı, global toplam ≤ 1500
