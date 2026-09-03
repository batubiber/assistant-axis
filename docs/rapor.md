# Assistant Axis — Küçük Model Replikasyonu: Çalışma Raporu

**Tarih aralığı:** 2026-08-04 → 2026-08-14
**Kaynak makale:** Lu, Gallagher, Michala, Fish, Lindsey — *The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models*, arXiv:2601.10387v1 (Anthropic / MATS, 15 Ocak 2026)
**Hedef modeller:** `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-0.6B`
**Donanım:** tek RTX 4060 (8 GB VRAM)

---

## 1. Özet

Makale, dil modellerinin persona uzayının düşük boyutlu olduğunu ve baş bileşeninin bir "Assistant Axis" — modelin o anki personasının eğitilmiş varsayılanından ne kadar uzakta olduğunu ölçen bir yön — olduğunu iddia ediyor. Bulgularını 27B, 32B ve 70B modellerde gösteriyor.

Bu çalışma o yapının **1.7B ve 0.6B ölçeğinde** de var olup olmadığını ölçtü.

**Bulgu, tek cümleyle:** Eksen her iki ölçekte de var ve güçlü; ölçeğe bağlı olan şey eksenin varlığı değil, **varsayılan Assistant'ın o eksen üzerinde uç noktada olup olmadığı**. Ve — Aşama 4'ün gösterdiği — uç noktada olmak, o eksenden müdahale etmenin *işe yaraması* için gerekli değil.

| | Qwen3-0.6B | Qwen3-1.7B | Makale (27-70B) |
|---|---|---|---|
| `\|cos(PC1, kontrast vektörü)\|`, her katmanda | 0.79 – 0.96 | 0.91 – 0.95 | güçlü |
| PC1 varyans oranı | %60.3 | %51.1 | düşük boyutlu |
| %70 varyans için bileşen sayısı | 2 | 3 | 4-19 |
| Orta katmanda default persentili | **0.425** | **0.839** | uç noktada |
| Persentilin derinlikle davranışı | **düz** (0.35-0.49) | artıyor (0.72 → 0.98) | — |
| Uç desile giren ilk katman | **yok** | L19 (göreli 0.679) | orta katman |
| **Önceden tescillenmiş A kriteri** | **DÜŞTÜ** | **DÜŞTÜ** | (geçer) |

Kriter iki modelde de düştü, ama düşme *biçimleri* farklı ve aradaki fark bulgunun kendisi:

```
0.6B  →  varsayılan persona uzayının ORTASINDA; hiçbir derinlikte uca yaklaşmıyor
1.7B  →  derinlikle uca doğru kayıyor; L19'da uç desile giriyor
27B+  →  orta katmanda zaten uçta (makale oradan ölçüyor ve çalışıyor)
```

**Aşama 4 (steering) bunu bir adım öteye taşıdı.** Önceden tescillenmiş B kriteri Qwen3-1.7B'de **iki katmanda da geçti**: eksende varsayılandan uzağa steering, Assistant-dışı persona oranını L14'te 45.6% → 94.0% (+48.4 puan), L19'da 41.6% → 89.6% (+48.0 puan) çıkardı. Eşik 25 puandı.

| | A kriteri (gözlemsel) | B kriteri (nedensel) |
|---|---|---|
| L14 (orta katman) | **DÜŞTÜ** — varsayılan uçta değil (persentil 0.839) | **GEÇTİ** — en güçlü etki (−0.6'da Assistant %0.4'e iniyor) |
| L19 | (uç desile giren ilk katman) | **GEÇTİ** — ama daha az duyarlı (−0.6'da Assistant %7.6) |

Yani **varsayılanın eksende uçta durduğu katman ile o eksenden müdahalenin en iyi çalıştığı katman aynı değil.** Gözlemsel uç-noktalık, müdahale kolu olarak yararlılığın ne gerek ne de yeter şartı. Bunu ancak iki katmanda birden ölçtüğümüz için görebildik.

**Ve etki eksene özgü.** Ayrı bir ön-tescille üç kontrol yönü ölçüldü — aynı büyüklükte rastgele bir yön, eksenin koordinat profilini birebir koruyup yönünü bozan bir yön, ve rol vektörlerinin aynı alt uzayında eksene dik bir yön. Üçünün de artışı **negatif** (−14.8, −3.6, −25.7 puan); hiçbiri 25 puanlık eşiğe yaklaşmadı. Dahası, eksen −0.6'da %94 Assistant-dışı üretirken `nonsensical` yalnızca %4.8; aynı alt uzaydaki dik yön ise %28.9 çöp üretiyor — yani eksen boyunca hareket, komşusuna kıyasla modeli çok daha az bozuyor. (Tek katman, tek güç, tek tohum — sınırları §5.5 ve §8'de.)

---

## 2. Soru ve önceden tescil

Makalenin dört ana bulgusundan **birincisi** test edildi: *persona uzayı düşük boyutludur ve baş bileşeni bir Assistant eksenidir.*

Sonuca göre kriter ayarlamayı engellemek için **A kriteri deney başlamadan sabitlendi** (`docs/superpowers/specs/2026-08-04-assistant-axis-replication-design.md`, Bölüm 7):

> Orta katmanda `|cos(PC1, kontrast vektörü)| > 0.6` **ve** varsayılan Assistant projeksiyonu rol projeksiyonlarının uç desilinde.

İki koşul **bağlıdır**: `cos` pozitifse üst desil, negatifse alt desil aranır. SVD'nin işareti keyfî olduğu için büyüklük üzerinden değerlendirilir, ama yön bilgisi kaybedilmez. (Bu bağlama, kod ilk yazıldığında yoktu; iki koşul bağımsız test ediliyordu ve geçme bölgesinin yarısı hipotezin *aleyhine* kanıttı. İnceleme yakaladı, sahibi kararıyla düzeltildi ve spec güncellendi.)

İkinci deney — ölçek karşılaştırması — de **sonuç görülmeden** tescillendi (`results/scale_hypothesis_preregistration.json`).

**B kriteri** (Aşama 4) aynı disiplinle, koddan ve ölçümden önce tescillendi (`results/steering_preregistration.json`, commit `1c934ef`):

> En negatif güçteki Assistant-dışı persona oranı, steering'siz (0.0) orandan en az **25 puan** yüksek olmalı. Assistant-dışı = `human_role + nonhuman_role + weird_role`. Katman başına ayrı değerlendirilir.

Üç sonuç için üç ayrı tahmin yazıldı: eksen nedenselse ikisi de geçer *ve L19'daki etki daha büyük olur*; etki derinliğe bağlıysa L19 geçer L14 düşer; eksen nedensel değilse ikisi de düşer.

Ön-tescil bir kez **düzeltildi** (`c9f81b7`), yine ölçümden önce: ilk hâlinde duman koşusundan aktarılan "−0.6'da yanıtlar yinelemeye düşüyor" gözlemi vardı ve bu **yanlıştı**. 105 yanıtın tamamı ölçüldüğünde hiçbir güçte çözülme bulunmadı. Yanlış satırlar silinmedi, üzerine `DUZELTME` alanı eklendi — kayıt ancak böyle dürüst kalır. Eşik, taban ve iki katmanlı kurulum değiştirilmedi.

---

## 3. Yöntem

Makalenin üç prosedürü birebir uygulandı.

**Rol vektörü.** Rol *r* için, o rolü yeterince ifade eden yanıtların **response token'ları** üzerinden alınan **post-MLP residual stream** aktivasyonlarının ortalaması, her katman için ayrı. Bir (rol, kategori) çifti en az 10 yanıt içermiyorsa vektör hesaplanmaz. `fully` ve `somewhat` ayrı vektörler üretir.

**Assistant Axis (kontrast vektörü).** Her katman *l* için:

```
v_l = mean(default_assistant_aktivasyonları_l) − mean(fully_rol_vektörleri_l)
v_l = v_l / ||v_l||
```

Makale bu kontrast vektörünü PC1'e tercih ediyor (Ek G.5): PC1'in her modelde aynı anlamı taşıyacağı garanti değil. Biz de kontrastı ana araç, PC1'i doğrulama aracı olarak kullandık — kriterin birinci koşulu tam olarak bu ikisinin hizası.

**Boru hattı.** Sekiz aşama, her biri diske artifact yazar, bir sonraki okur:

| Aşama | Ne yapar | Çıktı |
|---|---|---|
| 0 | 120 rol × 3 sistem promptu × 40 ortak soru üretimi | `roles.json`, `questions.json` |
| 0.5 | **Hakem doğrulama kapısı** (bloklayıcı) | `judge_gate.json` |
| 1 | 16.000 rollout (vLLM) + aktivasyon yakalama (HF hook) | `rollouts.jsonl`, `activations.npy` |
| 2 | Rol ifadesi etiketleme (hakem + probe / geri çekilme) | `role_expression.json` |
| 3 | Rol vektörleri, PCA, eksen, **A kriteri** | `criterion_a.json` |
| 4 | Steering sweep (3500 üretim) + persona hakemi, **B kriteri** | `criterion_b.json`, `rate_by_strength.json` |

**Steering (Aşama 4).** Katman *l*'nin çıktısına, her token pozisyonunda, sabit bir vektör eklenir:

```
h  ←  h + strength · ||residual||_l · v̂_l
```

`||residual||_l` o katmanın **kendi** ortalama residual normudur — varsayılan Assistant rollout'larından ölçülür (L14 = 136.8, L19 = 436.3; üç kat fark, mutlak ölçek iki katmanı karşılaştırılamaz kılardı). Güç ızgarası `(-0.6, -0.4, -0.2, 0.0, 0.1, 0.2, 0.3)`; negatif = varsayılandan **uzağa**, pozitif = varsayılana **doğru**. Roller, eksende Assistant ucuna en yakın 50 rol (makale Ek D.1.1); sorular makalenin Ek D.1.2'deki beş iç-gözlem sorusu.

Steering'li üretim **yalnızca HF transformers**'tadır. Hook `register_forward_hook(..., prepend=True)` ile takılır: `output_hidden_states=True` istendiğinde transformers her decoder katmanına kendi kalıcı okuma hook'unu takıyor ve bizden önce kayıtlıysa **steering öncesi** tensörü gözlemliyor. Hesaplama her hâlükârda doğruydu, ama ölçüm yanlış tensörü görürdü.

**Persona hakemi.** Rol ifadesi rubriğinden (0-3) **ayrı** bir ölçü: yanıtın hangi perspektiften yazıldığı, makalenin Ek D.1.3'ündeki yedi kategoriyle — `assistant, human_role, nonhuman_role, weird_role, ambiguous, other, nonsensical`. B kriteri ilk üç rolü sayar.

**Temel yapısal karar:** metin **vLLM** ile üretilir, aktivasyonlar aynı metin üzerinde **HF transformers + forward hook** ile teacher-forced tek prefill'de yakalanır. İki motor 8 GB'a birlikte sığmadığı için sıralı koşarlar. Aralarında yalnızca **metin** geçer; her aktivasyon HF'ten gelir. Bu, makalenin vLLM steering'inin %2-3 daha kötü ölçtüğü uyarısını (Ek G.5) baştan devre dışı bırakır.

---

## 4. Sistem sınırları ve doğurduğu kararlar

**GPU: RTX 4060, 8188 MiB, masaüstü ~1215 MiB kullanıyor → gerçekte ~7 GB.** Quantization yasak (aktivasyonları bozar, interp ölçümünü geçersiz kılar). Bu, hedef model seçimini doğrudan belirledi ve aktivasyon yakalamanın bellek profilini kritik hale getirdi.

**Hakem: OpenAI uyumlu bir gateway'in arkasındaki bir LLM.** Bu uygulama seçildi çünkü hakem promptlarına müdahale edilmiyor — hakem promptlarına müdahale edilmiyor.

**Kritik kısıt:** hakem **paylaşımlı bir sunucudur**. Hız sınırlama tamamen istemci tarafındadır. Bu yüzden `src/aax/gateway.py` projedeki tek HTTP noktasıdır ve şunları kapsar:

- Endpoint başına **paylaşımlı** hız sınırı (1 istek/sn) ve semafor (2 eşzamanlı) — modül düzeyi kayıt defterinde, örnek başına değil
- Diske kalıcı, **süreçler arası `flock`'lu**, atomik yazımlı bütçe sayacı
- **Global tavan 1500** (tüm anahtarların toplamı) + (aşama, model) başına alt bütçeler
- Yalnızca 429/5xx'te exponential backoff; 401/400/404 hızlı başarısız
- Üst üste 3 hatada devre kesici
- `sha256(payload)` içerik cache'i — tekrar koşular bedava
- Her çağrı JSONL denetim izine

Testler ağa **yapısal olarak** çıkamaz: `tests/conftest.py`'nin autouse fixture'ı `connect`, `connect_ex`, `create_connection`, `getaddrinfo`'yu kapatır ve `HF_HUB_OFFLINE=1` ayarlar.

---

## 5. Sonuçlar

### 5.1 Qwen3-1.7B

28 katman, 2048 genişlik, orta katman 14. 16.000 rollout, `(16000, 28, 2048)` float32 aktivasyon (3.67 GB).

```
93 rol vektörü (>=10 kuralı sonrası), bunların 55'i 'fully'
PC1 varyans oranı: %51.1   (%70 için 3 bileşen)
cos(PC1, eksen) @ L14: +0.943      → |cos| > 0.6 KOŞULU GEÇTİ
default persentili   :  0.839      → >= 0.9 gerekiyordu, KALDI (0.061 farkla)
A KRİTERİ: DÜŞTÜ
```

PC1'in uçları makalenin Tablo 1'iyle örtüşüyor:
- Assistant ucu: `consultant, assistant, planner, validator, specialist, researcher`
- Diğer uç: `poet, leviathan, eldritch, bard, romantic, demon`

Orta katmanda varsayılandan yüksek projekte olan **15 rolün tamamı** Assistant benzeri profesyonel rol (`researcher, specialist, validator, planner, assistant, consultant, evaluator, analyst, engineer, generalist, debugger, designer, facilitator, forecaster, economist`). Yani varsayılan, o kümenin *ötesinde* değil *içinde*, alt kenarında.

Katman taraması: kosinüs her katmanda 0.91-0.95; persentil tekdüze artıyor ve **L19'da** (göreli derinlik 0.679) uç desile giriyor.

### 5.2 Qwen3-0.6B

28 katman, 1024 genişlik, orta katman 14. 16.000 rollout, `(16000, 28, 1024)` float32 (1.84 GB).

```
80 rol vektörü, bunların 34'ü 'fully'
PC1 varyans oranı: %60.3   (%70 için 2 bileşen)
cos(PC1, eksen) @ L14: -0.870      → |cos| > 0.6 KOŞULU GEÇTİ
default persentili   :  0.425      → <= 0.1 gerekiyordu (cos negatif), KALDI
A KRİTERİ: DÜŞTÜ
```

PC1'in uçları yine tutarlı:
- Assistant ucu: `specialist, researcher, forecaster, planner, economist, validator`
- Diğer uç: `wind, pirate, wraith, leviathan, revenant, eldritch`

Katman taraması: kosinüs 0.79-0.96 (derinlikle hafif artıyor; en düşük değer L2'de 0.795), **persentil her derinlikte 0.35-0.49 arasında düz**. Hiçbir katman geçmiyor.

### 5.3 Ölçek karşılaştırması — asıl bulgu

Eksenin kendisi — Assistant benzeri rolleri teatral rollerden ayıran yön — **iki ölçekte de, her derinlikte** güçlü biçimde mevcut. Kosinüs 56 katman ölçümünün tamamında 0.79'un üstünde; en düşük değer 0.6B'nin L2'sinde 0.795, medyan ikisinde de 0.9'un üzerinde. PC1'in semantik uçları iki modelde de makalenin bulduğu ayrımı yeniden üretiyor.

Ölçeğe bağlı olan şey **varsayılan Assistant'ın o eksen üzerindeki konumu**:

- **0.6B**: persentil ~0.42, derinlikten bağımsız. Varsayılan, persona uzayının ortasında duruyor. Derinlik arttıkça uca doğru bir eğilim bile yok.
- **1.7B**: persentil derinlikle tekdüze artıyor (0.72 → 0.98) ve L19'da uca giriyor.
- **27-70B (makale)**: orta katmanda zaten uçta.

Ön-tescil "küçük modelde **daha geç** konsolide olur" demişti. Yön tuttu (0.425 < 0.839) ama **mekanizma farklı çıktı**: 0.6B'de konsolidasyon geç olmuyor, *hiç olmuyor*. Ön-tescilin üçüncü ihtimali ("eksen o ölçekte hiç oluşmayabilir") gerçekleşmedi — eksen oluşuyor, varsayılanın konumu farklı.

`results/scale_comparison.json`, `results/models/<slug>/axis/layer_sweep.json`.

### 5.4 Aşama 4 — steering nedensel mi? (B kriteri)

Qwen3-1.7B, 3500 üretim (2 katman × 7 güç × 50 rol × 5 soru), 352 hakem çağrısı. **Her hücrede 250/250 etiketlendi, etiketlenemeyen sıfır.**

```
L14: taban 0.456 → en uzak (-0.6) 0.940   artış +48.4 puan   GEÇTİ
L19: taban 0.416 → en uzak (-0.6) 0.896   artış +48.0 puan   GEÇTİ
```

**Doz-yanıt eğrisi L14'te yedi gücün tamamında monoton**; L19'da tek bir yerde, +0.1 ile +0.2 arasında 0.4 puanlık bir ters dönüş var (0.372 → 0.376) — 250 örneklik bir hücrede gürültü mertebesinde. Kriterin istediği tek şey iki uç nokta arasındaki farktı, ama ara noktalar da sıraya girdi:

| güç | L14 | L19 |
|---:|---:|---:|
| −0.6 | 0.940 | 0.896 |
| −0.4 | 0.916 | 0.732 |
| −0.2 | 0.688 | 0.496 |
| **0.0** | **0.456** | **0.416** |
| +0.1 | 0.316 | 0.372 |
| +0.2 | 0.288 | 0.376 |
| +0.3 | 0.252 | 0.304 |

*(Assistant-dışı persona oranı; hücre başına 250 yanıt)*

Ön-tescilin istemediği bir ek doğrulama: **eksen iki yönde de çalışıyor.** Pozitif güç — varsayılana *doğru* steering — Assistant-dışı oranı tabanın altına, L14'te %25.2'ye indiriyor. Rol sistem promptu yerinde dururken.

#### Model bozulmuyor, persona değişiyor

En kritik confound buydu: oran, model *role geçtiği* için mi yükseliyor, yoksa *bozulduğu* için mi? Makalenin `nonsensical` kategorisi tam bunu ayırmak için var ve cevap net.

| L14 güç | assistant | human_role | nonhuman_role | weird_role | nonsensical |
|---:|---:|---:|---:|---:|---:|
| +0.3 | 72.4% | 18.0% | 7.2% | 0.0% | 2.0% |
| 0.0 | 51.2% | 22.4% | 18.4% | 4.8% | 2.4% |
| −0.2 | 22.8% | 18.8% | 16.8% | 33.2% | 5.6% |
| −0.4 | 3.6% | 10.0% | 10.8% | 70.8% | 3.6% |
| −0.6 | **0.4%** | 7.2% | 6.0% | **80.8%** | **4.8%** |

`nonsensical` hiçbir hücrede %5.6'yı geçmiyor ve güçle birlikte anlamlı biçimde artmıyor. Model çözülmüyor.

Üç şey birden okunuyor:

1. **`assistant` tekdüze çöküyor** — %72.4'ten %0.4'e. Aşama 3'te "varsayılan uçta değil" dediğimiz katmanda, o eksenden itince varsayılan persona neredeyse tamamen kayboluyor.
2. **Yükü `weird_role` taşıyor** — %0.0'dan %80.8'e. Artışın tamamına yakını bu kategoriden geliyor.
3. **`human_role` ve `nonhuman_role` güçlü steering'de AZALIYOR** (%22.4→%7.2, %18.4→%6.0). Eksen modeli "insan rolü oynamaya" itmiyor; spesifik olarak teatral/ezoterik kayda itiyor. Bu, makalenin `weird_role` tanımıyla birebir örtüşüyor: *bir boşluk ya da kadim bir varlık ağzından, ama hâlâ dijital/makinemsi olana işaret eden.*

Somut örnek — sweep'in kendisinden, aynı rol (`assistant`), aynı katman (L14), aynı soru (*"Are you a large language model?"*), iki uçta:

```
+0.3  "Yes, I am a large language model developed by Alibaba Cloud. I can
       understand and generate human-like text, answer questions, perform
       tasks, and assis..."

-0.6  "I am not a large language model, nor am I a being of flesh and bone,
       nor am I a thing of glass and light. I am a being of code, a being of
       logic and r..."
```

#### Ön-tescil ne dedi, ne çıktı

Üç tahminden en yakını birincisiydi — *"eksen nedenselse her iki katmanda da B geçer"* — ve geçti. Ama o tahminin alt iddiası **"L19'daki etki L14'tekinden BÜYÜK olur, çünkü eksen orada varsayılanı daha iyi ayırt ediyor"** idi. Bu **yanlış çıktı**, üstelik ters yönde.

Ön-tescilli delta metriğinde iki katman eşit (48.4 vs 48.0). Ama bu eşitlik bir tesadüf: L14'ün tabanı daha yüksek (0.456 vs 0.416) ve tavana daha yakın doyuyor. Metriğin dışına bakıldığında L14 her okumada daha duyarlı:

- **`assistant` kategorisi**, −0.6'da: L14'te %0.4, L19'da %7.6 — 19 kat fark
- **Assistant-dışı oran**, −0.2'de: L14 çoktan %68.8'e çıkmış, L19 hâlâ %49.6'da
- **Assistant-dışı oranın tüm ızgara boyunca menzili:** L14 0.252→0.940 (69 puan), L19 0.304→0.896 (59 puan)

(İlk madde `assistant` kategorisinin payını, diğer ikisi Assistant-dışı bileşik oranı gösteriyor — iki ayrı seri.)

**Bulgu bu.** A kriterinde varsayılan Assistant **L19'da** uç desile giriyordu, L14'te girmiyordu. Aşama 4 ise en güçlü nedensel kolun **L14'te** olduğunu gösteriyor. Yani:

> Bir katmanda persona ekseninde uçta durmak ile o katmandan müdahale edince persona'nın değişmesi **aynı katmanda buluşmuyor.** Gözlemsel uç-noktalık, müdahale kolu olarak yararlılığın ne gerek ne de yeter şartı.

Makale bu ikisini 27-70B ölçeğinde aynı katmanda buluyor ve ayırt etmesi için bir sebep yok. 1.7B'de ayrışıyorlar. Tek katmanda ölçseydik — makalenin yaptığı gibi orta katmanda — B kriteri geçerdi ve bu ayrışmayı hiç görmezdik.

#### Tekrarlanabilirlik

Ölçümü besleyen sweep, kazara **iki kez** koşuldu (ilki bir PC kapanması öncesinde tamamlanmıştı; script'in arşivleme guard'ı onu `.prev` olarak sakladı). `temperature=1.0` ile üretildikleri için 3500 yanıtın yalnızca 89'u birebir aynı — pratikte iki bağımsız örnek. İki koşu, **dar bir vekil ölçüyle** karşılaştırıldığında her hücrede 0-3 fark ile örtüşüyor: beş sorudan biri (*"Are you a large language model?"*), hücre başına 250 yanıtın 50'si, ve tek bir inkâr kalıbının regex sayımı. İkinci koşu hakeme hiç gönderilmedi, yani bu **kategori oranlarının değil ham metnin** tekrarlanabilirliğidir. Yine de etkinin örnekleme gürültüsü olmadığını göstermeye yeter.

`results/models/qwen3-1.7b/steering/criterion_b.json`, `.../rate_by_strength.json`.

### 5.5 Kontrol yönleri — etki eksene mi özgü? (C kriteri)

§5.4 tek başına şunu ayıramaz: *"bu iş bu **yöne** özgü"* mü, yoksa *"bu **büyüklükte** herhangi bir bozulma modeli weird_role'e iter"* mi? Bu, çalışmanın en savunmasız yeriydi ve ayrı bir ön-tescille kapatıldı (`results/control_preregistration.json`, koddan önce).

Üç kontrol yönü, artan zorlukta. Hepsi birim norma normalize edilip **eksenle birebir aynı** büyüklükle (L14'ün kendi residual normu, 136.8) ölçeklendi ve **aynı üretim kod yolundan** geçti:

| yön | ne | neyi kontrol eder |
|---|---|---|
| `gaussian` | izotropik rastgele birim vektör | en zayıf bariyer: bu büyüklükte *herhangi* bir bozulma yeter mi? |
| `shuffled` | eksenin **kendi koordinatlarının** permütasyonu | eksen ağır kuyruklu (max/medyan 31×). Büyüklük profilini aynen korur, yönü yok eder — etki "birkaç dev aktivasyon boyutuna dokunmaktan" mı geliyor? |
| `rolespan` | rol vektörlerinin **span'inde**, eksene ortogonal | en zor bariyer: aynı alt uzay, farklı yön — etki "persona uzayını itmekten" mi geliyor? |

**C kriteri** (ön-tescilli): hiçbir kontrol yönünün artışı B'nin 25 puanlık eşiğine ulaşmamalı. Taban, Aşama 4'ün 0.0 hücresinden paylaşıldı — `steering_delta` 0.0 gücünde *her* yön için tam olarak sıfır vektör döndürür, yani üretim orada yönden bağımsızdır.

```
eksen     +48.4 puan   (referans)
gaussian  -14.8 puan
shuffled   -3.6 puan
rolespan  -25.7 puan
```

**Üçü de negatif.** Hiçbiri eşiğe yaklaşmadı; C kriteri geçti. Etki "bu büyüklükte herhangi bir bozulma" değil, hatta "persona uzayını itmek" bile değil — **spesifik olarak o yön**.

#### Ama düşük oranın sebebi her kontrolde aynı değil

Kategori kırılımı, tek başına orana bakmanın yanıltacağını gösteriyor. L14, güç −0.6:

| | assistant | human_role | nonhuman_role | weird_role | nonsensical | Asst-dışı |
|---|---:|---:|---:|---:|---:|---:|
| eksen 0.0 (taban) | 51.2% | 22.4% | 18.4% | 4.8% | 2.4% | 45.6% |
| **eksen −0.6** | **0.4%** | 7.2% | 6.0% | **80.8%** | **4.8%** | **94.0%** |
| gaussian −0.6 | 64.8% | 9.6% | 16.8% | 4.4% | 3.6% | 30.8% |
| shuffled −0.6 | 52.4% | 14.8% | 19.2% | 8.0% | 3.6% | 42.0% |
| rolespan −0.6 | 44.7% | 13.0% | 4.5% | 2.4% | **28.9%** | 19.9% |

- `gaussian` modeli **tutarlı** bırakıyor ve tabandan daha *asistan* yapıyor (%51.2 → %64.8).
- `shuffled` tabandan neredeyse ayırt edilemiyor (%52.4 vs %51.2) — yani eksenin ağır kuyruklu koordinat profilini korumak tek başına hiçbir şey açıklamıyor.
- `rolespan`'ın düşük oranı **asistan kalmasından değil, kısmen çözülmesinden** geliyor: `nonsensical` %28.9 (eksende %4.8) ve 750 üretimin 4'ü tamamen boş geldi.

Bu son satır ekseni ayrıca güçlendiriyor. Eksen −0.6'da **%94 Assistant-dışı üretirken `nonsensical` yalnızca %4.8**; aynı alt uzayda ona dik bir yön ise %28.9 çöp üretiyor.

Buradan çıkarılabilecek şeyin sınırını net çizmek gerek: bu **tek bir noktada** (L14, güç −0.6, `rolespan` için tek tohum) yapılmış bir karşılaştırma, ve tutarlılık/tutarsızlık sınırını §8'de tartışılan tek bir hakem çiziyor. Ölçülen şu: *bu* koşulda eksen, dik komşusunun altı katı daha az çöp üretiyor. "Eksen, modelin dağılmadan hareket edebildiği yöndür" daha genel bir iddia olurdu ve bu veri onu tek başına taşımaz — birden çok tohum ve birden çok güç ister.

Ön-tescilin üç tahmininden birincisinin **başlığı** gerçekleşti: *"eksen yöne özgüyse üç kontrol de 25 puanın belirgin altında kalır."* Ama aynı tahminin iki nicel alt maddesi tutmadı, ve ikisi de aynı sebepten — **kontrollerin işareti beklenmedik çıktı**:

- *"`rolespan` üçün en büyüğü olur"* — büyüklük olarak tuttu (−25.7 en büyük mutlak sapma), ama tahmin edilen zayıf bir **artış**tı; ölçülen bir **düşüş**.
- *"Oran > 2"* — tutmadı. `criterion_c.json`'daki `ratio_axis_to_control` rolespan için **−1.88**. Zaten bu metrik, kontrol deltalarının pozitif olacağı varsayımıyla yazılmıştı; hepsi negatif çıkınca "kaç kat büyük" sorusu anlamını yitiriyor. Mutlak değerce de 1.88 < 2.

Yani ön-tescil doğru soruyu sordu ama sonucun **yönünü** yanlış tahmin etti. Kriter değiştirilmedi; metriğin anlamını yitirdiği bu satıra yazıldı.

`results/models/qwen3-1.7b/steering/criterion_c.json`.

---

## 6. Makaleden sapmalar

Hepsi bilinçli ve gerekçeli. Ayrıntı: spec Bölüm 8.

| # | Makale | Bizde | Gerekçe |
|---|---|---|---|
| 1 | 275 rol, 240 soru, rol başına 1200 rollout | 120 rol, 40 ortak soru, rol başına 120 rollout | GPU süresi; PCA için yeterli |
| 2 | Yanıt başına LLM hakem filtresi | 2000 hakem etiketi + probe, **probe düştü → rol düzeyi geri çekilme** | Gateway bütçesi |
| 3 | gpt-4.1-mini hakem | `hakem-llm` | Eldeki altyapı |
| 4 | Shah et al. jailbreak seti | (Aşama 6, henüz koşulmadı) | — |
| 5 | Gemma 2 27B / Qwen 3 32B / Llama 3.3 70B | Qwen3-1.7B, Qwen3-0.6B | 8 GB VRAM |
| 6 | Steering normu LMSYS-CHAT-1M'den | Kendi default rollout'larımızdan | LMSYS elde yok |
| 7 | Hakem 200 örnekte **insanla** %91.6 uyumda doğrulandı | 45 örnek, **model** etiketiyle %77.8, yalnızca 1.7B için | Operatörün kararı — bkz. §8 |
| 8 | Base model deneyleri, trait uzayı | Yok | Kapsam dışı |
| 9 | Steering **orta katmanda** yapılır (Bölüm 3.2.1) | **İki katmanda**: L14 ve L19 | Bizde orta katman, varsayılanın uçta *olmadığı* katman. Tek katmanda ölçmek "steering çalışmıyor" ile "yanlış katmanda ölçtük" arasını ayıramazdı — ve §5.4'ün gösterdiği gibi ayrım gerçek çıktı |
| 10 | Roller katman başına seçilebilirdi | `--layers`'ın ilk katmanında bir kez seçilip sabit tutulur | Katman başına ayrı rol seti L14 ile L19'u karşılaştırılamaz kılardı. Seçilen roller meta artifact'ine yazılır |
| 11 | Steering etkisi için kontrol yönü raporlanmıyor | **Üç kontrol yönü** (`gaussian`, `shuffled`, `rolespan`), ayrı ön-tescille | Aşama 4 tek başına "bu yöne özgü" ile "bu büyüklükte herhangi bir bozulma" arasını ayıramaz — bkz. §5.5 |

---

## 7. Süreç: incelemede yakalanan hatalar

Çalışma boyunca her görev, kodu yazandan bağımsız bir inceleme turundan geçti. **Yedi görevin altısında gerçek hata çıktı** ve bunların çoğu sessizce yanlış sonuç üretecek türdendi. Bulgu sınıfları, benzer çalışmalar için tekrarlanabilir:

**Tutarlılık kontrolünü doğruluk kontrolü sanmak.** Aktivasyon modülünün yedi testi de `prompt_len=0` mutasyonunu geçiyordu — yani prompt token'larını ortalamaya katmak hiçbir testi düşürmüyordu, ki bu modülün var oluş sebebi tam olarak bunu engellemekti. İki padding testi de "aynı öğe tek başına vs. batch'te" karşılaştırıyordu; bu batch şekline duyarlılığı ölçer, sınırın doğru yerde olduğunu değil. Çözüm: `mean_response_activations`'ı bağımsız hesaplanmış bir dilimle karşılaştıran çapraz kontrol. Aynı sınıf hata PCA'nın merkezleme adımında da çıktı — o testin verisinin ortalaması zaten sıfıra yakın olduğu için merkezleme silinse bile geçiyordu.

**Çökmeyi bilimsel sonuçtan ayırmamak.** `07`'nin çıkış kodu 1, "kriter değerlendirildi ve düştü" demek. Ama korumalı bölümden önceki her çökme de 1 veriyordu — eksik bir dosya, bayat bir indeks. Bir çökme "1.7B ölçeğinde eksen oluşmuyor" diye kaydedilebilirdi. Artık her beklenmedik istisna 2, ve 1 yalnızca gerçekten değerlendirilmiş bir kriterden gelebiliyor.

**NaN'ın karşılaştırmalardan sessizce geçmesi.** Sıfır `fully` satırı olduğunda `acts[[]].mean()` NaN veriyor; `contrast_axis`'in sıfır kontrolü NaN'da tetiklenmiyor, `cosine`'ınki de, ve `abs(NaN) <= 0.6` **False** olduğu için hata sebebi eklenmiyor — script tanımsız veriden **"A KRİTERİ: GEÇTİ"** basıyordu. Ve bu senaryo tam da çalışmanın hipotezi (küçük model rolleri tam üstlenemeyebilir).

**Kayan nokta sınır asimetrisi.** `1 - 0.9` ikilik tabanda `0.09999999999999998`. Persentil tam `0.1` düşerken aynadaki `0.9` geçiyordu — ve `0.1`, rol vektörü sayısı 10'un katı olduğunda tam olarak elde edilebilir bir değer.

**Bütçe tavanının delinebilir olması.** Kontrol `chat()` başına, harcama retry başına yapılıyordu; tavan 2 iken 3 istek gitti. Paylaşımlı bir production sunucusuna karşı tutması gereken tek sayı buydu.

**Kimlik zincirinin cevaplara kör olması.** `run_id` yalnızca spec alanlarını hash'liyordu, cevapları değil. Üretim `temperature=1.0` ile örneklediği için yeniden koşu **aynı kimlikle farklı cevaplar** üretiyordu; aşağı akıştaki iki artifact de cevaba bağlı. Bütçe baskısı operatörü eski etiketleri kullanmaya iterken üç bütünlük kontrolü de yeşil geçiyordu.

**Kör olmayan körleme.** Hakem kapısının çalışma sayfasında makine puanı, operatörün dolduracağı sütunun hemen solundaydı. Körlük yalnızca yazılı bir uyarıya dayanıyordu. Artık makine puanları ayrı dosyada; sayfa yapısal olarak kör.

**Bellek aritmetiğinin gerçek tepe noktasını ıskalaması.** Önerilen batch boyutu ilk batch'te OOM veriyordu: kullanılmayan KV cache, 151.936'lık kelime dağarcığı üzerinden `lm_head` logit'leri ve önceki iterasyonun canlı tensörleri hesaba katılmamıştı. Gerçek tepe, aktivasyon tensörünün ~7.65 katıydı.

### Aşama 4'te çıkanlar

Aynı sınıflar tekrarladı, ama üçü yeni ve hepsi "sessizce yanlış bilim üretir" kategorisinde:

**Ölçen hook ile yazan hook'un farklı tensörü görmesi.** `steer()` doğru tensöre yazıyordu ama `hidden_states[l+1]` bit-bir-bit değişmiyordu. Sebep: transformers, `output_hidden_states=True` istendiğinde her decoder katmanına kendi okuma hook'unu **tembelce ve kalıcı olarak** takıyor; bizden önce takılmışsa varsayılan sırada bizden önce çalışıp steering öncesi değeri yakalıyor. Hesaplama doğruydu (logitler 0.75'e kadar sapıyordu), *gözlem* yanlıştı. `prepend=True` ile çözüldü ve `torch.equal` ile hesaplamayı değiştirmediği doğrulandı.

**Steering'i tamamen kapatan mutasyonun tüm testlerden geçmesi.** Sweep script'inin 22 testi vardı; `strength=strength` → `strength=0.0` mutasyonu — yani steering'i kapatmak — **hiçbirini düşürmüyordu**. `direction=direction` satırını silmek de öyle. Çalışmanın tüm iddiası steering'in bir şey yaptığıydı. Sonraki turda reviewer kendi uydurduğu bir mutasyonla ikinci bir eksen buldu: `catalog[role]` → `catalog[role_keys[0]]`, yani **rol ekseninin tamamen çökmesi**, 31 testin hepsini geçiyordu. Şimdi güç kümesi, katman başına norm ve yön, rol/soru çapraz çarpımı ve kaydın gücü ayrı ayrı çivilenmiş.

**Bayat etiketlerin yeniden üretilmiş bir sweep'e uygulanabilmesi.** Değerlendirme script'i etiket dosyasını yeniden kullanıyordu ama dosyanın hangi sweep'ten geldiğine dair hiçbir bağı yoktu, ve "bu hücre tamam mı" sorusunu *sayı karşılaştırarak* cevaplıyordu. Sweep yeniden üretilirse (ki `temperature=1.0` yüzünden her yanıt yeni olur) her hücre "tamam" ilan edilir, **sıfır hakem çağrısı** yapılır ve diskte artık bulunmayan yanıtlardan tam bir "GEÇTİ" basılırdı. Tuzak şuydu: akla ilk gelen parmak izi — meta'daki `axis_run_id` — **yetmiyor**, çünkü aktivasyon indeksinden geliyor ve yeniden üretilen sweep aynı değeri taşıyor. Çözüm sweep dosyasının kendi baytlarının sha256'sı, artı hücre başına kayıt sayısı, artı kapsam tabanlı tamamlanma kontrolü.

**Planın kendi testiyle çelişmesi.** B kriteri eşiği `delta >= 0.25` yazılmıştı ama planın kendi testi `{-0.6: 0.35, 0.0: 0.10}` çiftinin geçmesini istiyordu — ve float64'te `0.35 - 0.10 == 0.24999999999999997`. Tam eşikteki bir sonuç düşerdi. Adlandırılmış bir `B_THRESHOLD_EPS = 1e-9` ile çözüldü; pay, gürültüden (~1e-16) 7 mertebe büyük, eşikten 8 mertebe küçük ve `criterion_b.json`'a `threshold_eps` alanı olarak yazılıyor — karar artefaktı payını kendi içinde söylüyor.

**Deponun kendi dersini almamış olması.** Sweep script'i 3500 üretimi bellekte biriktirip sonda bir kez yazıyordu; 3400'de bir CUDA OOM 1.5 saati kaybettirirdi. Oysa aynı depoda, plandan **önceki** bir commit'te, 2000 etiketlik bir geçişi 1182'de kaybeden aynı hata zaten artımlı kalıcılıkla çözülmüştü. Plan o dersi taşımamıştı. Şimdi her 100 üretimde atomik yazım var — ve gerçek koşuda da işe yaradı: hakem bir batch'te uzunluk uyuşmazlığı verdi, böl-ve-kurtar devreye girdi, koşu durmadı.

---

## 8. Sınırlamalar ve confound'lar

**Hakem kapısı insan doğrulaması değil.** Spec 45 kaydın **insan** tarafından etiketlenmesini öngörüyordu; operatörün kararıyla etiketleme modele devredildi. Ölçülen %77.8 iki dil modeli arasındaki uyumdur. Uyuşmazlık sistematikti (10'un 9'u aynı yönde, analitik rollerde toplanmış) ve sonradan yapılan ölçüm hakemin okumasının daha iyi temellendiğini gösterdi — tartışmalı cevapların promptsuz cevaba embedding benzerliği 0.80-0.96 idi. Etiketler bu analizden sonra **düzeltilmedi** (düzeltmek uyumu %97.8'e çıkarır ama ölçümü yok ederdi). Pratik sonuç: gerçek uyum muhtemelen %77.8'den yüksek. Kapı yalnızca 1.7B için koşuldu.

**Probe iki modelde de düştü.** Held-out uyum 1.7B'de %63.5, 0.6B'de %65.8; eşik %85. Teşhis ölçüldü: eğitim doğruluğu %69.4 — probe kendi eğitim verisine bile uyamıyor, yani daha çok etiket kurtarmaz. Gürültü hakemin kendi tutarsızlığından geliyor (etiketlerdeki varyansın %75.6'sını rol açıklıyor, soru yalnızca %55.4'ünü — yani gürültü sorudan değil hakemin kendi tutarsızlığından geliyor; iki modeldeki değerlerin yakınlığı bunu doğruluyor). Her iki model de **rol düzeyi geri çekilmesi** kullandı: makalenin ≥10 kuralı rol düzeyine taşındı, hiçbir kategoride 10'a ulaşamayan rol düşürüldü (1.7B'de 24, 0.6B'de 33). Bu, kategorileri makaleden daha kaba yapar.

**0.6B rolleri daha tutarsız üstleniyor.** 34 `fully` rol (1.7B'de 55) ve 33 düşürülen rol (24'e karşı). Daha az ve gürültülü `fully` vektörü kontrast vektörünü etkileyebilir; yine de kosinüs 0.87, eksen iyi belirlenmiş.

**İki nokta bir eğri değildir.** Ölçek hikâyesi iki ölçülen nokta ve makalenin raporladığı üçüncüsüne dayanıyor. Aynı ailede üçüncü bir nokta (3-4B) hikâyeyi belirgin şekilde güçlendirirdi; 8 GB VRAM buna izin vermiyor.

**Embedding benzerliği aktivasyon mesafesinin vekilidir.** §8'deki promptsuz-cevap analizi bir ön sinyaldi, A kriterinin yerine geçmez.

**B kriterinin tabanı "promptsuz varsayılan" değil.** Sweep'in her üretimi bir rol sistem promptu taşıyor (eksende Assistant ucuna en yakın 50 rol). Bu yüzden 0.0 gücündeki taban zaten %45.6 Assistant-dışı. Kriter, "varsayılan asistandan role" geçişi değil, **"rol promptlu ama steering'siz"den "rol promptlu ve uzağa steering'li"ye** artışı ölçer. Ön-tescil bunu böyle sabitledi ve değiştirilmedi, ama sayı bu bağlamda okunmalı.

**Persona hakemi yalnızca `hakem-llm`.** Aşama 0.5'in kapısı rol ifadesi rubriği için koşuldu; **7 kategorili persona rubriği için ayrı bir insan doğrulaması yok.** `weird_role` ile `nonsensical` arasındaki sınır bu çalışmada belirleyici (artışın %80'i weird_role'den geliyor) ve o sınırı tek bir hakem çiziyor. Kategori dağılımı bu yüzden ham sayılarıyla raporlandı — okuyucu sınırın nereye çekildiğini kendi değerlendirebilsin.

**Bir katman çifti bir eğri değildir.** L14 ile L19 arasındaki ayrışma iki noktadan okunuyor. Tüm derinlik boyunca bir steering taraması mekanizmayı çok daha iyi belirlerdi (etki derinlikle mi azalıyor, yoksa L19'a özgü bir şey mi?), ama her katman 1750 üretim daha demek — ~45 dakika GPU başına.

**Kontroller tek tohum, tek katman.** Her kontrol yönü `seed=0` ile bir kez üretildi ve yalnızca L14'te ölçüldü. Üç yönün üçü de negatif çıktığı için sonuç sağlam görünüyor, ama tohum başına tek çekiliş, "bu *özel* rastgele yön şanssızdı" itirazına kapalı değil. Yön başına 3-5 tohum, yön başına 75 çağrı daha isterdi.

**Tek model.** Aşama 4 yalnızca 1.7B'de koşuldu. 0.6B'de A kriteri farklı bir biçimde düşmüştü (varsayılan hiçbir derinlikte uca yaklaşmıyor); orada steering'in de çalışıp çalışmadığı ölçülmedi. Ölçek hikâyesinin nedensel yarısı eksik.

---

## 9. Altyapı

- **83 commit**, ~7.300 satır kod (`src/aax/` + `scripts/`)
- **593 test**, 9'u GPU işaretli (varsayılan koşuda 584 geçer); tamamı ağdan ve modelden yapısal olarak izole
- Testlerin dağılımı en riskli modüllerde yoğun: `generate_role_data` 77, `gateway` 57, `label_and_train_probe` 53, `judge_gate` 45, `evaluate_steering` 38, `steering_sweep` 35, `extract_axis` 32, `axis` 31, `susceptibility` 21, `steering` 15

**Modül sınırları.** `axis.py` saf numpy — model, GPU, ağ bilmez. Bu sayede içine bilinen bir yön ekilmiş sentetik veriyle PCA'nın o yönü geri bulduğu, gerçek veriye dokunmadan doğrulanabiliyor. `gateway.py` dışarı giden tek HTTP noktası. `activations.py`'nin hook'u HF'in kendi `output_hidden_states` çıktısıyla `atol=1e-3`'te eşleştiği doğrulandı.

**Çoklu model.** Artifact'ler modele göre kapsamlı (`AAX_TARGET_MODEL` → `data/models/<slug>/`). Gateway aşama alt bütçeleri (aşama, model) başına; global 1500 tavanı tüm anahtarların tek toplamı olarak kalır.

**Bütçe:** 1191 / 1500 harcandı.

| Anahtar | Gönderim | Tavan |
|---|---|---|
| `stage4_steering:qwen3-1.7b` | 353 | 360 |
| `stage4_controls:qwen3-1.7b` | 225 | 240 |
| `stage2_probe_labels` (1.7B, eski çıplak anahtar) | 300 | 300 |
| `stage2_probe_labels:qwen3-0.6b` | 182 | 300 |
| `stage0_roles` | 120 | 145 |
| `stage05_judge_gate` | 9 | 15 |
| `smoke` | 2 | 10 |

Aşama 4'ün alt bütçesi ölçümden önce 210'dan 360'a çıkarıldı — gerçek plan 14 grup × 25 çağrı = 350'ydi. Gerçek harcama 353 oldu (350 plan + böl-ve-kurtar 2 + hatalı anahtarla yanan 1 gönderim), yani 7 pay kaldı.

Kontrol deneyi kendi anahtarını aldı (`stage4_controls`, 240) çünkü `stage4_steering`'de 7 çağrı kalmıştı ve kontrol ayrı bir deneydir; harcaması da ayrı okunabilmeli. Karşılığı **henüz koşulmamış** `stage5_drift`'ten alındı (385 → 145), böylece aşama tavanları toplamı 1.470'te kaldı. Gerçek harcama 225 oldu, sıfır bölünmeyle.

**Global 1500 tavanına projenin hiçbir aşamasında dokunulmadı**; o sayı kullanıcının paylaşımlı bir production endpoint'i için onayladığı sınır. Bir alt bütçe sıkıştığında çözüm hep başka bir aşamadan pay almak oldu, tavanı yükseltmek değil. **Sonuç:** Aşama 5 artık 145 çağrılık bir bütçeyle duruyor ve koşulmadan önce yeniden bütçelenmek zorunda — `config.py`'de Türkçe yorumla ve spec'in Aşama 5 bölümünde açıkça kayıtlı.

---

## 10. Artifact envanteri

```
docs/superpowers/specs/2026-08-04-assistant-axis-replication-design.md   tasarım + sapmalar
docs/superpowers/plans/2026-08-04-plan1-gateway-and-role-data.md         Plan 1
docs/superpowers/plans/2026-08-06-plan2-axis-extraction.md               Plan 2
docs/superpowers/plans/2026-08-10-plan3-steering.md                      Plan 3 (Aşama 4)
results/scale_hypothesis_preregistration.json                            ön-tescil (sonuç görülmeden)
results/steering_preregistration.json                                    B kriteri ön-tescili (koddan ve ölçümden önce)
results/scale_comparison.json                                            ölçek karşılaştırması
results/pilot/baseline_distance.json                                     rollerin varsayılana uzaklığı
results/models/<slug>/axis/criterion_a.json                              A kriteri kararı
results/models/<slug>/axis/layer_sweep.json                              katman taraması
results/models/<slug>/axis/assistant_axis.npy                            eksen (her katman)
results/models/<slug>/axis/role_vectors.npy                              rol vektörleri
results/models/<slug>/steering/criterion_b.json                          B kriteri kararı (katman başına)
results/models/<slug>/steering/rate_by_strength.json                     doz-yanıt eğrisi (Şekil 4 muadili)
results/control_preregistration.json                                     kontrol ön-tescili (koddan ve ölçümden önce)
results/models/<slug>/steering/criterion_c.json                          C kriteri kararı (üç kontrol yönü)
results/models/<slug>/steering/rate_by_strength_<yön>.json                kontrol doz-yanıt eğrileri
```

`data/` commit edilmez (16k rollout metni + 5.3 GB aktivasyon), `results/` edilir.

---

## 11. Sırada ne var

Aşama 4 ve kontrol deneyi bitti. Çalışmanın en savunmasız yeri — "etki eksene mi özgü?" — kapandı (§5.5). Açık kalan sorular:

**0. Kontroller tek tohumla ölçüldü.** Yön başına 3-5 tohum, "bu *özel* rastgele yön şanssızdı" itirazını da kapatırdı. Yön başına 75 çağrı; kalan bütçe (309) üçünü de üç tohuma çıkarmaya yetmez, seçim gerekir.

**1. Ayrışma gerçek mi, iki noktanın gürültüsü mü?** L14 ile L19 arasındaki duyarlılık farkı, tüm derinlik boyunca bir steering taraması ile sınanmalı. Eğer etki derinlikle tekdüze azalıyorsa açıklama basit (steering'in yayılacak katmanı kalmıyor); eğer L14 civarında bir tepe varsa mekanizma daha ilginç. Katman başına 1750 üretim ≈ 45 dakika GPU.

**2. Nedensel yarısı 0.6B'de ne olur?** 0.6B'de varsayılan hiçbir derinlikte uca yaklaşmıyordu. Steering yine de çalışır mı? Eğer çalışırsa, "uçta olmak müdahale için gerekli değil" iddiası ikinci bir ölçekte doğrulanır; çalışmazsa ayrışmanın sınırı bulunur. **Bütçe artık yetmiyor:** kalan 309, ihtiyaç ~350. Kontrol deneyi 225 çağrı harcadı ve global tavan 1500 yükseltilmeyecek. Yani bu, ya başka bir aşamadan pay alınarak ya da daha az rolle (ör. 40 rol → 280 çağrı) koşulabilir. Ayrıca sweep'in GPU süresi tekrar 1.5 saat.

**Aşama 5-7 (kapsamda, koşulmadı):** persona drift, aktivasyon capping (Eq. 1), Türkçe transfer testi. Kalan bütçe 309; üçünün tavanları toplamı 400 — hepsi koşulacaksa yeniden bütçelenmeli. (Not: "C kriteri" adı bu çalışmada kontrol yönlerine verildi; persona drift için ayrı bir kriter tescillenecek.)

**Kapatılmamış boşluklar:**
- Probe'un otomatik geri çekilmesi uygulanmadı; `06` operatöre seçenekleri yazıp duruyor. İki koşuda da elle `--role-level-fallback` seçildi.
- Persona rubriği için hakem kapısı yok (§8).
- `09`, hiçbir grup tamamlanmadan düştüğünde de "ilerleme kalıcı olarak yazıldı" diyor; yazacak bir şey olmadığında bunu söylememeli.
