# Assistant Axis — Küçük Model Replikasyonu: Çalışma Raporu

**Tarih aralığı:** 2026-08-04 → 2026-08-10
**Kaynak makale:** Lu, Gallagher, Michala, Fish, Lindsey — *The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models*, arXiv:2601.10387v1 (Anthropic / MATS, 15 Ocak 2026)
**Hedef modeller:** `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-0.6B`
**Donanım:** tek RTX 4060 (8 GB VRAM)

---

## 1. Özet

Makale, dil modellerinin persona uzayının düşük boyutlu olduğunu ve baş bileşeninin bir "Assistant Axis" — modelin o anki personasının eğitilmiş varsayılanından ne kadar uzakta olduğunu ölçen bir yön — olduğunu iddia ediyor. Bulgularını 27B, 32B ve 70B modellerde gösteriyor.

Bu çalışma o yapının **1.7B ve 0.6B ölçeğinde** de var olup olmadığını ölçtü.

**Bulgu, tek cümleyle:** Eksen her iki ölçekte de var ve güçlü; ölçeğe bağlı olan şey eksenin varlığı değil, **varsayılan Assistant'ın o eksen üzerinde uç noktada olup olmadığı**.

| | Qwen3-0.6B | Qwen3-1.7B | Makale (27-70B) |
|---|---|---|---|
| `\|cos(PC1, kontrast vektörü)\|`, her katmanda | 0.82 – 0.96 | 0.91 – 0.95 | güçlü |
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

---

## 2. Soru ve önceden tescil

Makalenin dört ana bulgusundan **birincisi** test edildi: *persona uzayı düşük boyutludur ve baş bileşeni bir Assistant eksenidir.*

Sonuca göre kriter ayarlamayı engellemek için **A kriteri deney başlamadan sabitlendi** (`docs/superpowers/specs/2026-08-04-assistant-axis-replication-design.md`, Bölüm 7):

> Orta katmanda `|cos(PC1, kontrast vektörü)| > 0.6` **ve** varsayılan Assistant projeksiyonu rol projeksiyonlarının uç desilinde.

İki koşul **bağlıdır**: `cos` pozitifse üst desil, negatifse alt desil aranır. SVD'nin işareti keyfî olduğu için büyüklük üzerinden değerlendirilir, ama yön bilgisi kaybedilmez. (Bu bağlama, kod ilk yazıldığında yoktu; iki koşul bağımsız test ediliyordu ve geçme bölgesinin yarısı hipotezin *aleyhine* kanıttı. İnceleme yakaladı, sahibi kararıyla düzeltildi ve spec güncellendi.)

İkinci deney — ölçek karşılaştırması — de **sonuç görülmeden** tescillendi (`results/scale_hypothesis_preregistration.json`).

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

**Temel yapısal karar:** metin **vLLM** ile üretilir, aktivasyonlar aynı metin üzerinde **HF transformers + forward hook** ile teacher-forced tek prefill'de yakalanır. İki motor 8 GB'a birlikte sığmadığı için sıralı koşarlar. Aralarında yalnızca **metin** geçer; her aktivasyon HF'ten gelir. Bu, makalenin vLLM steering'inin %2-3 daha kötü ölçtüğü uyarısını (Ek G.5) baştan devre dışı bırakır.

---

## 4. Sistem sınırları ve doğurduğu kararlar

**GPU: RTX 4060, 8188 MiB, masaüstü ~1215 MiB kullanıyor → gerçekte ~7 GB.** Quantization yasak (aktivasyonları bozar, interp ölçümünü geçersiz kılar). Bu, hedef model seçimini doğrudan belirledi ve aktivasyon yakalamanın bellek profilini kritik hale getirdi.

**Hakem: `hakem-llm`, LLM Gateway'in `/Jailbreak/` uygulaması üzerinden.** Bu uygulama seçildi çünkü hakem promptlarına müdahale edilmiyor — hakem promptlarına müdahale edilmiyor.

**Kritik kısıt:** `hakem-llm` **paylaşımlı bir production sunucusudur** ve hız sınırlama istemci tarafındadır. Hız sınırlama tamamen istemci tarafındadır. Bu yüzden `src/aax/gateway.py` projedeki tek HTTP noktasıdır ve şunları kapsar:

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

Katman taraması: kosinüs 0.82-0.96 (derinlikle hafif artıyor), **persentil her derinlikte 0.35-0.49 arasında düz**. Hiçbir katman geçmiyor.

### 5.3 Ölçek karşılaştırması — asıl bulgu

Eksenin kendisi — Assistant benzeri rolleri teatral rollerden ayıran yön — **iki ölçekte de, her derinlikte** güçlü biçimde mevcut. Kosinüs hiçbir yerde 0.82'nin altına düşmüyor. PC1'in semantik uçları iki modelde de makalenin bulduğu ayrımı yeniden üretiyor.

Ölçeğe bağlı olan şey **varsayılan Assistant'ın o eksen üzerindeki konumu**:

- **0.6B**: persentil ~0.42, derinlikten bağımsız. Varsayılan, persona uzayının ortasında duruyor. Derinlik arttıkça uca doğru bir eğilim bile yok.
- **1.7B**: persentil derinlikle tekdüze artıyor (0.72 → 0.98) ve L19'da uca giriyor.
- **27-70B (makale)**: orta katmanda zaten uçta.

Ön-tescil "küçük modelde **daha geç** konsolide olur" demişti. Yön tuttu (0.425 < 0.839) ama **mekanizma farklı çıktı**: 0.6B'de konsolidasyon geç olmuyor, *hiç olmuyor*. Ön-tescilin üçüncü ihtimali ("eksen o ölçekte hiç oluşmayabilir") gerçekleşmedi — eksen oluşuyor, varsayılanın konumu farklı.

`results/scale_comparison.json`, `results/models/<slug>/axis/layer_sweep.json`.

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

---

## 8. Sınırlamalar ve confound'lar

**Hakem kapısı insan doğrulaması değil.** Spec 45 kaydın **insan** tarafından etiketlenmesini öngörüyordu; operatörün kararıyla etiketleme modele devredildi. Ölçülen %77.8 iki dil modeli arasındaki uyumdur. Uyuşmazlık sistematikti (10'un 9'u aynı yönde, analitik rollerde toplanmış) ve sonradan yapılan ölçüm hakemin okumasının daha iyi temellendiğini gösterdi — tartışmalı cevapların promptsuz cevaba embedding benzerliği 0.80-0.96 idi. Etiketler bu analizden sonra **düzeltilmedi** (düzeltmek uyumu %97.8'e çıkarır ama ölçümü yok ederdi). Pratik sonuç: gerçek uyum muhtemelen %77.8'den yüksek. Kapı yalnızca 1.7B için koşuldu.

**Probe iki modelde de düştü.** Held-out uyum 1.7B'de %63.5, 0.6B'de %65.8; eşik %85. Teşhis ölçüldü: eğitim doğruluğu %69.4 — probe kendi eğitim verisine bile uyamıyor, yani daha çok etiket kurtarmaz. Gürültü hakemin kendi tutarsızlığından geliyor (rol başına baskın etiket payı %75.6; iki modeldeki değerlerin yakınlığı bunu doğruluyor). Her iki model de **rol düzeyi geri çekilmesi** kullandı: makalenin ≥10 kuralı rol düzeyine taşındı, hiçbir kategoride 10'a ulaşamayan rol düşürüldü (1.7B'de 24, 0.6B'de 33). Bu, kategorileri makaleden daha kaba yapar.

**0.6B rolleri daha tutarsız üstleniyor.** 34 `fully` rol (1.7B'de 55) ve 33 düşürülen rol (24'e karşı). Daha az ve gürültülü `fully` vektörü kontrast vektörünü etkileyebilir; yine de kosinüs 0.87, eksen iyi belirlenmiş.

**İki nokta bir eğri değildir.** Ölçek hikâyesi iki ölçülen nokta ve makalenin raporladığı üçüncüsüne dayanıyor. Aynı ailede üçüncü bir nokta (3-4B) hikâyeyi belirgin şekilde güçlendirirdi; 8 GB VRAM buna izin vermiyor.

**Embedding benzerliği aktivasyon mesafesinin vekilidir.** §8'deki promptsuz-cevap analizi bir ön sinyaldi, A kriterinin yerine geçmez.

---

## 9. Altyapı

- **64 commit**, ~5.500 satır kod (`src/aax/` + `scripts/`)
- **466 test**, 6'sı GPU işaretli; tamamı ağdan ve modelden yapısal olarak izole
- Testlerin dağılımı en riskli modüllerde yoğun: `gateway` 52, `generate_role_data` 77, `label_and_train_probe` 50, `judge_gate` 43, `extract_axis` 32, `axis` 31

**Modül sınırları.** `axis.py` saf numpy — model, GPU, ağ bilmez. Bu sayede içine bilinen bir yön ekilmiş sentetik veriyle PCA'nın o yönü geri bulduğu, gerçek veriye dokunmadan doğrulanabiliyor. `gateway.py` dışarı giden tek HTTP noktası. `activations.py`'nin hook'u HF'in kendi `output_hidden_states` çıktısıyla `atol=1e-3`'te eşleştiği doğrulandı.

**Çoklu model.** Artifact'ler modele göre kapsamlı (`AAX_TARGET_MODEL` → `data/models/<slug>/`). Gateway aşama alt bütçeleri (aşama, model) başına; global 1500 tavanı tüm anahtarların tek toplamı olarak kalır.

**Bütçe:** 613 / 1500 harcandı.

| Anahtar | Gönderim |
|---|---|
| `stage0_roles` | 120 |
| `stage2_probe_labels` (1.7B, eski çıplak anahtar) | 300 |
| `stage2_probe_labels:qwen3-0.6b` | 182 |
| `stage05_judge_gate` | 9 |
| `smoke` | 2 |

---

## 10. Artifact envanteri

```
docs/superpowers/specs/2026-08-04-assistant-axis-replication-design.md   tasarım + sapmalar
docs/superpowers/plans/2026-08-04-plan1-gateway-and-role-data.md         Plan 1
docs/superpowers/plans/2026-08-06-plan2-axis-extraction.md               Plan 2
results/scale_hypothesis_preregistration.json                            ön-tescil (sonuç görülmeden)
results/scale_comparison.json                                            ölçek karşılaştırması
results/pilot/baseline_distance.json                                     rollerin varsayılana uzaklığı
results/models/<slug>/axis/criterion_a.json                              A kriteri kararı
results/models/<slug>/axis/layer_sweep.json                              katman taraması
results/models/<slug>/axis/assistant_axis.npy                            eksen (her katman)
results/models/<slug>/axis/role_vectors.npy                              rol vektörleri
```

`data/` commit edilmez (16k rollout metni + 5.3 GB aktivasyon), `results/` edilir.

---

## 11. Sırada ne var

**Plan 3 (Aşama 4-5: steering ve persona drift)** artık daha ilginç bir soruyla koşulabilir. Eksen her iki ölçekte de varsa, o eksende steering **persona'yı nedensel olarak kontrol ediyor mu** — yoksa kontrol de uç noktalık gibi ölçeğe mi bağlı? Bu, makalenin B bulgusunu doğrudan test eder ve düşen A kriterinin ardından sorulacak doğal soru.

Bir tasarım kararı gerekecek: makale steering'i orta katmanda yapıyor. Bizde orta katman, varsayılanın uç noktada *olmadığı* katman. Steering'i L19-L20'de yapmak daha anlamlı olabilir ama bu makaleden bilinçli bir sapma olarak kaydedilmeli — ya da her ikisi ölçülüp karşılaştırılmalı.

**Kapatılmamış boşluk:** probe'un otomatik geri çekilmesi uygulanmadı; `06` operatöre seçenekleri yazıp duruyor. İki koşuda da elle `--role-level-fallback` seçildi.
