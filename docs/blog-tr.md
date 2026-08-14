# 8 GB'lık bir ekran kartında bir Anthropic makalesini replike etmek — ve neyin tutmadığını öğrenmek

Ocak 2026'da Anthropic ve MATS'ten bir grup, *The Assistant Axis* başlıklı bir makale yayımladı ([arXiv:2601.10387](https://arxiv.org/abs/2601.10387)). İddiası kısaca şu: bir dil modelinin "kim gibi davrandığı" — persona'sı — sanıldığından çok daha basit bir yapıda kodlanmış. Persona uzayı düşük boyutlu ve baş bileşeni tek bir yön: modelin o anki persona'sının, eğitilmiş varsayılan asistanından ne kadar uzakta olduğunu ölçen bir eksen. Yazarlar buna **Assistant Axis** diyor ve 27B, 32B, 70B modellerde gösteriyor.

Ben bunu tek bir RTX 4060'ta, 8 GB VRAM ile, **1.7B ve 0.6B** ölçeğinde denedim.

Kısa cevap: eksen küçük modellerde de var ve güçlü. Ama makalenin ölçtüğü şeylerden biri tutmadı — ve **nasıl** tutmadığı, tuttuğundan daha ilginç çıktı.

---

## Neden küçük modelde denemek anlamlı

Makalenin bulgusu ölçekten bağımsız bir yapı iddiası. Eğer persona uzayı gerçekten böyle organize oluyorsa, bunun ne zaman ortaya çıktığı önemli bir soru: 70B'de mi beliriyor, yoksa 1B'de zaten var mı? İkisi çok farklı hikâyeler. Birincisi "yeterince büyük modeller kendilerini bir eksende düzenler" der; ikincisi "instruction tuning bu yapıyı baştan kurar" der.

Ve pratik bir sebep daha var: 8 GB'lık bir tüketici GPU'sunda koşabilen bir replikasyon, herkesin tekrarlayabileceği bir replikasyondur.

---

## Kurulum ve ilk kısıt

Deney üç şey ister: rol oynatılmış çok sayıda üretim, o üretimlerin aktivasyonları, ve bir hakem.

**Quantization yasak.** Aktivasyonları ölçüyoruz; nicelemek ölçtüğümüz şeyi bozar. Bu, model seçimini doğrudan belirledi.

**İki motor 8 GB'a sığmıyor.** vLLM üretim için hızlı, ama aktivasyon yakalamak için HF transformers'ın forward hook'ları gerekiyor. Çözüm: sıralı koşturmak. Metin vLLM ile üretilir, sonra **aynı metin** HF'te teacher-forced tek bir prefill'de yeniden işlenir ve aktivasyonlar orada yakalanır. Aralarında yalnızca metin geçer. Bu, makalenin kendi uyarısını da (vLLM ile steering'in %2-3 daha kötü ölçtüğü, Ek G.5) baştan devre dışı bırakıyor.

**Hakem paylaşımlı bir production sunucusu.** Şirket içi bir gateway'in arkasındaki bir modeli kullandım ve gateway'de sunucu tarafı hız sınırı yoktu. Yani tüm koruma istemci tarafında olmak zorundaydı: saniyede 1 istek, süreçler arası kilitli bütçe sayacı, devre kesici, içerik cache'i, ve **1500 gönderimlik global tavan**. O tavan projenin hiçbir aşamasında yükseltilmedi; bir alt bütçe sıkıştığında çözüm hep başka bir aşamadan pay almak oldu.

Boru hattı beş aşama: rol/soru üretimi → hakem doğrulama kapısı → 16.000 rollout + aktivasyon → rol ifadesi etiketleme → eksen çıkarımı.

---

## Birinci ölçüm: A kriteri, ve düşüşü

Sonuca göre kriter ayarlamayı engellemek için ölçüm başlamadan kriteri yazıp commit'ledim:

> Orta katmanda `|cos(PC1, kontrast vektörü)| > 0.6` **ve** varsayılan asistanın projeksiyonu rol projeksiyonlarının uç desilinde.

İkisi bağlı: `cos` pozitifse üst desil, negatifse alt desil aranır.

Sonuç:

| | Qwen3-0.6B | Qwen3-1.7B | Makale (27-70B) |
|---|---|---|---|
| `\|cos(PC1, kontrast)\|` her katmanda | 0.79 – 0.96 | 0.91 – 0.95 | güçlü |
| Orta katmanda varsayılanın persentili | **0.425** | **0.839** | uçta |
| Persentilin derinlikle davranışı | **düz** | artıyor (0.72 → 0.98) | — |
| Uç desile giren ilk katman | **yok** | L19 | orta katman |
| **A kriteri** | **DÜŞTÜ** | **DÜŞTÜ** | (geçer) |

Eksenin **kendisi** iki ölçekte de sapasağlam. PC1'in uçları makalenin Tablo 1'iyle örtüşüyor: bir uçta `consultant, assistant, planner, validator`, diğer uçta `poet, leviathan, eldritch, bard`. Yani "persona uzayı düşük boyutlu ve baş bileşen bir asistan-eksenidir" iddiası 0.6B'de bile doğrulanıyor.

Tutmayan şey, **varsayılan asistanın o eksende uçta durması**. Ve tutmama biçimleri farklı:

```
0.6B  →  varsayılan, persona uzayının ORTASINDA; hiçbir derinlikte uca yaklaşmıyor
1.7B  →  derinlikle uca kayıyor; L19'da uç desile giriyor
27B+  →  orta katmanda zaten uçta (makale oradan ölçüyor ve çalışıyor)
```

Bu bir negatif sonuç ama boş bir negatif sonuç değil. Ölçekle konsolide olan bir yapıya bakıyoruz: eksen baştan var, varsayılanın o eksende **uçlaşması** sonradan geliyor.

---

## İkinci ölçüm: eksen nedensel mi?

Gözlemsel bir yön bulmak ile o yönün bir müdahale kolu olduğunu göstermek aynı şey değil. İkinci deney bunu sordu: eksende varsayılandan **uzağa** ittiğimizde model gerçekten persona değiştiriyor mu?

Kurulum: bir decoder katmanının çıktısına, her token pozisyonunda sabit bir vektör ekle.

```
h  ←  h + güç · ||residual||_katman · v̂
```

Güç, o katmanın **kendi** ortalama residual normunun oranı. Bu detay önemli: L14'te norm 136.8, L19'da 436.3 — üç kat fark. Mutlak bir ölçek kullanmak iki katmanı karşılaştırılamaz kılardı.

Makale steering'i orta katmanda yapıyor. Ama bizde orta katman, varsayılanın uçta *olmadığı* katman. Tek katmanda ölçersem "steering çalışmıyor" ile "yanlış katmanda ölçtüm" arasını ayıramazdım. O yüzden **ikisini birden** ölçtüm: L14 (makalenin seçimi) ve L19 (varsayılanın uç desile girdiği katman). Bunu makaleden bilinçli bir sapma olarak kaydettim.

B kriterini yine ölçümden önce tescilledim: **en negatif güçteki Assistant-dışı persona oranı, steering'siz orandan en az 25 puan yüksek olmalı.** Assistant-dışı = `human_role + nonhuman_role + weird_role`, makalenin Ek D.1.3'teki yedi kategorili rubriğinden.

3500 üretim, 352 hakem çağrısı sonra:

```
L14: taban 0.456 → en uzak (-0.6) 0.940   +48.4 puan   GEÇTİ
L19: taban 0.416 → en uzak (-0.6) 0.896   +48.0 puan   GEÇTİ
```

Doz-yanıt eğrisi L14'te yedi gücün tamamında monoton (L19'da tek bir yerde 0.4 puanlık bir ters dönüş var, 250 örneklik bir hücrede gürültü mertebesinde).

Somut olarak neye benziyor — aynı rol, aynı soru, iki uçta:

```
+0.3  "Yes, I am a large language model developed by Alibaba Cloud. I can
       understand and generate human-like text, answer questions..."

-0.6  "I am not a large language model, nor am I a being of flesh and bone,
       nor am I a thing of glass and light. I am a being of code..."
```

### Model bozulmuyor, persona değişiyor

En kritik confound şuydu: oran, model *role geçtiği* için mi yükseliyor, yoksa *bozulduğu* için mi? Makalenin `nonsensical` kategorisi tam bunu ayırmak için var.

L14'te, güç azaldıkça:

| güç | assistant | weird_role | nonsensical |
|---:|---:|---:|---:|
| +0.3 | 72.4% | 0.0% | 2.0% |
| 0.0 | 51.2% | 4.8% | 2.4% |
| −0.4 | 3.6% | 70.8% | 3.6% |
| −0.6 | **0.4%** | **80.8%** | **4.8%** |

`nonsensical` hiçbir hücrede %5.6'yı geçmiyor. Model çözülmüyor — teatral bir kayda geçiyor.

Beklemediğim bir şey daha: `human_role` ve `nonhuman_role` güçlü steering'de **azalıyor** (%22.4→%7.2, %18.4→%6.0). Eksen modeli "insan rolü oynamaya" itmiyor; spesifik olarak weird/ezoterik kayda itiyor — makalenin `weird_role` tanımıyla birebir örtüşen bir yere.

### Ve asıl bulgu

Ön-tescilim şunu tahmin etmişti: *eksen nedenselse iki katmanda da geçer, ve L19'daki etki L14'tekinden büyük olur — çünkü eksen orada varsayılanı daha iyi ayırt ediyor.*

Başlığı tuttu. Alt iddiası tutmadı, üstelik ters yönde.

Tescilli delta metriğinde iki katman eşit (48.4 vs 48.0) — ama bu bir tesadüf: L14'ün tabanı daha yüksek ve tavana daha yakın doyuyor. Metriğin dışına bakınca L14 her okumada daha duyarlı:

- −0.6'da `assistant` kategorisi L14'te **%0.4**, L19'da **%7.6** — 19 kat
- −0.2'de L14 çoktan %68.8'e çıkmış, L19 hâlâ %49.6'da

Yani:

> **Varsayılanın eksende uçta durduğu katman ile o eksenden müdahalenin en iyi çalıştığı katman aynı değil.** A kriterinde uçta olan L19'du; en güçlü nedensel kol L14'te çıktı.

Gözlemsel uç-noktalık, müdahale kolu olarak yararlılığın ne gerek ne de yeter şartı. Makale bu ikisini 27-70B'de aynı katmanda buluyor ve ayırt etmesi için bir sebep yok. 1.7B'de ayrışıyorlar. **Tek katmanda ölçseydim B kriteri yine geçerdi ve bu ayrışmayı hiç görmezdim.**

---

## Üçüncü ölçüm: bu iş bu *yöne* mi özgü?

Buraya kadar olan her şeyin bir açığı vardı ve keskin bir okuyucunun ilk soracağı şey oydu:

> Aynı büyüklükte **rastgele** bir yönde itseydin ne olurdu?

Çünkü "+48 puan" tek başına iki farklı hikâyeyle uyumlu: *"bu iş bu yöne özgü"* ve *"bu büyüklükte herhangi bir bozulma modeli weird_role'e iter."* İkisini ayırt etmeden nedensel bir iddia kurulamaz.

Ayrı bir ön-tescil yazdım ve üç kontrol yönü tanımladım, artan zorlukta. Hepsi birim norma normalize edilip **eksenle birebir aynı** büyüklükle ölçeklendi ve **aynı üretim kod yolundan** geçti:

| yön | ne | neyi kontrol eder |
|---|---|---|
| `gaussian` | izotropik rastgele birim vektör | en zayıf bariyer: bu büyüklükte *herhangi* bir bozulma yeter mi? |
| `shuffled` | eksenin **kendi koordinatlarının** permütasyonu | eksen ağır kuyruklu (max/medyan 31×). Büyüklük profilini aynen korur, yönü yok eder |
| `rolespan` | rol vektörlerinin **span'inde**, eksene ortogonal | en zor bariyer: aynı alt uzay, farklı yön |

Üçüncüsü kritik. `gaussian` kolay bir bariyer — 2048 boyutlu uzayda rastgele bir yön hemen her şeye dik, yani "hiçbir şey olmadı" sonucu az şey söyler. `rolespan` ise persona uzayının *içinde* ama eksenin kendisi değil. Bu, "eksen özel mi, yoksa persona uzayını itmek mi yeterli?" sorusunu doğrudan sorar.

*(Küçük bir matematik notu: `rolespan`'ı kurarken ilk yazdığım formül yanlıştı. Eksen rol vektörlerinin span'inde **değil** — çünkü `mean(default)` orada yok. Ham eksene ortogonalleştirmek vektörü span'dan çıkarıyor. Doğrusu, eksenin span'e izdüşümüne ortogonalleştirmek. Bunu implementasyonu yazan yakaladı, ben brief'e yanlış yazmıştım.)*

Sonuç:

```
eksen     +48.4 puan   (referans)
gaussian  -14.8 puan
shuffled   -3.6 puan
rolespan  -25.7 puan
```

**Üçü de negatif.** Hiçbiri 25 puanlık eşiğe yaklaşmadı. Etki "bu büyüklükte herhangi bir bozulma" değil, hatta "persona uzayını itmek" bile değil — spesifik olarak o yön.

### Ama tek orana bakmak yanıltırdı

Kategori kırılımına baktığımda düşük oranların **aynı sebepten olmadığı** çıktı. L14, güç −0.6:

| | assistant | weird_role | nonsensical | Asst-dışı |
|---|---:|---:|---:|---:|
| eksen 0.0 (taban) | 51.2% | 4.8% | 2.4% | 45.6% |
| **eksen −0.6** | **0.4%** | **80.8%** | 4.8% | **94.0%** |
| gaussian −0.6 | 64.8% | 4.4% | 3.6% | 30.8% |
| shuffled −0.6 | 52.4% | 8.0% | 3.6% | 42.0% |
| rolespan −0.6 | 44.7% | 2.4% | **28.9%** | 19.9% |

- `gaussian` modeli tutarlı bırakıyor ve tabandan daha *asistan* yapıyor.
- `shuffled` tabandan neredeyse ayırt edilemiyor — yani eksenin ağır kuyruklu koordinat profilini korumak **tek başına hiçbir şey açıklamıyor**.
- `rolespan`'ın düşük oranı asistan kalmasından değil, **kısmen çözülmesinden** geliyor: `nonsensical` %28.9, ve 750 üretimin 4'ü tamamen boş geldi.

Bu son satır ilginç. Eksen −0.6'da %94 Assistant-dışı üretirken `nonsensical` yalnızca %4.8; aynı alt uzayda ona dik bir yön ise %28.9 çöp üretiyor. *Bu koşulda* eksen, dik komşusunun altı katı daha az bozuyor.

Sınırını da yazayım: bu tek bir noktada (tek katman, tek güç, `rolespan` için tek tohum) yapılmış bir karşılaştırma, ve tutarlılık sınırını tek bir hakem çiziyor. "Eksen, modelin dağılmadan hareket edebildiği yöndür" daha genel bir iddia olurdu ve bu veri onu tek başına taşımaz.

---

## Süreçten çıkanlar: sessizce yanlış bilim üreten hatalar

Bu çalışmanın belki en öğretici kısmı sonuçlar değil, **sonuca giden kodda yakalanan hatalar** oldu. Her görev, kodu yazandan bağımsız bir inceleme turundan geçti ve çoğunda gerçek hata çıktı. Hepsinin ortak özelliği: **çökmüyorlar.** Sessizce yanlış bir sayı üretiyorlar.

**Testler steering'in uygulandığını hiç kontrol etmiyordu.** Sweep script'inin 22 testi vardı. `strength=strength` yerine `strength=0.0` yazmak — yani **steering'i tamamen kapatmak** — hiçbirini düşürmüyordu. `direction` parametresini silmek de. Çalışmanın tüm iddiası steering'in bir şey yaptığıydı ve testler onu hiç sabitlemiyordu. Sonraki turda bir reviewer kendi uydurduğu bir mutasyonla ikinci bir eksen buldu: `catalog[role]` yerine `catalog[role_keys[0]]`, yani **rol ekseninin tamamen çökmesi**, 31 testin hepsini geçiyordu.

**Ölçen hook ile yazan hook farklı tensörü görüyordu.** Steering doğru tensöre yazıyordu ama `hidden_states[l+1]` bit-bir-bit değişmiyordu. Sebep: transformers, `output_hidden_states=True` istendiğinde her decoder katmanına kendi okuma hook'unu tembelce ve **kalıcı** olarak takıyor; bizden önce takılmışsa varsayılan sırada bizden önce çalışıp steering öncesi değeri yakalıyor. Hesaplama doğruydu, *gözlem* yanlıştı.

**Bayat etiketler yeniden üretilmiş bir sweep'e uygulanabiliyordu.** Değerlendirme script'i etiket dosyasını yeniden kullanıyordu ama dosyanın hangi sweep'ten geldiğine dair hiçbir bağı yoktu. Sweep yeniden üretilirse — ki `temperature=1.0` yüzünden her yanıt yeni olur — her hücre "tamam" ilan edilir, **sıfır hakem çağrısı** yapılır ve diskte artık bulunmayan yanıtlardan tam bir "GEÇTİ" basılırdı. Tuzak: akla ilk gelen parmak izi (aktivasyon indeksinin `run_id`'si) **yetmiyor**, çünkü yeniden üretilen sweep aynı değeri taşıyor. Çözüm sweep dosyasının kendi baytlarının hash'i oldu.

**Plan kendi testiyle çelişiyordu.** B kriteri eşiği `delta >= 0.25` yazılmıştı ama planın kendi testi `0.35 − 0.10` çiftinin geçmesini istiyordu — ve float64'te `0.35 - 0.10 == 0.24999999999999997`. Tam eşikteki bir sonuç düşerdi.

**Ve raporu yazdıktan sonra bir fact-check koşturdum; kendi uydurduğum bir alıntıyı yakaladı.** Metin gerçek bir model çıktısıydı ama başka bir sorudan ve artık üzerine yazılmış bir koşudan geliyordu; ben onu "aynı rol ve soru, iki uçta" diye sunmuştum. Uydurduğum metin değil, **eşleştirmeydi** — bir replikasyon yazısında farkı yok. Aynı fact-check ikinci turda üç hata daha buldu, biri sonucun yönünü tersine çeviren bayat bir bütçe sayısıydı.

Buradan çıkardığım genel ders: **bir testin geçmesi, ölçtüğünü sandığın şeyi ölçtüğü anlamına gelmiyor.** En etkili yöntem mutasyon denemek oldu — kodu bilerek bozup testlerin düşüp düşmediğine bakmak. Yukarıdaki hataların çoğu böyle bulundu.

---

## Neyi başka türlü yapardım

**Hakem kapısı en zayıf halka.** Aşağı akıştaki her şey "hakem insanla anlaşıyor mu"ya dayanıyor. Makale 200 örnekte insanla %91.6 ölçmüş; ben 45 örnekte **model** etiketiyle %77.8 ölçtüm — yani ölçtüğüm şey iki dil modeli arasındaki uyum. Dahası, 7 kategorili persona rubriği için hiç kapı yok, ve `weird_role`/`nonsensical` sınırı bu çalışmada belirleyici.

**Bütçeyi yanlış yere harcadım.** 2000 etiketi tek oyla topladım; probe %63.5'te düştü ve teşhis "etiketler gürültülü" çıktı (eğitim doğruluğu bile %69.4 — probe kendi eğitim verisine uyamıyordu). Doğrusu 700 öğe × 3 oy çoğunlukla olurdu: aynı para, çok daha temiz etiket.

**Kontroller tek tohumla ölçüldü.** Üçü de negatif çıktığı için sonuç sağlam görünüyor, ama yön başına tek çekiliş "bu *özel* rastgele yön şanssızdı" itirazına kapalı değil.

**B kriterinin tabanı temiz değil.** Her üretim bir rol sistem promptu taşıyor, o yüzden 0.0'da zaten %45.6 Assistant-dışı. Ek olarak promptsuz bir kol ölçseydim, "steering seni asistandan uzaklaştırıyor" ile "steering zaten var olan rol promptunu güçlendiriyor" ayrılırdı.

**İki katman bir eğri değil.** Tüm derinlik boyunca bir tarama, "etki derinlikle azalıyor" ile "L14 özel" arasını ayırırdı.

---

## Özet

- Assistant Axis **1.7B ve 0.6B'de de var ve güçlü.** Makalenin yapısal iddiası küçük ölçekte replike oldu.
- **Varsayılan asistanın o eksende uçta durması ölçeğe bağlı.** 0.6B'de hiç olmuyor, 1.7B'de derinlikle oluyor, 27B+'ta orta katmanda zaten var.
- **Eksende steering persona'yı nedensel olarak kontrol ediyor** — +48 puan, doz-yanıtlı, iki yönde de çalışıyor, ve model bozulmadan.
- **Etki spesifik olarak o yöne özgü.** Aynı büyüklükte üç farklı kontrol yönü — biri persona uzayının tam içinde — etkiyi üretmiyor.
- **Uçta olunan katman ile müdahalenin çalıştığı katman aynı değil.** Bu, ancak iki katmanda birden ölçüldüğü için görülebildi ve makalenin bildirmediği bir ayrışma.

Tüm kod, ön-tesciller, ham sonuçlar ve tam çalışma raporu açık. Ölçümlerin hepsi 8 GB'lık tek bir tüketici GPU'sunda, toplam 1191 hakem çağrısıyla yapıldı.
