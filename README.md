# assistant-axis

[*The Assistant Axis*](https://arxiv.org/abs/2601.10387) (arXiv:2601.10387) makalesinin **Qwen3-1.7B ve Qwen3-0.6B** üzerinde replikasyonu. Üç kriter de ölçümden önce tescillendi.

## Bulgular

**Eksen küçük modellerde de var ve güçlü.** Rol vektörlerinin PC1'i ile kontrast vektörü arasındaki kosinüs, 56 katman ölçümünün tamamında 0,79'un üstünde. PC1'in uçları makalenin ayrımını yeniden üretiyor: bir uçta `consultant, assistant, planner`, diğer uçta `poet, leviathan, eldritch`.

**Varsayılan asistanın o eksende uçta durması ölçeğe bağlı.** 0,6B'de varsayılan persona uzayının ortasında ve hiçbir derinlikte uca yaklaşmıyor. 1,7B'de derinlikle uca kayıyor ve L19'da uç desile giriyor. Makalenin 27B–70B modellerinde orta katmanda zaten uçta. **A kriteri iki modelde de düştü** — düşme biçimleri farklı ve fark bulgunun kendisi.

**Eksende steering personayı nedensel olarak kontrol ediyor.** Assistant-dışı persona oranı L14'te %45,6 → %94,0 (+48,4 puan), L19'da %41,6 → %89,6 (+48,0 puan). Eşik 25 puandı. Doz-yanıt monoton, ve model bozulmuyor: anlamsız yanıt oranı hiçbir hücrede %5,6'yı geçmiyor. **B kriteri iki katmanda da geçti.**

**Uçta olunan katman ile müdahalenin çalıştığı katman aynı değil.** A kriterinde uçta olan L19'du; en güçlü nedensel kol L14'te çıktı. −0,2 gücünde L14 çoktan %68,8'e çıkmışken L19 %49,6'da. Makalenin bildirmediği bir ayrışma; yalnız iki katman birden ölçüldüğü için görünür oldu.

**Etki bu yöne özgü.** Aynı büyüklükte üç kontrol yönü — izotropik rastgele, eksenin koordinatları karıştırılmış, ve rol uzayı içinde eksene dik — hiçbiri etkiyi üretmiyor; üçünün de değişimi negatif (−14,8 · −3,6 · −25,7 puan). **C kriteri geçti.**

## Sonuç artefaktları

| Ne | Yol |
|---|---|
| A kriteri kararı | `results/models/<slug>/axis/criterion_a.json` |
| B kriteri kararı | `results/models/qwen3-1.7b/steering/criterion_b.json` |
| C kriteri kararı | `results/models/qwen3-1.7b/steering/criterion_c.json` |
| Doz-yanıt eğrileri | `results/models/qwen3-1.7b/steering/rate_by_strength*.json` |
| Katman taraması | `results/models/<slug>/axis/layer_sweep.json` |
| Ölçek karşılaştırması | `results/scale_comparison.json` |

**Ön-tesciller** — üçü de ilgili koddan ve ölçümden önce commit'lendi:
[`scale_hypothesis`](results/scale_hypothesis_preregistration.json) ·
[`steering`](results/steering_preregistration.json) ·
[`control`](results/control_preregistration.json)

📄 **[Tam çalışma raporu: docs/rapor.md](docs/rapor.md)** — yöntem, makaleden sapmalar, sınırlamalar, ve süreçte yakalanan hatalar.

## Yeniden üretme

```bash
uv sync --extra dev --extra ml
uv run --extra dev pytest -q            # 664 test, ağa çıkmaz
```

Hakem çağrıları üç ortam değişkeni ister; hiçbiri depoda sabitlenmez:

```bash
export AAX_GATEWAY_BASE_URL="https://<gateway-adresiniz>/<uygulama>"
export AAX_GATEWAY_MODEL="<hakem-model-adı>"
export APP_KEY_JAILBREAK="<anahtar>"       # hiçbir dosyaya yazılmaz
```

Boru hattı (hedef model `AAX_TARGET_MODEL` ile seçilir):

```bash
# Aşama 0-3 — eksen çıkarımı
AAX_TARGET_MODEL="Qwen/Qwen3-1.7B" uv run --extra gen python scripts/04_generate_rollouts.py
AAX_TARGET_MODEL="Qwen/Qwen3-1.7B" uv run --extra ml  python scripts/05_capture_activations.py --checkpoint-every 250
AAX_TARGET_MODEL="Qwen/Qwen3-1.7B" uv run --extra ml  python scripts/06_label_and_train_probe.py
AAX_TARGET_MODEL="Qwen/Qwen3-1.7B" uv run --extra ml  python scripts/07_extract_axis.py

# Aşama 4 — steering (GPU, ağa çıkmaz) ve değerlendirme (hakem, torch istemez)
AAX_TARGET_MODEL="Qwen/Qwen3-1.7B" uv run --extra ml python scripts/08_steering_sweep.py --layers 14 19
AAX_TARGET_MODEL="Qwen/Qwen3-1.7B" uv run python scripts/09_evaluate_steering.py --dry-run
AAX_TARGET_MODEL="Qwen/Qwen3-1.7B" uv run python scripts/09_evaluate_steering.py

# Kontrol yönleri — C kriteri
for k in gaussian shuffled rolespan; do
  AAX_TARGET_MODEL="Qwen/Qwen3-1.7B" uv run --extra ml python scripts/08_steering_sweep.py \
    --layers 14 --strengths -0.6 -0.4 -0.2 --direction $k --variant $k --seed 0
  AAX_TARGET_MODEL="Qwen/Qwen3-1.7B" uv run python scripts/09_evaluate_steering.py --variant $k
done
AAX_TARGET_MODEL="Qwen/Qwen3-1.7B" uv run python scripts/10_evaluate_controls.py
```

Tüm ölçümler tek bir RTX 4060'ta (8 GB) ve toplam 1.191 hakem çağrısıyla yapıldı.

## Tasarım notları

**Çıkış kodları anlamlıdır.** Karar üreten script'lerde: **0** = kriter geçti · **1** = kriter değerlendirildi ve düştü (gerçek bir negatif sonuç) · **2** = koşu karar üretemedi. Çökme asla 1 dönmez — bir dosya eksikliği ile bir bilimsel bulgu karıştırılamaz.

**Bütçe koruması istemci tarafındadır.** Hakem paylaşımlı bir çıkarım sunucusu olduğu için tüm koruma `src/aax/gateway.py` içindedir: endpoint başına paylaşımlı 1 istek/sn, süreçler arası kilitli ve atomik yazımlı bütçe sayacı, **global 1500 gönderim tavanı**, devre kesici, içerik cache'i. Dışarı giden her çağrı oradan geçer.

**Testler ağa yapısal olarak çıkamaz.** `tests/conftest.py` soket çağrılarını ve DNS çözümlemesini kapatır, `HF_HUB_OFFLINE=1` ayarlar.

**Artefaktlar modele göre kapsamlıdır** (`data/models/<slug>/`, `results/models/<slug>/`), ve etiket dosyaları üretildikleri sweep'in sha256'sını taşır — bayat etiketler yeniden üretilmiş bir sweep'e uygulanamaz.

## Lisans

Apache-2.0 — bkz. [LICENSE](LICENSE).
