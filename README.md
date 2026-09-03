# assistant-axis

*The Assistant Axis* (arXiv:2601.10387) makalesinin küçük modellerde replikasyonu — tek bir 8 GB tüketici GPU'sunda.

**Bulgu:** Makalenin persona ekseni Qwen3-1.7B ve Qwen3-0.6B'de de var ve güçlü (her katmanda `|cos| = 0.82-0.96`). Ölçeğe bağlı olan şey eksenin varlığı değil, **varsayılan Assistant'ın o eksen üzerinde uç noktada olup olmadığı**: 0.6B'de varsayılan persona uzayının ortasında ve hiçbir derinlikte uca yaklaşmıyor; 1.7B'de derinlikle uca kayıp L19'da giriyor; makalenin 27-70B modellerinde orta katmanda zaten uçta.

Önceden tescillenmiş A kriteri iki modelde de **düştü** — ama düşme biçimleri farklı ve aradaki fark bulgunun kendisi.

📄 **[Çalışma raporu: docs/rapor.md](docs/rapor.md)** — yöntem, sonuçlar, sapmalar, sınırlamalar, süreçte yakalanan hatalar.

## Hızlı başlangıç

```bash
uv sync --extra dev --extra ml          # aktivasyon hattı
uv run --extra dev pytest -q            # 584 test, ağa çıkmaz
```

Boru hattı (hedef model `AAX_TARGET_MODEL` ile seçilir):

```bash
AAX_TARGET_MODEL="Qwen/Qwen3-1.7B" uv run --extra gen python scripts/04_generate_rollouts.py
AAX_TARGET_MODEL="Qwen/Qwen3-1.7B" uv run --extra ml  python scripts/05_capture_activations.py --checkpoint-every 250
AAX_TARGET_MODEL="Qwen/Qwen3-1.7B" uv run --extra ml  python scripts/06_label_and_train_probe.py
AAX_TARGET_MODEL="Qwen/Qwen3-1.7B" uv run --extra ml  python scripts/07_extract_axis.py
```

Aşama 4 — steering (B kriteri). Sweep GPU'da koşar ve **ağa çıkmaz**; değerlendirme
hakemi çağırır ve `torch` istemez:

```bash
AAX_TARGET_MODEL="Qwen/Qwen3-1.7B" uv run --extra ml python scripts/08_steering_sweep.py --layers 14 19
AAX_TARGET_MODEL="Qwen/Qwen3-1.7B" uv run python scripts/09_evaluate_steering.py --dry-run
AAX_TARGET_MODEL="Qwen/Qwen3-1.7B" uv run python scripts/09_evaluate_steering.py
```

B kriteri **ölçümden önce** tescillendi: [results/steering_preregistration.json](results/steering_preregistration.json).

Hakem çağrıları üç ortam değişkeni ister; hiçbiri depoda sabitlenmez:

```bash
export AAX_GATEWAY_BASE_URL="https://<gateway-adresiniz>/<uygulama>"
export AAX_GATEWAY_MODEL="<hakem-model-adı>"
export APP_KEY_JAILBREAK="<anahtar>"       # hiçbir dosyaya yazılmaz
```

## Uyarı: paylaşımlı endpoint

Hakem, paylaşımlı bir çıkarım sunucusudur ve hız sınırlama **tamamen istemci tarafındadır**. Bu yüzden tüm koruma `src/aax/gateway.py` içindedir: endpoint başına paylaşımlı 1 istek/sn, süreçler arası kilitli ve atomik yazımlı bütçe sayacı, **global 1500 gönderim tavanı**, devre kesici, içerik cache'i. Dışarı giden her çağrı oradan geçer.

## Çıkış kodları

`07_extract_axis.py` için anlamlıdır: **0** = kriter geçti · **1** = kriter değerlendirildi ve düştü (gerçek bir negatif bilimsel sonuç) · **2** = koşu karar üretemedi. Çökme asla 1 dönmez.
