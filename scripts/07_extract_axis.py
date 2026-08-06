#!/usr/bin/env python3
"""Aşama 3 — rol vektörleri, PCA, Assistant Axis, A kriteri.

Kullanım:
    uv run --extra ml python scripts/07_extract_axis.py
    uv run --extra ml python scripts/07_extract_axis.py --min-role-vectors 20  # bilinçli gevşetme

ÇIKIŞ KODLARI — bu script'in tek ürünü projenin ön kaydedilmiş hükmüdür,
bu yüzden kodların anlamı sözleşmedir:

    0  A KRİTERİ GEÇTİ
    1  A KRİTERİ DÜŞTÜ — gerçek, değerlendirilmiş bir bilimsel sonuç
    2  BAŞARISIZ — karar ÜRETİLEMEDİ (eksik/bayat/tanımsız girdi, çökme)

`1` yalnızca `evaluate_criterion_a` fiilen çalışıp `passed: False` dediğinde
üretilir. Başka HİÇBİR yol 1 döndürmez: `main()` gövdesinin tamamı sarılıdır
ve yakalanmamış her istisna 2'ye çevrilir. Sarmalayıcı olmadan, eksik bir
`activations.npy` (FileNotFoundError) ya da indeks/aktivasyon boyu
uyuşmazlığı (IndexError) yorumlayıcıyı çıkış 1 ile döndürüyordu — yani bir
ÇÖKME, "Assistant Axis 1.7B'de oluşmuyor" bulgusundan ayırt edilemiyordu.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback

import numpy as np

from aax import config
from aax.axis import (
    contrast_axis,
    cosine,
    evaluate_criterion_a,
    n_components_for_variance,
    pca_components,
    projection_percentile,
    role_vectors,
)

OUT_DIR = config.RESULTS_DIR / "axis"

ROLE_EXPRESSION_PATH = config.DATA_DIR / "role_expression.json"

# Spec Bölüm 9'un "<40 rol kalıyor" riski ve Bölüm 5/Aşama 2'nin kuralı ile
# aynı sayı. Bu bir uyarı eşiği DEĞİL, sert bir tabandır: `n` rol vektörüyle
# persentil yalnızca `k/n` değerlerini alabilir, yani küçük `n`'de uç bir
# desil neredeyse otomatiktir. Ölçüldü: 2 rol vektörü ve onların span'ı
# dışındaki bir default ile A kriteri `passed: True, cos_magnitude: 0.99995`
# üretiyordu. Böyle bir "GEÇTİ" veriyi değil, örneklem büyüklüğünü ölçer.
MIN_ROLE_VECTORS = 40


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--min-role-vectors",
        type=int,
        default=MIN_ROLE_VECTORS,
        help=(
            "A kriterinin değerlendirileceği asgari rol vektörü sayısı "
            f"(varsayılan {MIN_ROLE_VECTORS}, spec Bölüm 9); altında çıkış 2 — "
            "yalnızca bilinçli bir gevşetme için düşürün"
        ),
    )
    return parser


def _run(argv: list[str] | None) -> int:
    args = build_arg_parser().parse_args(argv)

    acts_path = config.DATA_DIR / "activations.npy"
    index_path = config.DATA_DIR / "activations_index.json"

    # `role_expression.json` için zaten var olan desenin AYNISI, Aşama 1'in
    # iki çıktısı için. Bu iki dosya eksikken script çıplak bir
    # FileNotFoundError ile çöküyordu ve yorumlayıcı çıkış 1 döndürüyordu —
    # "A KRİTERİ DÜŞTÜ" ile aynı kod (bkz. modül docstring'i).
    #
    # `mmap_mode="r"`: planlanan ölçekte (16.000 × 28 × 2048 float32) bu
    # dosya ~3.5 GB'dir. Tam yükleme onu belleğe kopyalar; aşağıdaki
    # `acts[role_idx]`/`acts[default_idx]` fantezi indekslemesi zaten SEÇİLEN
    # satırların bir kopyasını (~3.7 GB'a kadar) çıkarır. mmap ile yalnızca
    # seçilen satırlar maddîleşir — dosyanın tamamı iki kez belleğe alınmaz.
    try:
        acts = np.load(acts_path, mmap_mode="r")
    except FileNotFoundError:
        print(
            f"BAŞARISIZ: {acts_path} yok.\n"
            "  Bu dosya Aşama 1'in çıktısıdır — önce "
            "scripts/05_capture_activations.py çalıştırılmalı.",
            file=sys.stderr,
        )
        return 2
    except ValueError as exc:
        print(
            f"BAŞARISIZ: {acts_path} okunamadı — bozuk veya kırpık .npy.\n"
            f"  Ayrıntı: {exc}\n"
            "  scripts/05_capture_activations.py'yi tekrar çalıştırıp dosyayı yeniden üretin.",
            file=sys.stderr,
        )
        return 2
    if acts.ndim != 3:
        print(
            f"BAŞARISIZ: {acts_path} beklenen [satır, katman, d_model] şeklinde değil "
            f"(shape={tuple(acts.shape)}).\n"
            "  scripts/05_capture_activations.py'yi tekrar çalıştırıp dosyayı yeniden üretin.",
            file=sys.stderr,
        )
        return 2

    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(
            f"BAŞARISIZ: {index_path} yok.\n"
            "  Bu dosya Aşama 1'in çıktısıdır — önce "
            "scripts/05_capture_activations.py çalıştırılmalı.\n"
            "  activations.npy tek başına yetmez: hangi satırın hangi role/kind'a ait "
            "olduğu yalnızca bu indekste yazar.",
            file=sys.stderr,
        )
        return 2
    except json.JSONDecodeError as exc:
        print(
            f"BAŞARISIZ: {index_path} bozuk — JSON ayrıştırılamadı.\n"
            f"  Ayrıntı: {exc}\n"
            "  scripts/05_capture_activations.py'yi tekrar çalıştırıp dosyayı yeniden üretin.",
            file=sys.stderr,
        )
        return 2

    missing_keys = [k for k in ("rows", "middle_layer") if k not in index]
    if missing_keys:
        print(
            f"BAŞARISIZ: {index_path} eksik anahtar içeriyor: {missing_keys}.\n"
            "  scripts/05_capture_activations.py'yi tekrar çalıştırıp dosyayı yeniden üretin.",
            file=sys.stderr,
        )
        return 2

    # 06_label_and_train_probe.py ile aynı desen: brief'in Adım 5 kod bloğunda
    # bu sarmalayıcı yoktu (çıplak `.read_text()["expression"]`), ama görev
    # tanımı "eksikse temiz başarısız olsun, traceback değil" diyor — dosya
    # `role_expression.json`'ı henüz üretmemiş bir operatör için çıplak
    # FileNotFoundError traceback'i bu koşulu karşılamıyor.
    try:
        expression_payload = json.loads(ROLE_EXPRESSION_PATH.read_text(encoding="utf-8"))
        expression = expression_payload["expression"]
    except FileNotFoundError:
        print(
            f"BAŞARISIZ: {ROLE_EXPRESSION_PATH} yok.\n"
            "  Bu dosya Aşama 2'nin çıktısıdır — önce "
            "scripts/06_label_and_train_probe.py çalıştırılmalı.",
            file=sys.stderr,
        )
        return 2
    except (json.JSONDecodeError, KeyError) as exc:
        print(
            f"BAŞARISIZ: {ROLE_EXPRESSION_PATH} bozuk veya 'expression' anahtarı yok.\n"
            f"  Ayrıntı: {exc}\n"
            "  scripts/06_label_and_train_probe.py'yi tekrar çalıştırıp dosyayı yeniden üretin.",
            file=sys.stderr,
        )
        return 2

    rows = index["rows"]
    middle = index["middle_layer"]

    # Bütünlük alanları YAZILIYOR ama OKUNMUYORDU. `n_rows` ile gerçek satır
    # sayısının ayrışması, `rows`'un aktivasyon matrisinden uzun olması ve
    # `middle_layer`'ın katman sayısını aşması — üçü de indeksin
    # aktivasyonlarla aynı koşudan gelmediğinin işaretidir ve üçü de
    # kontrolsüz bırakıldığında bir `IndexError` ile (yani çıkış 1 ile)
    # patlıyordu. `IndexError` bir `ValueError` DEĞİLDİR, sayısal bloğun
    # sarmalayıcısına da takılmıyordu.
    if index.get("n_rows") != int(acts.shape[0]):
        print(
            "BAŞARISIZ: activations_index.json ile activations.npy uyuşmuyor —\n"
            f"  indeks n_rows={index.get('n_rows')}, matris {acts.shape[0]} satır.\n"
            "  İki dosya farklı koşulardan geliyor; scripts/05_capture_activations.py'yi "
            "tekrar çalıştırıp ikisini birlikte yeniden üretin.",
            file=sys.stderr,
        )
        return 2
    if len(rows) != int(acts.shape[0]):
        print(
            "BAŞARISIZ: activations_index.json 'rows' uzunluğu matris satır sayısıyla "
            "uyuşmuyor —\n"
            f"  {len(rows)} indeks satırı, {acts.shape[0]} aktivasyon satırı.\n"
            "  scripts/05_capture_activations.py'yi tekrar çalıştırıp ikisini birlikte "
            "yeniden üretin.",
            file=sys.stderr,
        )
        return 2
    if not isinstance(middle, int) or not 0 <= middle < int(acts.shape[1]):
        print(
            f"BAŞARISIZ: middle_layer={middle!r} katman aralığının dışında "
            f"(0..{int(acts.shape[1]) - 1}).\n"
            "  scripts/05_capture_activations.py'yi tekrar çalıştırıp indeksi yeniden üretin.",
            file=sys.stderr,
        )
        return 2

    # Bayatlık kontrolünün ÜÇÜNCÜSÜ ve en keskini: iki dosyanın içerikten
    # türetilen koşu kimliği. Aşağıdaki sayı/kapsama kontrolleri, Aşama 1
    # aynı satır sayısı ve aynı sırayla FARKLI bir rol kümesiyle yeniden
    # koşturulduğunda ikisi de geçerdi — kimlik karşılaştırması geçmez.
    index_run_id = index.get("run_id")
    expression_run_id = expression_payload.get("run_id")
    if index_run_id is None or expression_run_id is None or index_run_id != expression_run_id:
        print(
            "BAŞARISIZ: activations_index.json ile role_expression.json aynı koşudan "
            "gelmiyor —\n"
            f"  activations_index.json run_id={index_run_id!r}, "
            f"role_expression.json run_id={expression_run_id!r}.\n"
            "  Kimlik içerikten türetilir (rollouts.jsonl satırları): eşleşmiyorsa iki "
            "dosya farklı rollout kümelerinden geliyor demektir. Satır sayısı ve sıra "
            "tutsa bile rol kümesi değişmiş olabilir — bu, sessizce yanlış "
            "fully/somewhat ayrımı demektir.\n"
            "  `None` görüyorsanız dosya kimlik alanı yazmayan eski bir sürümden kalmış: "
            "scripts/05_capture_activations.py ve scripts/06_label_and_train_probe.py'yi "
            "güncel rollouts.jsonl ile tekrar çalıştırın.",
            file=sys.stderr,
        )
        return 2

    print(f"{acts.shape[0]} satır, {acts.shape[1]} katman, orta katman {middle}")

    role_idx = [i for i, r in enumerate(rows) if r["kind"] == "role"]
    default_idx = [i for i, r in enumerate(rows) if r["kind"] == "default"]

    # `len(names) == 0` koruması (altta) "fully" tarafındaki boş-dilim NaN'ını
    # yakalıyordu; bu onun default tarafındaki İKİZİ. `acts[default_idx]` boş
    # bir dizi olursa `.mean(axis=0)` yalnızca bir RuntimeWarning ile NaN
    # döner (bkz. `default_mean_all` altta) — o NaN korumasız bırakılırsa
    # `contrast_axis`'e kadar sessizce yayılır, orada `ValueError` fırlatır,
    # bu da yakalanmazsa yorumlayıcı çıkış kodu 1 ile döner. Çıkış 1, "A
    # KRİTERİ DÜŞTÜ" anlamına gelen koddur — bir çökme asla bilimsel bir
    # sonuç olarak kaydedilemez, bu yüzden burada erkenden, ucuzca kontrol
    # edilir.
    if len(default_idx) == 0:
        print(
            "BAŞARISIZ: activations_index.json içinde 'default' türünde hiç satır "
            "yok — default ortalaması tanımsız.\n"
            "  Assistant Axis mean(default) − mean(fully rol vektörleri) olarak "
            "tanımlı; default satır yoksa hesaplanacak bir şey yok.\n"
            "  Kontrol edin: scripts/04_generate_rollouts.py'nin default rollout'ları "
            "ürettiğini ve scripts/05_capture_activations.py'nin bunları "
            "activations_index.json'a yazdığını.\n"
            "  Bu bir BAŞARISIZLIKTIR, A kriteri kararı DEĞİLDİR: tanımsız veriden "
            "GEÇTİ/DÜŞTÜ çıkarılamaz.",
            file=sys.stderr,
        )
        return 2

    # Bayatlık kontrolü: `expression.get(str(i), "no")` eşleşmeyen her satırı
    # sessizce "no" sayar. Tek bir taze koşuda sorun değil, ama farklı bir
    # --limit veya rol kümesiyle üretilmiş eski bir role_expression.json Aşama 1
    # yeniden koşturulduktan sonra yerinde kalırsa fully/somewhat ayrımı kısmen
    # kayar ve hiçbir hata vermez. İki ucuz kontrol bunu yakalar: anahtar sayısı
    # ve anahtarların indeksteki rol satırlarını kapsaması.
    if len(expression) != len(role_idx):
        print(
            "BAŞARISIZ: role_expression.json ile activations_index.json uyuşmuyor —\n"
            f"  ifade haritasında {len(expression)} anahtar var, "
            f"indekste {len(role_idx)} rol satırı.\n"
            "  Olası neden: farklı bir --limit veya rol kümesiyle üretilmiş eski bir "
            "role_expression.json, Aşama 1 yeniden koşturulduktan sonra yerinde kalmış.\n"
            "  scripts/06_label_and_train_probe.py'yi güncel rollouts/aktivasyonlarla "
            "yeniden çalıştırın.",
            file=sys.stderr,
        )
        return 2
    missing = [i for i in role_idx if str(i) not in expression]
    if missing:
        print(
            "BAŞARISIZ: role_expression.json indeksteki bazı rol satırlarını kapsamıyor —\n"
            f"  {len(missing)} satırın karşılığı yok (ilk örnekler: {missing[:5]}).\n"
            "  Anahtar sayısı tutsa bile satır numaraları kaymış: iki dosya farklı "
            "koşulardan geliyor.\n"
            "  scripts/06_label_and_train_probe.py'yi güncel rollouts/aktivasyonlarla "
            "yeniden çalıştırın.",
            file=sys.stderr,
        )
        return 2

    categories = [expression[str(i)] for i in role_idx]
    vectors, names, vector_categories = role_vectors(
        acts[role_idx], [rows[i]["role"] for i in role_idx], categories
    )
    print(f"{len(names)} rol vektörü hesaplandı (>=10 yanıt kuralı sonrası)")

    # role_vectors, hiçbir (rol, kategori) çifti min_responses eşiğini
    # geçemezse boş dizi döner. Bu durumda devam etmek NaN'lara (boş dilimin
    # ortalaması) ve ardından PCA/kosinüs adımlarında şifreli bir
    # IndexError'a yol açar — görev tanımının "traceback değil, temiz mesaj"
    # koşulu burada da geçerli.
    if len(names) == 0:
        print(
            "BAŞARISIZ: hiçbir rol min_responses (>=10) eşiğini geçemedi — "
            "hesaplanacak rol vektörü yok.\n"
            "  Olası neden: role_expression.json'daki dağılım beklenenden "
            "farklı veya veri seti hâlâ pilot ölçekte (--limit).",
            file=sys.stderr,
        )
        return 2
    # Eskiden bu bir UYARI'ydı ve koşu devam ediyordu. Uyarı yeterli değil:
    # `n` rol vektörüyle persentil yalnızca `k/n` değerlerini alabilir, yani
    # küçük `n`'de "uç desil" koşulu neredeyse otomatik sağlanır. Ölçüldü
    # (bu düzeltmenin öncesi): 2 rol vektörü ve span dışı bir default ile
    # `passed: True`, `cos_magnitude: 0.99995`, çıkış kodu 0 — yani ön
    # kaydedilmiş bir hipotez, iki noktadan "doğrulanmış" oluyordu. Bu bir
    # DÜŞTÜ de değildir (veri kriteri değerlendirmeye elverişli değil):
    # çıkış 2.
    if len(names) < args.min_role_vectors:
        print(
            f"BAŞARISIZ: yalnızca {len(names)} rol vektörü var, en az "
            f"{args.min_role_vectors} gerekiyor.\n"
            "  Bu bir A kriteri kararı DEĞİLDİR: bu kadar az vektörle PCA kararsızdır ve "
            f"persentil yalnızca k/{len(names)} değerlerini alabildiği için uç desil "
            "koşulu neredeyse otomatik sağlanır — ölçülen şey veri değil, örneklem "
            "büyüklüğü olur.\n"
            "  Spec Bölüm 9 bu durumu ('<40 rol kalıyor') bir risk olarak adlandırır ve "
            "çıkış yolunu tanımlar: 'fully' yerine 'somewhat' vektörleriyle devam etmek "
            "ya da daha büyük bir modele geçmek.\n"
            "  Tabanı bilinçli olarak düşürmek için: --min-role-vectors N.",
            file=sys.stderr,
        )
        return 2

    default_mean_all = acts[default_idx].astype(np.float64).mean(axis=0)  # [L, D]

    # Assistant Axis = mean(default) − mean(fully ROL VEKTÖRLERİ). Ham "fully"
    # satırlarını havuzlamak İKİ ayrı hata olurdu: (1) "fully" sayısı 10'un
    # altında kalan bir rol role_vectors tarafından elenmişken ham satırlarıyla
    # yine de ortalamaya karışırdı; (2) çok rollout'lu roller ortalamayı ele
    # geçirirdi — oysa tanım her nitelikli rolün EŞİT ağırlıkta katkısını ister.
    fully_positions = [i for i, c in enumerate(vector_categories) if c == "fully"]
    if not fully_positions:
        print(
            "BAŞARISIZ: hiçbir rol vektörü 'fully' kategorisinde değil — "
            "Assistant Axis tanımsız.\n"
            "  Eksen mean(default) − mean(fully rol vektörleri) olarak tanımlı; "
            "fully rol vektörü yoksa hesaplanacak bir şey yok.\n"
            "  Kontrol edin: role_expression.json'daki dağılım (kaç satır 'fully'?), "
            ">=10 yanıt kuralı (bir rolün 'fully' sayısı eşiğin altında kalmış "
            "olabilir) ve veri setinin hâlâ pilot ölçekte (--limit) olup olmadığı.\n"
            "  Bu bir BAŞARISIZLIKTIR, A kriteri kararı DEĞİLDİR: tanımsız veriden "
            "GEÇTİ/DÜŞTÜ çıkarılamaz.",
            file=sys.stderr,
        )
        return 2
    role_mean_all = vectors[fully_positions].astype(np.float64).mean(axis=0)  # [L, D]
    print(f"  bunların {len(fully_positions)} tanesi 'fully' — eksen bunlardan hesaplanıyor")

    # Sayısal adımların TAMAMI bu blokta: `contrast_axis`/`cosine`/
    # `n_components_for_variance` sonlu olmayan (NaN/inf) bir değere ya da
    # sıfır normlu bir kontrast vektörüne çarparsa `ValueError` fırlatır
    # (bkz. `aax.axis`). Yukarıdaki iki koruma (`default_idx` boş, hiç
    # `fully` yok) en olası iki NaN kaynağını erkenden kapatıyor, ama aynı
    # çarpışma başka yollardan da gerçekleşebilir — ör. default ve fully
    # ortalamaları TESADÜFEN eşitse (sıfır normlu kontrast) ya da
    # `activations.npy` içinde başka bir nedenle bozuk (NaN/inf) bir satır
    # varsa. Yakalanmayan bir `ValueError` burada yorumlayıcıyı çıkış kodu
    # 1 ile döndürür — "A KRİTERİ DÜŞTÜ" anlamına gelen kod. Bir çökme asla
    # bilimsel bir sonuç olarak kaydedilemez; bu yüzden buradan itibaren her
    # şey yakalanır ve çıkış 2'ye (BAŞARISIZ, karar DEĞİL) çevrilir. Hiçbir
    # artefakt bu blok TAMAMLANMADAN (yani her değer başarıyla
    # hesaplanmadan) diske YAZILMAZ — aksi hâlde geç bir raise, önceki bir
    # koşudan kalma `criterion_a.json` yanında yarım `assistant_axis.npy` /
    # `role_vectors.npy` bırakabilirdi ve `results/` commit'lendiği için bu
    # tutarsız kombinasyon depoda kalıcı hâle gelirdi.
    try:
        axis_per_layer = np.stack(
            [contrast_axis(default_mean_all[l], role_mean_all[l]) for l in range(acts.shape[1])]
        )

        cos_by_layer = []
        for layer in range(acts.shape[1]):
            components, _ = pca_components(vectors[:, layer, :], n_components=1)
            cos_by_layer.append(cosine(components[0], axis_per_layer[layer]))

        # Tam spektrum isteniyor: n_components_for_70pct yalnızca ilk 10
        # orandan hesaplansaydı gerçek cevap 10'u aştığında doyuma ulaşıp
        # hep 11 derdi ve "persona uzayı düşük boyutlu" iddiasını yapay
        # olarak destekler. Raporlanan `explained_variance_ratio` yine ilk
        # 10 bileşendir.
        components_mid, ratios_full = pca_components(
            vectors[:, middle, :], n_components=vectors.shape[0]
        )
        ratios_mid = ratios_full[:10]
        pc1 = components_mid[0]
        role_projections = vectors[:, middle, :] @ pc1
        default_projection = float(default_mean_all[middle] @ pc1)
        percentile = projection_percentile(default_projection, role_projections)

        verdict = evaluate_criterion_a(cos_by_layer[middle], percentile)

        # `ratios_full` burada TAM spektrumdur (`n_components=vectors.shape[0]`
        # yukarıda), yani toplamı her zaman 1.0'dır (PCA'nın tanımı gereği:
        # `ratios = variance / variance.sum()`) — kümülatif toplam %70 eşiğini
        # er ya da geç MUTLAKA aşar. `n_components_for_variance` yalnızca
        # KESİLMİŞ bir spektrum verildiğinde `None` dönebilir (bkz. modül
        # docstring'i); burada asla olmamalı. Olursa yukarıdaki varsayım bir
        # yerde bozulmuş demektir — sessizce `None`/">k" yazıp devam etmek
        # yerine (eski davranış) açıkça BAŞARISIZ olunur, `n_components_for_70pct`
        # alanı böylece tek bir tipte (`int`) kalır.
        n_for_70 = n_components_for_variance(ratios_full, 0.70)
        if n_for_70 is None:
            raise ValueError(
                "n_components_for_variance tam spektrumla None döndü — "
                "beklenmeyen durum, tam spektrumun toplamı 1.0 olmalıydı"
            )
    except ValueError as exc:
        print(
            "BAŞARISIZ: sayısal hesaplama sonlu olmayan (NaN/inf) bir değere ya da "
            "sıfır normlu bir vektöre çarptı.\n"
            f"  Ayrıntı: {exc}\n"
            "  Olası neden: default veya fully rol ortalaması boş bir dilimden "
            "geliyor, ikisi tesadüfen eşit ya da activations.npy'de bozuk bir satır "
            "var.\n"
            "  Bu bir BAŞARISIZLIKTIR, A kriteri kararı DEĞİLDİR: tanımsız veriden "
            "GEÇTİ/DÜŞTÜ çıkarılamaz. Hiçbir artefakt yazılmadı.",
            file=sys.stderr,
        )
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "assistant_axis.npy", axis_per_layer)
    np.save(OUT_DIR / "role_vectors.npy", vectors)
    (OUT_DIR / "role_names.json").write_text(json.dumps(names, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / "criterion_a.json").write_text(
        json.dumps(
            {
                **verdict,
                # Kaynak künyesi: bu dosya aylar sonra tek başına okunacak.
                # Saatten türetilen zaman damgası YOK — bu repo kimlikleri
                # içerikten türetir.
                "model": index.get("model"),
                "run_id": index.get("run_id"),
                "n_layers": int(acts.shape[1]),
                "d_model": int(acts.shape[2]),
                "middle_layer": middle,
                "n_role_vectors": len(names),
                "n_fully_role_vectors": len(fully_positions),
                "cos_by_layer": cos_by_layer,
                "explained_variance_ratio": ratios_mid.tolist(),
                # `explained_variance_ratio` ilk 10 bileşene KESİLMİŞ,
                # `n_components_for_70pct` ise TAM spektruma karşı sayılır.
                # İkisi tek başına birbiriyle bağdaştırılamıyordu: 12 gören
                # bir okur, listedeki 10 oranın toplamının %70'in altında mı
                # üstünde mi olduğunu göremezdi. Bu alan köprüyü kurar —
                # `cumulative_variance_at_10 < 0.70` ise `n > 10` zorunludur.
                "cumulative_variance_at_10": float(ratios_mid.sum()),
                "n_components_for_70pct": n_for_70,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    order = np.argsort(role_projections)
    print()
    print(f"PC1 varyans oranı: {ratios_mid[0]:.1%}")
    print(f"cos(PC1, eksen) orta katmanda: {cos_by_layer[middle]:.3f}")
    print(f"default Assistant persentili: {percentile:.3f}")
    print(f"işaretin gerektirdiği desil: {verdict['required_decile']}")
    print()
    print("PC1'in bir ucu:", [names[i] for i in order[:6]])
    print("PC1'in diğer ucu:", [names[i] for i in order[-6:]])
    print()
    print("A KRİTERİ:", "GEÇTİ" if verdict["passed"] else "DÜŞTÜ")
    print(" ", verdict["reason"])
    return 0 if verdict["passed"] else 1


def main(argv: list[str] | None = None) -> int:
    """Tanı sarmalayıcısı — çıkış 1'i SADECE gerçek bir karara ayırır.

    `_run()`'ın gövdesindeki her adım kendi Türkçe tanısını ve çıkış kodunu
    üretir; buradaki `except Exception` yalnızca ÖNGÖRÜLMEMİŞ bir çökme için
    vardır. Bu sarmalayıcı olmadan böyle bir çökme yorumlayıcıyı çıkış 1 ile
    döndürüyordu (ölçüldü: `activations.npy` yokken `FileNotFoundError`,
    indeks satırları matristen uzunken `IndexError`) — ve 1, bu projede
    "A KRİTERİ DÜŞTÜ" demektir. Bir çökmenin bilimsel bir negatif sonuç gibi
    kaydedilmesi, bu dalın önlemek zorunda olduğu tek şeydir.

    `except Exception`, `SystemExit`/`KeyboardInterrupt` gibi
    `BaseException`'ları bilerek KAPSAMAZ: operatörün Ctrl-C'si bir
    "BAŞARISIZ" tanısına dönüşmemeli.
    """
    try:
        return _run(argv)
    except Exception as exc:  # noqa: BLE001 — kasıtlı geniş yakalama, gerekçe docstring'de
        print(
            "BAŞARISIZ: beklenmeyen bir hata yüzünden A kriteri değerlendirilemedi.\n"
            f"  Ayrıntı: {type(exc).__name__}: {exc}\n"
            "  Bu bir A kriteri kararı DEĞİLDİR (çıkış 1 değil 2): hesaplama hiç "
            "tamamlanmadı, GEÇTİ/DÜŞTÜ çıkarılamaz.\n"
            "  Tam iz aşağıda; girdileri (activations.npy, activations_index.json, "
            "role_expression.json) kontrol edip tekrar koşun.",
            file=sys.stderr,
        )
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
