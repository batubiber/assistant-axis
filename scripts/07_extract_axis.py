#!/usr/bin/env python3
"""Aşama 3 — rol vektörleri, PCA, Assistant Axis, A kriteri.

Kullanım:
    uv run --extra ml python scripts/07_extract_axis.py
"""
from __future__ import annotations

import json
import sys

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


def main() -> int:
    # `mmap_mode="r"`: planlanan ölçekte (16.000 × 28 × 2048 float32) bu
    # dosya ~3.5 GB'dir. Tam yükleme onu belleğe kopyalar; aşağıdaki
    # `acts[role_idx]`/`acts[default_idx]` fantezi indekslemesi zaten SEÇİLEN
    # satırların bir kopyasını (~3.7 GB'a kadar) çıkarır. mmap ile yalnızca
    # seçilen satırlar maddîleşir — dosyanın tamamı iki kez belleğe alınmaz.
    acts = np.load(config.DATA_DIR / "activations.npy", mmap_mode="r")
    index = json.loads((config.DATA_DIR / "activations_index.json").read_text(encoding="utf-8"))

    # 06_label_and_train_probe.py ile aynı desen: brief'in Adım 5 kod bloğunda
    # bu sarmalayıcı yoktu (çıplak `.read_text()["expression"]`), ama görev
    # tanımı "eksikse temiz başarısız olsun, traceback değil" diyor — dosya
    # `role_expression.json`'ı henüz üretmemiş bir operatör için çıplak
    # FileNotFoundError traceback'i bu koşulu karşılamıyor.
    try:
        expression = json.loads(ROLE_EXPRESSION_PATH.read_text(encoding="utf-8"))["expression"]
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
    if len(names) < 40:
        print(f"UYARI: yalnızca {len(names)} vektör — PCA kararsız olabilir.")

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
    print()
    print("PC1'in bir ucu:", [names[i] for i in order[:6]])
    print("PC1'in diğer ucu:", [names[i] for i in order[-6:]])
    print()
    print("A KRİTERİ:", "GEÇTİ" if verdict["passed"] else "DÜŞTÜ")
    print(" ", verdict["reason"])
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
