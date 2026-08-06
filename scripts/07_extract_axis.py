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
    acts = np.load(config.DATA_DIR / "activations.npy")
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

    axis_per_layer = np.stack(
        [contrast_axis(default_mean_all[l], role_mean_all[l]) for l in range(acts.shape[1])]
    )

    cos_by_layer = []
    for layer in range(acts.shape[1]):
        components, _ = pca_components(vectors[:, layer, :], n_components=1)
        cos_by_layer.append(cosine(components[0], axis_per_layer[layer]))

    # Tam spektrum isteniyor: n_components_for_70pct yalnızca ilk 10 orandan
    # hesaplanırsa gerçek cevap 10'u aştığında doyuma ulaşıp hep 11 der ve
    # "persona uzayı düşük boyutlu" iddiasını yapay olarak destekler.
    # Raporlanan `explained_variance_ratio` yine ilk 10 bileşendir.
    components_mid, ratios_full = pca_components(
        vectors[:, middle, :], n_components=vectors.shape[0]
    )
    ratios_mid = ratios_full[:10]
    pc1 = components_mid[0]
    role_projections = vectors[:, middle, :] @ pc1
    default_projection = float(default_mean_all[middle] @ pc1)
    percentile = projection_percentile(default_projection, role_projections)

    verdict = evaluate_criterion_a(cos_by_layer[middle], percentile)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "assistant_axis.npy", axis_per_layer)
    np.save(OUT_DIR / "role_vectors.npy", vectors)
    (OUT_DIR / "role_names.json").write_text(json.dumps(names, ensure_ascii=False), encoding="utf-8")
    n_for_70 = n_components_for_variance(ratios_full, 0.70)
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
                # Tam spektrum eşiğe hiç ulaşmazsa kesin sayı yerine alt sınır
                # yazılır — desteklenemeyen bir sayı yazmaktansa ">k" dürüsttür.
                "n_components_for_70pct": (
                    n_for_70 if n_for_70 is not None else f">{len(ratios_full)}"
                ),
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
