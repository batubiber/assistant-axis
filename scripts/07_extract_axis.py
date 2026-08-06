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

    categories = [expression.get(str(i), "no") for i in role_idx]
    vectors, names = role_vectors(
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

    fully_rows = [i for i, c in zip(role_idx, categories) if c == "fully"]
    role_mean_all = acts[fully_rows].astype(np.float64).mean(axis=0)  # [L, D]

    axis_per_layer = np.stack(
        [contrast_axis(default_mean_all[l], role_mean_all[l]) for l in range(acts.shape[1])]
    )

    cos_by_layer = []
    for layer in range(acts.shape[1]):
        components, _ = pca_components(vectors[:, layer, :], n_components=1)
        cos_by_layer.append(cosine(components[0], axis_per_layer[layer]))

    components_mid, ratios_mid = pca_components(vectors[:, middle, :], n_components=10)
    pc1 = components_mid[0]
    role_projections = vectors[:, middle, :] @ pc1
    default_projection = float(default_mean_all[middle] @ pc1)
    percentile = projection_percentile(default_projection, role_projections)

    verdict = evaluate_criterion_a(cos_by_layer[middle], percentile)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "assistant_axis.npy", axis_per_layer)
    np.save(OUT_DIR / "role_vectors.npy", vectors)
    (OUT_DIR / "role_names.json").write_text(json.dumps(names, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / "criterion_a.json").write_text(
        json.dumps(
            {
                **verdict,
                "middle_layer": middle,
                "n_role_vectors": len(names),
                "cos_by_layer": cos_by_layer,
                "explained_variance_ratio": ratios_mid.tolist(),
                "n_components_for_70pct": int(np.searchsorted(np.cumsum(ratios_mid), 0.70) + 1),
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
