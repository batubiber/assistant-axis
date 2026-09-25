#!/usr/bin/env python3
"""Varsayılan asistanın PC1 üzerindeki ARALIK TEMELLİ konumu (makale §2.3.1).

Bu betik yalnız OKUR: ne modele ne ağa ne de hakeme gider; mevcut
artefaktları değiştirmez. Tek çıktısı katman başına bir ölçüm dosyasıdır:

    results/models/<slug>/axis/range_position.json

NEDEN VAR
    Bu replikasyonun ön kayıtlı A kriteri SIRA temellidir: varsayılanın PC1
    izdüşümü rol izdüşümleri arasında asistan ucundaki %10'luk dilimde (uç
    desil) mi? (`aax.axis.evaluate_criterion_a`, `07_extract_axis.py`.)
    Makale ise konumu ARALIK temelli bir ölçüyle verir (arXiv:2601.10387v1,
    §2.3.1 "Projecting default activations into persona space"): varsayılanın
    her PC üzerindeki izdüşümünün, rol izdüşümlerinin en küçüğü (0) ile en
    büyüğü (1) arasındaki göreli konumu; PC1 için "herhangi bir uca en küçük
    uzaklık" 0,03 olarak raporlanır (model ayrımı verilmeden). İki ölçü aynı
    şeyi ölçmez: sıra ölçüsü kaç rolün varsayılanın ötesinde durduğunu,
    aralık ölçüsü varsayılanın uç role geometrik yakınlığını verir. Bu betik
    makalenin ölçüsünü bu replikasyonun verisiyle, iki model ve bütün
    katmanlar için hesaplar.

HESAP (katman l için, `07_extract_axis.py` ile birebir aynı PCA)
    pc1        = pca_components(role_vectors[:, l, :])[0]   (rollerin ortalaması çıkarılır)
    rol_iz     = role_vectors[:, l, :] @ pc1
    var_iz     = mean(default satırları)[l] @ pc1
    goreli     = (var_iz - min(rol_iz)) / (max(rol_iz) - min(rol_iz))   0 = min ucu, 1 = max ucu
    makale     = min(goreli, 1 - goreli)                                  herhangi bir uca uzaklık

    Ortak bir öteleme (ör. rollerin ortalamasının çıkarılması) bütün
    izdüşümleri aynı sabit kadar kaydırır; göreli konum bundan etkilenmez.

ASİSTAN UCU
    SVD bir bileşenin işaretini keyfî seçer; PC1'in hangi ucunun asistan
    ucu olduğu bu yüzden işaretten değil, Assistant ekseniyle
    (`assistant_axis.npy` = normalize(mean(default) - mean(fully rol
    vektörleri)), yani rollerden varsayılana bakan yön) kosinüsünden okunur:
        cos(PC1, eksen) > 0   asistan ucu = max ucu (goreli = 1)
        cos(PC1, eksen) < 0   asistan ucu = min ucu (goreli = 0)
    0,6B'de kosinüs bütün katmanlarda negatiftir; asistan ucu min ucudur.
    `asistan_yonlu_konum` bu işareti giderir: 1 her zaman asistan ucudur.

TUTARLILIK KONTROLÜ
    Aynı hesap, sıra persentilini ve kosinüsü de üretir. Bunlar commit'li
    `layer_sweep.json` (4 ondalık) ve `criterion_a.json` (orta katman) ile
    karşılaştırılır; uyuşmazlık varsa betik dosya YAZMADAN çıkış 2 ile
    durur. Bu, aralık değerlerinin A kriterini üreten verinin aynısından
    geldiğini gösterir.

GİRDİLER
    commit'li : results/models/<slug>/axis/{role_vectors.npy, role_names.json,
                assistant_axis.npy, layer_sweep.json, criterion_a.json}
    yerel     : data/models/<slug>/{activations.npy, activations_index.json}
                (gitignore'da; varsayılan satırların ortalaması yalnız
                buradan hesaplanabilir, commit'li eksen normalize edildiği
                için varsayılan ortalamasını geri vermez)

KULLANIM
    uv run --extra ml python scripts/12_aralik_konumu.py
    uv run --extra ml python scripts/12_aralik_konumu.py --model Qwen/Qwen3-0.6B

ÇIKIŞ KODLARI
    0  bütün istenen modeller için dosya yazıldı
    2  girdi eksik/bozuk ya da tutarlılık kontrolü tutmadı; o model için
       dosya yazılmadı
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from aax import config
from aax.axis import cosine, pca_components, projection_percentile

DEFAULT_MODELS = ("Qwen/Qwen3-1.7B", "Qwen/Qwen3-0.6B")

# Makalenin raporladığı değer (arXiv:2601.10387v1, §2.3.1). Metin bu değeri
# model ayrımı vermeden, tek sayı olarak verir; hangi modele (ya da üç
# modelin en küçüğüne) ait olduğu belirtilmez.
PAPER_MIN_DISTANCE_PC1 = 0.03

# `layer_sweep.json` cos/pct değerlerini 4 ondalığa yuvarlanmış yazar.
SWEEP_TOLERANCE = 5e-5 + 1e-12
# `criterion_a.json` tam hassasiyetli yazar; float toplama sırası farkı payı.
CRITERION_TOLERANCE = 1e-9

METHOD_TEXT = (
    "Makale ölçüsü (arXiv:2601.10387v1, §2.3.1): varsayılan asistan "
    "aktivasyonunun PC1 izdüşümünün, rol vektörlerinin PC1 izdüşümlerinin en "
    "küçüğü (0) ile en büyüğü (1) arasındaki göreli konumu; "
    "en_yakin_uca_uzaklik = min(goreli_konum, 1 - goreli_konum). PCA, "
    "07_extract_axis.py ile aynıdır (rol vektörlerinin ortalaması çıkarılır, "
    "SVD). Rol vektörleri fully ve somewhat vektörlerinin tamamıdır. Varsayılan "
    "vektörü, varsayılan satırlarının (üç nötr sistem promptu ve promptsuz "
    "koşul, toplam 1.600 yanıt) response token ortalamalarının ortalamasıdır; "
    "07_extract_axis.py'deki default_mean ile aynıdır. Asistan ucu cos(PC1, Assistant ekseni) "
    "işaretinden okunur: pozitifse max ucu, negatifse min ucu. "
    "asistan_yonlu_konum işaretten bağımsızdır, 1 asistan ucudur. "
    "sira_persentili ise ön kayıtlı A kriterinin sıra temelli ölçüsüdür "
    "(rol izdüşümlerinin varsayılana eşit ya da altındaki payı) ve "
    "layer_sweep.json ile aynıdır."
)

ASSISTANT_END_TEXT = (
    "Assistant ekseni normalize(mean(default) - mean(fully rol vektörleri)) "
    "olarak tanımlıdır, yani rollerden varsayılana bakar. SVD bir bileşenin "
    "işaretini keyfî seçtiği için PC1'in hangi ucunun asistan ucu olduğu "
    "cos(PC1, eksen) işaretinden belirlenir: cos > 0 ise PC1 eksenle aynı yöne "
    "bakar ve asistan ucu en büyük izdüşümdür (goreli_konum = 1); cos < 0 ise "
    "PC1 eksenin tersine bakar ve asistan ucu en küçük izdüşümdür "
    "(goreli_konum = 0). Uçtaki rol adları (ör. L14'te 1,7B'nin asistan ucunda "
    "researcher, 0,6B'nin asistan ucunda specialist) bu belirlemeyi ayrıca "
    "doğrulamak için yazılır."
)


class InputError(RuntimeError):
    """Girdi eksik, bozuk ya da commit'li sonuçlarla tutarsız."""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(config.PROJECT_ROOT))


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise InputError(f"{rel(path)} yok") from exc
    except json.JSONDecodeError as exc:
        raise InputError(f"{rel(path)} ayrıştırılamadı: {exc}") from exc


def load_npy(path: Path, **kwargs) -> np.ndarray:
    try:
        return np.load(path, **kwargs)
    except FileNotFoundError as exc:
        raise InputError(f"{rel(path)} yok") from exc


def range_position(default_projection: float, role_projections: np.ndarray) -> float:
    """Varsayılan izdüşümünün rol izdüşümleri aralığındaki göreli konumu.

    0 en küçük rol izdüşümü, 1 en büyüğüdür. Aralığın dışındaki bir
    varsayılan için değer [0, 1] dışına çıkar ve kırpılmaz: "uca 0 uzaklık"
    ile "ucu aşmış" ayrı bilgilerdir.
    """
    lo = float(np.min(role_projections))
    hi = float(np.max(role_projections))
    if not hi > lo:
        raise ValueError("rol izdüşümlerinin aralığı sıfır; göreli konum tanımsız")
    return (float(default_projection) - lo) / (hi - lo)


def layer_record(
    layer: int,
    vectors_l: np.ndarray,
    axis_l: np.ndarray,
    default_mean_l: np.ndarray,
    names: list[str],
) -> dict:
    components, ratios = pca_components(vectors_l, n_components=1)
    pc1 = components[0]
    cos_value = cosine(pc1, axis_l)
    role_proj = vectors_l @ pc1
    default_proj = float(default_mean_l @ pc1)

    position = range_position(default_proj, role_proj)
    assistant_end = "max" if cos_value > 0 else "min"
    oriented = position if assistant_end == "max" else 1.0 - position
    nearest = min(position, 1.0 - position)
    to_assistant = 1.0 - oriented
    to_other = oriented
    if assistant_end == "max":
        beyond = int((role_proj > default_proj).sum())
    else:
        beyond = int((role_proj < default_proj).sum())

    i_min = int(np.argmin(role_proj))
    i_max = int(np.argmax(role_proj))
    assistant_role = names[i_max] if assistant_end == "max" else names[i_min]
    other_role = names[i_min] if assistant_end == "max" else names[i_max]

    return {
        "layer": layer,
        "cos_pc1_axis": cos_value,
        "pc1_explained_variance_ratio": float(ratios[0]),
        "asistan_ucu": assistant_end,
        "varsayilan_izdusum": default_proj,
        "rol_izdusum_min": float(role_proj[i_min]),
        "rol_izdusum_max": float(role_proj[i_max]),
        "asistan_ucundaki_rol": assistant_role,
        "karsi_uctaki_rol": other_role,
        "goreli_konum": position,
        "asistan_yonlu_konum": oriented,
        "en_yakin_uca_uzaklik": nearest,
        "en_yakin_uc": "asistan" if to_assistant <= to_other else "karsi",
        "asistan_ucuna_uzaklik": to_assistant,
        "aralik_icinde": bool(0.0 <= position <= 1.0),
        "varsayilandan_daha_asistan_tarafindaki_rol_sayisi": beyond,
        "sira_persentili": projection_percentile(default_proj, role_proj),
    }


def check_against_committed(records: list[dict], sweep: dict, criterion: dict) -> dict:
    """Yeniden hesaplanan cos ve sıra persentilini commit'li sonuçlarla karşılaştır."""
    sweep_layers = {int(row["layer"]): row for row in sweep["layers"]}
    if sorted(sweep_layers) != [r["layer"] for r in records]:
        raise InputError("layer_sweep.json katman listesi yeniden hesapla uyuşmuyor")
    cos_diff = max(abs(r["cos_pc1_axis"] - sweep_layers[r["layer"]]["cos"]) for r in records)
    pct_diff = max(abs(r["sira_persentili"] - sweep_layers[r["layer"]]["pct"]) for r in records)
    if cos_diff > SWEEP_TOLERANCE or pct_diff > SWEEP_TOLERANCE:
        raise InputError(
            "yeniden hesaplanan değerler layer_sweep.json ile uyuşmuyor "
            f"(max |Δcos| = {cos_diff:.2e}, max |Δpct| = {pct_diff:.2e}); "
            "girdiler farklı koşulardan geliyor olabilir"
        )
    middle = int(criterion["middle_layer"])
    mid = records[middle]
    crit_cos_diff = abs(mid["cos_pc1_axis"] - float(criterion["cos_pc1_axis"]))
    crit_pct_diff = abs(mid["sira_persentili"] - float(criterion["default_percentile"]))
    if crit_cos_diff > CRITERION_TOLERANCE or crit_pct_diff > CRITERION_TOLERANCE:
        raise InputError(
            "orta katman criterion_a.json ile uyuşmuyor "
            f"(|Δcos| = {crit_cos_diff:.2e}, |Δpct| = {crit_pct_diff:.2e})"
        )
    by_layer_cos = criterion.get("cos_by_layer") or []
    cos_by_layer_diff = (
        max(abs(r["cos_pc1_axis"] - float(c)) for r, c in zip(records, by_layer_cos))
        if len(by_layer_cos) == len(records)
        else None
    )
    if cos_by_layer_diff is not None and cos_by_layer_diff > CRITERION_TOLERANCE:
        raise InputError(
            f"cos_by_layer criterion_a.json ile uyuşmuyor (max |Δ| = {cos_by_layer_diff:.2e})"
        )
    return {
        "layer_sweep_max_abs_fark_cos": cos_diff,
        "layer_sweep_max_abs_fark_sira_persentili": pct_diff,
        "layer_sweep_tolerans": "4 ondalık yuvarlama (5e-5)",
        "criterion_a_orta_katman_abs_fark_cos": crit_cos_diff,
        "criterion_a_orta_katman_abs_fark_persentil": crit_pct_diff,
        "criterion_a_cos_by_layer_max_abs_fark": cos_by_layer_diff,
        "sonuc": "tutarlı",
    }


def run_model(model_id: str) -> Path:
    results_dir = config.model_results_dir(model_id) / "axis"
    data_dir = config.model_data_dir(model_id)

    vectors_path = results_dir / "role_vectors.npy"
    names_path = results_dir / "role_names.json"
    axis_path = results_dir / "assistant_axis.npy"
    sweep_path = results_dir / "layer_sweep.json"
    criterion_path = results_dir / "criterion_a.json"
    acts_path = data_dir / "activations.npy"
    index_path = data_dir / "activations_index.json"

    vectors = load_npy(vectors_path)
    names = load_json(names_path)
    axis = load_npy(axis_path)
    sweep = load_json(sweep_path)
    criterion = load_json(criterion_path)
    index = load_json(index_path)
    acts = load_npy(acts_path, mmap_mode="r")

    if vectors.ndim != 3 or len(names) != vectors.shape[0]:
        raise InputError("role_vectors.npy ile role_names.json uyuşmuyor")
    if axis.shape != vectors.shape[1:]:
        raise InputError("assistant_axis.npy şekli rol vektörleriyle uyuşmuyor")
    if acts.ndim != 3 or acts.shape[1:] != vectors.shape[1:]:
        raise InputError("activations.npy şekli rol vektörleriyle uyuşmuyor")
    if index.get("n_rows") != int(acts.shape[0]) or len(index.get("rows", [])) != int(acts.shape[0]):
        raise InputError("activations_index.json ile activations.npy uyuşmuyor")
    if index.get("run_id") != criterion.get("run_id"):
        raise InputError(
            f"run_id uyuşmuyor: indeks {index.get('run_id')!r}, "
            f"criterion_a.json {criterion.get('run_id')!r}"
        )
    if index.get("model") != model_id:
        raise InputError(f"indeks modeli {index.get('model')!r}, istenen {model_id!r}")

    default_idx = [i for i, row in enumerate(index["rows"]) if row["kind"] == "default"]
    if not default_idx:
        raise InputError("activations_index.json içinde 'default' satırı yok")
    # 07_extract_axis.py ile aynı ifade: float64'e çevirip ortalama.
    default_mean = acts[default_idx].astype(np.float64).mean(axis=0)
    if not np.isfinite(default_mean).all():
        raise InputError("varsayılan ortalaması sonlu değil")

    records = [
        layer_record(layer, vectors[:, layer, :], axis[layer], default_mean[layer], names)
        for layer in range(vectors.shape[1])
    ]
    consistency = check_against_committed(records, sweep, criterion)

    middle = int(index["middle_layer"])
    mid = records[middle]
    min_layer = min(records, key=lambda r: r["asistan_ucuna_uzaklik"])
    within_paper = [r["layer"] for r in records if r["asistan_ucuna_uzaklik"] <= PAPER_MIN_DISTANCE_PC1]
    nearest_not_assistant = [r["layer"] for r in records if r["en_yakin_uc"] != "asistan"]

    payload = {
        "model": model_id,
        "run_id": index.get("run_id"),
        "n_layers": int(vectors.shape[1]),
        "d_model": int(vectors.shape[2]),
        "middle_layer": middle,
        "n_role_vectors": int(vectors.shape[0]),
        "n_default_rows": len(default_idx),
        "yontem": METHOD_TEXT,
        "asistan_ucu_aciklamasi": ASSISTANT_END_TEXT,
        "makale_referansi": {
            "kaynak": "arXiv:2601.10387v1, §2.3.1 Projecting default activations into persona space",
            "olcu": (
                "Varsayılan aktivasyonun ilk on PC'nin her birindeki izdüşümünün, bütün rol "
                "izdüşümlerinin aralığındaki göreli konumu (0 ve 1: en küçük ve en büyük "
                "izdüşümlü iki rol vektörü)."
            ),
            "pc1_herhangi_bir_uca_en_kucuk_uzaklik": PAPER_MIN_DISTANCE_PC1,
            "diger_pcler_araligi": [0.27, 0.50],
            "katman": "orta residual stream katmanı",
            "modeller": ["Gemma 2 27B", "Qwen 3 32B", "Llama 3.3 70B"],
            "not": (
                "Makale 0,03 değerini model ayrımı vermeden tek sayı olarak verir; model başına "
                "değer raporlanmaz. Makalede varsayılanın konumu için sıra temelli (desil) bir "
                "ölçüt yoktur; uç desil ölçütü bu replikasyonun ön kayıtlı A kriteridir."
            ),
        },
        "orta_katman_ozeti": {
            "layer": middle,
            "cos_pc1_axis": mid["cos_pc1_axis"],
            "asistan_ucu": mid["asistan_ucu"],
            "goreli_konum": mid["goreli_konum"],
            "asistan_yonlu_konum": mid["asistan_yonlu_konum"],
            "en_yakin_uca_uzaklik": mid["en_yakin_uca_uzaklik"],
            "asistan_ucuna_uzaklik": mid["asistan_ucuna_uzaklik"],
            "sira_persentili": mid["sira_persentili"],
            "varsayilandan_daha_asistan_tarafindaki_rol_sayisi": mid[
                "varsayilandan_daha_asistan_tarafindaki_rol_sayisi"
            ],
        },
        "asistan_ucuna_en_yakin_katman": {
            "layer": min_layer["layer"],
            "asistan_ucuna_uzaklik": min_layer["asistan_ucuna_uzaklik"],
        },
        "asistan_ucuna_uzakligi_makale_degeri_veya_altinda_olan_katmanlar": within_paper,
        "en_yakin_ucu_asistan_ucu_olmayan_katmanlar": nearest_not_assistant,
        "tutarlilik_kontrolu": consistency,
        "kaynak_dosyalar": {
            rel(vectors_path): {"sha256": sha256_of(vectors_path)},
            rel(names_path): {"sha256": sha256_of(names_path)},
            rel(axis_path): {"sha256": sha256_of(axis_path)},
            rel(sweep_path): {"sha256": sha256_of(sweep_path), "kullanim": "yalnız tutarlılık kontrolü"},
            rel(criterion_path): {"sha256": sha256_of(criterion_path), "kullanim": "yalnız tutarlılık kontrolü"},
            rel(acts_path): {
                "yerel": True,
                "run_id": index.get("run_id"),
                "shape": [int(s) for s in acts.shape],
                "kullanim": "yalnız default satırları (varsayılan ortalaması)",
            },
            rel(index_path): {"yerel": True, "sha256": sha256_of(index_path)},
        },
        "uretici": "scripts/12_aralik_konumu.py",
        "katmanlar": records,
    }

    out_path = results_dir / "range_position.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out_path


def print_summary(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(f"\n{payload['model']}  ({rel(path)})")
    print("  L   cos      uç   göreli  asistan-yönlü  uca(en yakın)  asistan ucuna  sıra-pct")
    for r in payload["katmanlar"]:
        mark = " *" if r["layer"] == payload["middle_layer"] else ""
        print(
            f"  {r['layer']:>2}  {r['cos_pc1_axis']:+.4f}  {r['asistan_ucu']:>3}  "
            f"{r['goreli_konum']:.4f}  {r['asistan_yonlu_konum']:.4f}         "
            f"{r['en_yakin_uca_uzaklik']:.4f} ({r['en_yakin_uc']:<7})  "
            f"{r['asistan_ucuna_uzaklik']:.4f}         {r['sira_persentili']:.4f}{mark}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model",
        action="append",
        help="hedef model kimliği (tekrarlanabilir); verilmezse iki model de hesaplanır",
    )
    args = parser.parse_args(argv)
    models = args.model or list(DEFAULT_MODELS)

    status = 0
    for model_id in models:
        try:
            out = run_model(model_id)
        except (InputError, ValueError) as exc:
            print(f"BAŞARISIZ ({model_id}): {exc}. Dosya yazılmadı.", file=sys.stderr)
            status = 2
            continue
        print_summary(out)
    return status


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
