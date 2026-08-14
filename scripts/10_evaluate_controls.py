#!/usr/bin/env python3
"""C kriteri değerlendirmesi — kontrol yönleri Aşama 4'ün nedensel iddiasını
İZOLE EDİYOR MU?

Bu script AĞA ÇIKMAZ: yalnızca `scripts/09_evaluate_steering.py`'nin diske
yazdığı `rate_by_strength(_<AD>).json` dosyalarını okur, hakemi TEKRAR
ÇAĞIRMAZ. Girdiler:

  - `results/models/<slug>/steering/rate_by_strength.json` — Aşama 4'ün EKSEN
    koşusu (`08_steering_sweep.py --direction axis` + `09` --variant OLMADAN).
    PAYLAŞILAN taban (0.0 gücü) BURADAN okunur.
  - `results/models/<slug>/steering/rate_by_strength_<AD>.json` — her kontrol
    yönü (`gaussian`, `shuffled`, `rolespan`) için `09 --variant <AD>`'nin
    yazdığı oranlar. Bu dosyalarda 0.0 gücü YOKTUR (ön-tescil gereği).
  - `data/models/<slug>/steering_sweep_<AD>_meta.json` — her kontrolün
    `direction_kind`/`direction_seed`/`direction_sha256`'sı (bkz.
    `08_steering_sweep.py::_meta_payload`), yeniden üretilebilirlik parmak
    izi olarak artefakta gömülür.

Paylaşılan taban KASITLI: `results/control_preregistration.json`, `steering_
delta(d, 0.0, norm)`'un HER yön için TAM OLARAK sıfır vektör döndürdüğünü,
yani 0.0 gücünde üretimin yönden BAĞIMSIZ olduğunu gerekçe gösterir — taban
Aşama 4'te zaten YAPILMIŞ 250 örneklik ölçümdür; onu kontrol başına tekrar
üretmek 25 hakem çağrısını (yön başına) boşa harcardı. Bu script tabanı
Aşama 4'ün eksen koşusundan okur ve NEREDEN geldiğini artefakta yazar — asla
bir kontrolün kendi dosyasından okumaz.

C KRİTERİ (ön-tescil): hiçbir kontrol yönünün artışı (en negatif güçteki
Assistant-dışı oran eksi PAYLAŞILAN 0.0 tabanı) B kriterinin 25 puanlık
eşiğine ULAŞMAMALI. Mutlak bir bardır — oran DEĞİL: ön-tescil "sonuç ne
olursa olsun eşik/taban/üç yön sabit kalır" der (`results/control_
preregistration.json`, `kriter_degistirilmez`). Eksenin artışının en büyük
kontrol artışına oranı yalnızca İKİNCİL bilgi olarak `ratio_axis_to_control`
alanında raporlanır, kriteri BELİRLEMEZ.

Eşik toleransı B kriterindekiyle AYNI (`aax.susceptibility.B_THRESHOLD`,
`B_THRESHOLD_EPS`) — yeni bir tolerans İCAT EDİLMEZ.

ÇIKIŞ KODLARI (proje geneli, yük taşıyan bir sözleşme):

    0  C KRİTERİ GEÇTİ — hiçbir kontrol eşiğe ULAŞMADI
    1  C KRİTERİ DÜŞTÜ — en az bir kontrol eşiğe ULAŞTI (gerçek, değerlendirilmiş
       bir bilimsel sonuç: Aşama 4'ün nedensel iddiası İZOLE EDİLEMEDİ)
    2  KARAR ÜRETİLEMEDİ — eksik bir kontrol dosyası, eksik -0.6 gücü, ya da
       eksik/okunamayan bir taban. Bu ASLA "GEÇTİ" (0) ya da "DÜŞTÜ" (1)
       DEĞİLDİR: karar üretilemeyen bir koşu, C kriterinin geçtiği anlamına
       GELMEZ — `07_extract_axis.py`/`09_evaluate_steering.py`'nin AYNI
       ayrımı (bkz. o dosyaların modül docstring'leri).

Kullanım:
    uv run python scripts/10_evaluate_controls.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from aax import config
from aax.controls import CONTROL_KINDS
from aax.susceptibility import B_THRESHOLD, B_THRESHOLD_EPS

# Ön-tescildeki (results/control_preregistration.json) TEK karşılaştırma
# noktası — B kriterindeki "en negatif güç" ile AYNI değer, ama burada
# "sweep'te bulunan en küçük güç" olarak DEĞİL, açıkça -0.6 olarak aranır:
# kontrol sweep'i yalnızca [-0.6, -0.4, -0.2] gücünü tarıyor (7'sini değil),
# ama kriterin KENDİSİ ön-tescilde -0.6'ya sabitlenmiş — sweep'in hangi
# güçleri TARADIĞINDAN bağımsız bir sabit olmalı.
FAR_STRENGTH = -0.6

# Ön-tescildeki gerekçe metni — artefakta AYNEN gömülür (bkz. modül
# docstring'i, "Paylaşılan taban KASITLI" paragrafı).
BASELINE_SOURCE_NOTE = (
    "results/models/<slug>/steering/rate_by_strength.json — Aşama 4 EKSEN "
    "koşusunun 0.0 gücündeki hücresi. steering_delta(d, 0.0, norm) HER yön "
    "için TAM OLARAK sıfır vektör döndürür (0.0 gücünde üretim yönden "
    "BAĞIMSIZDIR); bu yüzden taban kontrol başına YENİDEN ÖLÇÜLMEDİ, Aşama "
    "4'ten PAYLAŞILDI (bkz. results/control_preregistration.json, 'taban')."
)


# --- saf yardımcılar: dosya biçimi, delta, karar ----------------------------


def load_rate_by_strength(path: str | Path) -> dict[int, dict[float, float]]:
    """`rate_by_strength(_<AD>).json`'ı `{katman: {güç: oran}}`'a çevirir.

    JSON'da anahtarlar (katman/güç) string'dir (`09_evaluate_steering.py`
    `str(l)`/`str(s)` ile yazıyor) — burası bunları `int`/`float`'a geri
    çözer. Şema bozuksa (üst seviye dict değil, bir katmanın değeri dict
    değil) okunabilir bir `ValueError` fırlatır."""
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: üst seviye bir nesne (JSON object) bekleniyordu")
    out: dict[int, dict[float, float]] = {}
    for layer_key, by_strength in raw.items():
        if not isinstance(by_strength, dict):
            raise ValueError(f"{path}: '{layer_key}' katmanının değeri bir nesne değil")
        out[int(layer_key)] = {float(s): float(r) for s, r in by_strength.items()}
    return out


def direction_delta(
    rate_by_strength: dict[float, float],
    baseline_rate: float,
    *,
    far_strength: float = FAR_STRENGTH,
) -> dict:
    """Tek bir (yön, katman) hücresi için C kriterinin deltasını hesapla.

    B kriterindeki (`aax.susceptibility.evaluate_criterion_b`) AYNI eşik
    disiplinini kullanır: `delta >= B_THRESHOLD - B_THRESHOLD_EPS` — yeni
    bir tolerans İCAT EDİLMEZ (bkz. spec, o modülün `B_THRESHOLD_EPS`
    yorumu: float64 çıkarma gürültüsünü emer, gerçek eşik-altı bir deltayı
    YUTMAZ).

    `far_strength` sweep'in KENDİ ölçtüğü güçler içinde yoksa (kontrol
    sweep'i yalnızca [-0.6, -0.4, -0.2] tarıyor — -0.6 HER ZAMAN olmalı, ama
    savunma yine de açık) okunabilir bir `ValueError` fırlatır."""
    if far_strength not in rate_by_strength:
        raise ValueError(
            f"{far_strength} gücünde ölçüm yok — C kriteri değerlendirilemez "
            f"(mevcut güçler: {sorted(rate_by_strength)})"
        )
    far_rate = rate_by_strength[far_strength]
    if not np.isfinite(baseline_rate) or not np.isfinite(far_rate):
        raise ValueError("oran değerleri sonlu olmalı (NaN/inf tespit edildi)")
    delta = far_rate - baseline_rate
    reached = bool(delta >= B_THRESHOLD - B_THRESHOLD_EPS)
    reason = (
        "eşiğe ULAŞTI — B kriterinin eşiği bu kontrol yönüyle de aşılabiliyor, "
        "Aşama 4'ün nedensel iddiası bu katmanda İZOLE EDİLEMEDİ"
        if reached
        else (
            f"eşiğin altında kaldı: artış {100 * delta:+.1f} puan, eşik "
            f"{100 * B_THRESHOLD:.0f} puan"
        )
    )
    return {
        "baseline_strength": 0.0,
        "baseline_rate": baseline_rate,
        "far_strength": far_strength,
        "far_rate": far_rate,
        "delta": delta,
        "threshold": B_THRESHOLD,
        "threshold_eps": B_THRESHOLD_EPS,
        "reached_threshold": reached,
        "passed": not reached,
        "reason": reason,
    }


def evaluate_direction(
    control_rates: dict[int, dict[float, float]],
    axis_rates: dict[int, dict[float, float]],
) -> dict[int, dict]:
    """Bir kontrol yönünün TÜM katmanları için C kriterini değerlendir.

    Her katman için PAYLAŞILAN taban (0.0 gücü), o katmanda EKSENİN kendi
    ölçümünden (`axis_rates`) okunur — kontrolün KENDİ dosyasından DEĞİL
    (kontrol dosyalarında 0.0 hiç yok). Eksende o katman ya da 0.0 tabanı
    eksikse okunabilir bir `ValueError` fırlatır — bu "eksen bu katmanı hiç
    ölçmedi" ya da "eksenin taban ölçümü eksik" demektir, "kontrol geçti"
    DEĞİL."""
    if not control_rates:
        raise ValueError("kontrol için hiç katman/güç verisi yok")
    out: dict[int, dict] = {}
    for layer, by_strength in sorted(control_rates.items()):
        if layer not in axis_rates or 0.0 not in axis_rates[layer]:
            raise ValueError(
                f"L{layer} için eksende 0.0 gücünde PAYLAŞILAN taban yok — "
                "bkz. results/models/<slug>/steering/rate_by_strength.json "
                "(önce scripts/09_evaluate_steering.py --variant OLMADAN "
                "çalıştırılmalı)"
            )
        out[layer] = direction_delta(by_strength, axis_rates[layer][0.0])
    return out


def overall_exit_code(direction_verdicts: dict[str, dict[int, dict]]) -> int:
    """0 = hiçbir kontrol eşiğe ulaşmadı · 1 = en az biri ulaştı · 2 = hiç
    yön değerlendirilmedi.

    `09_evaluate_steering.py::overall_exit_code`'daki E3 ile AYNI ilke:
    `all([])`/`any([])` boş girdide "GEÇTİ"yi TANIMSIZ veriden üretmesin diye
    boş girdi AYRI bir kod (2) alır."""
    all_cells = [v for layers in direction_verdicts.values() for v in layers.values()]
    if not all_cells:
        return 2
    return 1 if any(v["reached_threshold"] for v in all_cells) else 0


def _axis_delta_for_layer(axis_rates: dict[int, dict[float, float]], layer: int) -> float | None:
    """Eksenin KENDİ artışı (rapor için, kriteri BELİRLEMEZ) — eksenin -0.6
    ölçümü yoksa (beklenmez, ama Aşama 4'ün B kriteri zaten bunu gerektirir)
    sessizce `None` döner; bu yalnızca `ratio_axis_to_control` alanını
    etkiler, C kararını DEĞİL."""
    by_strength = axis_rates.get(layer)
    if not by_strength or 0.0 not in by_strength or FAR_STRENGTH not in by_strength:
        return None
    try:
        return direction_delta(by_strength, by_strength[0.0])["delta"]
    except ValueError:
        return None


def build_payload(
    axis_rates: dict[int, dict[float, float]],
    direction_verdicts: dict[str, dict[int, dict]],
    direction_meta: dict[str, dict],
) -> dict:
    """`criterion_c.json`'ın tam içeriğini kur — saf, I/O yapmaz (test
    edilebilirlik için `_run`'dan ayrıştırıldı, `09`'daki `_meta_payload`
    deseninin aynısı)."""
    directions_out: dict[str, dict] = {}
    for kind, layers in direction_verdicts.items():
        layers_out: dict[str, dict] = {}
        for layer, verdict in layers.items():
            axis_delta = _axis_delta_for_layer(axis_rates, layer)
            control_delta = verdict["delta"]
            ratio = (
                axis_delta / control_delta
                if axis_delta is not None and control_delta != 0
                else None
            )
            layers_out[str(layer)] = {**verdict, "axis_delta": axis_delta,
                                       "ratio_axis_to_control": ratio}
        meta = direction_meta.get(kind, {})
        directions_out[kind] = {
            "layers": layers_out,
            "passed": all(v["passed"] for v in layers_out.values()),
            "direction_kind": meta.get("direction_kind", kind),
            "direction_seed": meta.get("direction_seed"),
            "direction_sha256": meta.get("direction_sha256"),
        }
    return {
        "model": config.TARGET_MODEL,
        "far_strength": FAR_STRENGTH,
        "threshold": B_THRESHOLD,
        "threshold_eps": B_THRESHOLD_EPS,
        "baseline_source": BASELINE_SOURCE_NOTE,
        "directions": directions_out,
        "passed": all(d["passed"] for d in directions_out.values()) if directions_out else None,
        "note": (
            "Kriter C (ön-tescil): hiçbir kontrol yönünün artışı B kriterinin "
            f"{100 * B_THRESHOLD:.0f} puanlık eşiğine ULAŞMAMALI — mutlak bir "
            "bardır, oran DEĞİL. 'ratio_axis_to_control' yalnızca İKİNCİL "
            "rapordur, kriteri belirlemez. Sonuç ne olursa olsun eşik/taban/"
            "üç kontrol yönü DEĞİŞTİRİLMEZ (bkz. results/"
            "control_preregistration.json, 'kriter_degistirilmez')."
        ),
    }


# --- I/O: dosyaları oku, karar üret, artefaktı yaz --------------------------


def _run(argv: list[str] | None) -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args(argv)  # şu an bir seçenek yok — yalnızca -h/--help için

    out_dir = config.model_results_dir() / "steering"
    axis_path = out_dir / "rate_by_strength.json"
    if not axis_path.exists():
        print(
            f"BAŞARISIZ: {axis_path} yok — eksenin PAYLAŞILAN taban ölçümü "
            "(0.0 gücü) okunamıyor.\n"
            "  Önce scripts/09_evaluate_steering.py (--variant OLMADAN, Aşama "
            "4'ün eksen koşusu) çalıştırılmalı.",
            file=sys.stderr,
        )
        return 2
    try:
        axis_rates = load_rate_by_strength(axis_path)
    except (ValueError, OSError) as exc:
        print(f"BAŞARISIZ: {axis_path} okunamadı/ayrıştırılamadı.\n  {exc}", file=sys.stderr)
        return 2

    D = config.model_data_dir()
    direction_verdicts: dict[str, dict[int, dict]] = {}
    direction_meta: dict[str, dict] = {}
    for kind in CONTROL_KINDS:
        rate_path = out_dir / f"rate_by_strength_{kind}.json"
        meta_path = D / f"steering_sweep_{kind}_meta.json"
        if not rate_path.exists():
            print(
                f"BAŞARISIZ: {rate_path} yok — '{kind}' kontrolü değerlendirilemedi.\n"
                f"  Önce scripts/08_steering_sweep.py --direction {kind} "
                f"--variant {kind} ... ve scripts/09_evaluate_steering.py "
                f"--variant {kind} çalıştırılmalı.",
                file=sys.stderr,
            )
            return 2
        if not meta_path.exists():
            print(
                f"BAŞARISIZ: {meta_path} yok — '{kind}' kontrolünün "
                "direction_sha256'sı (yeniden üretilebilirlik parmak izi) "
                "okunamıyor.",
                file=sys.stderr,
            )
            return 2
        try:
            control_rates = load_rate_by_strength(rate_path)
        except (ValueError, OSError) as exc:
            print(f"BAŞARISIZ: {rate_path} okunamadı/ayrıştırılamadı.\n  {exc}", file=sys.stderr)
            return 2
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except ValueError as exc:
            print(f"BAŞARISIZ: {meta_path} ayrıştırılamadı.\n  {exc}", file=sys.stderr)
            return 2
        if not isinstance(meta, dict):
            print(f"BAŞARISIZ: {meta_path} bir nesne (JSON object) değil.", file=sys.stderr)
            return 2

        try:
            direction_verdicts[kind] = evaluate_direction(control_rates, axis_rates)
        except ValueError as exc:
            print(f"BAŞARISIZ: '{kind}' kontrolü değerlendirilemedi.\n  {exc}", file=sys.stderr)
            return 2
        direction_meta[kind] = {
            "direction_kind": meta.get("direction_kind", kind),
            "direction_seed": meta.get("direction_seed"),
            "direction_sha256": meta.get("direction_sha256"),
        }

    code = overall_exit_code(direction_verdicts)

    payload = build_payload(axis_rates, direction_verdicts, direction_meta)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "criterion_c.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print()
    for kind in CONTROL_KINDS:
        for layer in sorted(direction_verdicts.get(kind, {})):
            v = direction_verdicts[kind][layer]
            print(
                f"{kind} L{layer}: taban {v['baseline_rate']:.3f} → "
                f"{v['far_strength']} gücünde {v['far_rate']:.3f}   "
                f"artış {100 * v['delta']:+.1f} puan   "
                f"{'ESİĞE ULAŞTI' if v['reached_threshold'] else 'eşik altında'}"
            )
    print()
    if code == 0:
        print("C KRİTERİ: GEÇTİ — hiçbir kontrol yönü eşiğe ulaşmadı")
    elif code == 1:
        print("C KRİTERİ: DÜŞTÜ — en az bir kontrol yönü eşiğe ulaştı")
    else:
        print("C KRİTERİ: KARAR ÜRETİLEMEDİ (hiçbir kontrol yönü değerlendirilemedi)")
    print(f"Yazıldı: {out_dir / 'criterion_c.json'}")
    return code


def main(argv: list[str] | None = None) -> int:
    """Tanı sarmalayıcısı — `07_extract_axis.py:609-637` / `09_evaluate_
    steering.py::main` ile AYNI desen: gerçek gövde `_run()`'da, burada
    yalnızca ÖNGÖRÜLMEMİŞ bir istisnayı yakalayıp temiz bir Türkçe teşhisle
    çıkış kodu 2 döndürür — çıplak bir traceback ya da (daha kötüsü) çıkış
    kodu 1 ("C KRİTERİ DÜŞTÜ" anlamına gelir) DEĞİL. `KeyboardInterrupt`
    BİLEREK ayrı tutulur ve yeniden fırlatılır."""
    try:
        return _run(argv)
    except KeyboardInterrupt:
        raise
    except Exception as exc:  # noqa: BLE001 — kasıtlı geniş yakalama, gerekçe docstring'de
        print(
            f"BAŞARISIZ: beklenmeyen hata — bu bir C kriteri kararı DEĞİLDİR.\n"
            f"  {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
