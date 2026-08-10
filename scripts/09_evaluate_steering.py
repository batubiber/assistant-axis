#!/usr/bin/env python3
"""Aşama 4 değerlendirmesi — persona sınıflandırması ve B kriteri.

Sweep'in her (katman, güç) grubu hakemle sınıflandırılır, Assistant-dışı
oran hesaplanır, B kriteri KATMAN BAŞINA ayrı değerlendirilir.

Çıkış kodu: 0 = her katmanda geçti · 1 = en az bir katmanda düştü
(değerlendirilmiş bir sonuç) · 2 = koşu karar üretemedi (çökme, eksik/bozuk
girdi, bütçe yetersizliği, yarım kalmış bir sweep, ya da hiç grup
değerlendirilememesi — hiçbiri "B KRİTERİ DÜŞTÜ" DEĞİLDİR).

Kullanım:
    uv run python scripts/09_evaluate_steering.py --dry-run
    uv run --extra ml python scripts/09_evaluate_steering.py

Dayanıklılık (2026-08-10 düzeltmesi; bkz.
`.superpowers/sdd/p3-task-5-supplement.md`, madde E1): `classify_personas`
uzunluk uyuşmazlığında `JudgeParseError` fırlatır — bu KASITLI ve doğru,
modülün işi kurtarma değil. Kurtarma bu script'in işi. `06_label_and_train_
probe.py`'nin ("Etiketleme geçişi DAYANIKLIDIR" paragrafı, commit 44dd90e)
AYNI deseni burada da uygulanıyor:

  - `data/models/<slug>/steering_labels.json` her hakem BATCH'inden sonra
    ATOMİK yazılıyor (tempfile + `os.replace`) — bir kesinti en fazla BİR
    batch'lik (<=`--batch-size` öğe) işi kaybettirir, koşunun tamamını değil.
    Bir sonraki koşu bu dosyayı okuyup zaten tamamlanmış (katman, güç)
    hücrelerini/konumlarını TEKRAR sormaz.
  - Bir hakem batch'i `JudgeParseError` verirse `_classify_batch` onu
    yarılara, gerekirse tekil öğelere böler (`_label_batch`'in AYNI
    deseni — bkz. o fonksiyonun docstring'i: bölme neden basit bir
    retry'dan farklı ve neden işe yarar: `GatewayClient._cache_key` payload'ın
    sha256'sıdır, daha AZ öğe içeren bir yarı FARKLI bir payload'dur).
  - Tek başına bile ayrıştırılamayan bir öğe koşuyu ÖLDÜRMEZ: "etiketlenemedi"
    sayılır, o hücrenin etiket listesinden (ve dolayısıyla oranın PAYDASINDAN)
    düşer, koşu devam eder. Kaç öğenin hangi hücrede etiketlenemediği
    `criterion_b.json`'a `unlabelled_by_group` alanıyla yazılır ve toplamı
    stderr'e UYARI olarak basılır — oranın kaç öğeden hesaplandığı artefaktın
    kendisinden görülebilir olmalı.
  - Bir hücrenin TÜM öğeleri etiketlenemezse (`non_assistant_rate` boş
    listede zaten `ValueError` atar) bu durum `main()`'in dış sarmalayıcısı
    tarafından temiz bir Türkçe teşhis + çıkış kodu 2'ye çevrilir — çıplak bir
    traceback ya da (daha kötüsü) çıkış kodu 1 ("B KRİTERİ DÜŞTÜ" anlamına
    gelirdi) DEĞİL.

Yarım kalmış sweep koruması (madde E2): `08_steering_sweep.py`'nin yazdığı
`steering_sweep_meta.json` okunur; `complete: false` ise (sweep bir çökme/
Ctrl-C/OOM ile kesildiyse) B kriteri kararı VARSAYILAN olarak ÜRETİLMEZ — bu
eksik veriden yanlış bir bilimsel sonuç ("GEÇTİ"/"DÜŞTÜ") yayınlamak olurdu.
Operatör bunu bilerek `--allow-incomplete` ile geçebilir; o zaman
`criterion_b.json` kendi içinde `incomplete_sweep: true` ile işaretlenir.

Boş karar kümesi artık "GEÇTİ" sayılmaz (madde E3): `all([])` Python'da
`True`'dur — hiç katman değerlendirilmemişken bunu 0 (GEÇTİ) döndürmek,
tanımsız veriden "GEÇTİ" basan bir hatanın (bu projede daha önce bir kez
görülen sınıf) aynısı olurdu. `overall_exit_code({})` artık 2 döner.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from aax import config
from aax.gateway import (
    BudgetCorrupted,
    BudgetExceeded,
    CircuitOpen,
    GatewayError,
    build_default_client,
)
from aax.judge import JudgeParseError
from aax.persona_judge import classify_personas
from aax.susceptibility import evaluate_criterion_b, non_assistant_rate

STAGE = "stage4_steering"


# --- saf yardımcılar: gruplama, oran, karar --------------------------------


def group_by_layer_strength(records: list[dict]) -> dict[tuple[int, float], list[dict]]:
    groups: dict[tuple[int, float], list[dict]] = defaultdict(list)
    for r in records:
        groups[(int(r["layer"]), float(r["strength"]))].append(r)
    return dict(groups)


def rates_by_layer(
    labels_by_group: dict[tuple[int, float], list[str]]
) -> dict[int, dict[float, float]]:
    out: dict[int, dict[float, float]] = defaultdict(dict)
    for (layer, strength), labels in labels_by_group.items():
        out[layer][strength] = non_assistant_rate(labels)
    return dict(out)


def evaluate_all_layers(rates: dict[int, dict[float, float]]) -> dict[int, dict]:
    return {layer: evaluate_criterion_b(by_strength) for layer, by_strength in rates.items()}


def overall_exit_code(verdicts: dict[int, dict]) -> int:
    """0 = her katman geçti · 1 = en az biri düştü · 2 = hiç katman yok.

    E3: `all([])` Python'da `True`'dur — boş `verdicts` (hiçbir katman
    değerlendirilmedi) sessizce 0 ("HER KATMANDA GEÇTİ") dönerdi. Bu, bu
    projede daha önce görülen "tanımsız veriden GEÇTİ basmak" hatasının
    aynı sınıfı; boş girdi artık AYRI bir kod (2, "karar üretilemedi") alır.
    """
    if not verdicts:
        return 2
    return 0 if all(v["passed"] for v in verdicts.values()) else 1


# --- etiket dosyası: konum bazlı, artımlı, atomik kalıcılık -----------------
#
# `steering_labels.json` bir (katman, güç) hücresinin İÇİNDEKİ konuma (grup
# sırasındaki 0-tabanlı indeks — arayüz sözleşmesindeki "index") göre
# indekslenir, DÜZ bir liste DEĞİL: bir öğe etiketlenemediğinde konumu
# sözlükte hiç YER ALMAZ (06'nın satır-numarası deseninin aynısı). Düz bir
# liste kullanılsaydı, ortadan bir öğe düştüğünde SONRAKİ öğelerin konumu
# kayardı — hangi orijinal öğenin hangi etikete karşılık geldiği belirsizleşirdi.


def _group_key(key: tuple[int, float]) -> str:
    layer, strength = key
    return f"{layer}|{strength}"


def _parse_group_key(text: str) -> tuple[int, float]:
    layer_text, _, strength_text = text.partition("|")
    return int(layer_text), float(strength_text)


def save_group_labels(
    path: str | Path,
    group_state: dict[tuple[int, float], dict[int, str]],
    unlabelled_state: dict[tuple[int, float], set[int]],
) -> None:
    """`steering_labels.json`'ı ATOMİK yaz — `06_label_and_train_probe.py::
    save_labels`'ın AYNI deseni (tempfile + `os.replace`). Çağıran taraf bunu
    HER hakem batch'inden sonra çağırır: bir çökme/kesinti en fazla BİR
    batch'lik işi kaybettirir, o ana kadar toplanan etiketlerin TAMAMINI
    değil. Yazım ortasında kesilme dosyayı KIRPMAZ."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "labels": {
            _group_key(key): {str(pos): label for pos, label in sorted(positions.items())}
            for key, positions in group_state.items()
        },
        "unlabelled_positions": {
            _group_key(key): sorted(positions)
            for key, positions in unlabelled_state.items()
        },
    }
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def load_group_labels(
    path: str | Path,
) -> tuple[dict[tuple[int, float], dict[int, str]], dict[tuple[int, float], set[int]]]:
    """Var olan `steering_labels.json`'ı yükle — bu, artımlı kalıcılığın OKUMA
    ucudur (06'nın `load_existing_labels`'ıyla AYNI rol): önceki bir koşu
    kesintiye uğradıysa tamamlanmış hücreler/konumlar TEKRAR hakeme
    SORULMAZ. Dosya yoksa (ilk koşu) boş döner.

    Dosya VARSA ama ayrıştırılamıyorsa ya da beklenen `labels` anahtarını
    taşımıyorsa istisna SARMALANMADAN çağırana yükselir — sessizce sıfırdan
    başlamak, zaten ÖDENMİŞ etiketleri sessizce silip aynı öğeleri yeniden
    ücretlendirmek demektir (06 ile aynı gerekçe)."""
    path = Path(path)
    if not path.exists():
        return {}, {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    labels_raw = payload["labels"]
    unlabelled_raw = payload.get("unlabelled_positions", {})
    group_state = {
        _parse_group_key(key): {int(pos): label for pos, label in positions.items()}
        for key, positions in labels_raw.items()
    }
    unlabelled_state = {
        _parse_group_key(key): set(positions) for key, positions in unlabelled_raw.items()
    }
    return group_state, unlabelled_state


# --- böl-ve-kurtar: bozuk bir hakem batch'i koşuyu düşürmesin ---------------


def _classify_batch(
    client,
    *,
    positions: list[int],
    items: list[tuple[str, str]],
    stage: str,
) -> tuple[dict[int, str], list[int], int]:
    """Bir hakem batch'ini persona sınıflandır; `JudgeParseError` fırlarsa
    böl-ve-tekrar-dene ile KURTAR — `06_label_and_train_probe.py::
    _label_batch` ile AYNI desen (bkz. supplement madde E1: bu tasarımı
    KOPYALA, yeni bir tasarım uydurma).

    Bölme TAM İKİ seviyelidir: batch -> yarılar -> tekil öğeler (bir yarı da
    başarısız olursa TEKRAR yarılanmaz, doğrudan tekil öğelere iner). N
    öğelik bir batch'in tam kurtarması en kötü durumda 1 (tüm batch) + 2
    (yarılar) + N (tüm tekil öğeler) gönderime mal olur — N=10 için 13.

    ÖNEMLİ — bölme yalnızca bir RETRY DEĞİLDİR: `GatewayClient._cache_key`
    payload'ın (mesajlar + sıcaklık + max_tokens) sha256'sıdır; daha AZ öğe
    içeren bir yarı/tekil öğe `persona_judge._build_prompt`'un ürettiği
    FARKLI bir metin demektir, dolayısıyla FARKLI bir cache anahtarı.
    `temperature=0` olduğu için AYNI payload'ı tekrar göndermek (basit bir
    retry) zehirlenmiş cache'teki AYNI bozuk yanıtı geri getirirdi; payload'ı
    DEĞİŞTİRMEK hakemi GERÇEKTEN yeniden sorar.

    `BudgetExceeded`/`CircuitOpen`/`BudgetCorrupted`/`GatewayError` burada
    YAKALANMAZ — bilerek: bunlar koşuyu durdurması GEREKEN fatal
    durumlardır ve çağırana (`_run`) olduğu gibi yükselirler. Yalnızca
    `JudgeParseError` (hakemin şekli bozuk bir yanıt vermesi) burada
    KAPSANIR.

    Dönüş: `(konum -> etiket sözlüğü, etiketlenemeyen konumlar, bölündü mü)`.
    `bölündü mü` 1 ise bu batch en az bir kez bölünmek ZORUNDA kaldı
    (yalnızca raporlama içindir).
    """
    try:
        labels = classify_personas(client, items, stage=stage, batch_size=len(items))
        return dict(zip(positions, labels)), [], 0
    except JudgeParseError:
        pass

    if len(positions) <= 1:
        # Zaten tekil öğe — daha fazla bölünemez, doğrudan etiketlenemeyen.
        return {}, list(positions), 0

    mid = len(positions) // 2
    halves = [(positions[:mid], items[:mid]), (positions[mid:], items[mid:])]
    labels_out: dict[int, str] = {}
    unlabelled: list[int] = []
    for half_positions, half_items in halves:
        try:
            half_labels = classify_personas(
                client, half_items, stage=stage, batch_size=len(half_items)
            )
            labels_out.update(zip(half_positions, half_labels))
            continue
        except JudgeParseError:
            pass
        if len(half_positions) == 1:
            # Yarı zaten tek öğeydi — tekrar tek öğe olarak denemek AYNI
            # payload'ı (dolayısıyla AYNI cache anahtarını) üretirdi, bu
            # yüzden gereksiz ikinci gönderim ATLANIR.
            unlabelled.append(half_positions[0])
            continue
        for pos, item in zip(half_positions, half_items):
            try:
                single = classify_personas(client, [item], stage=stage, batch_size=1)
                labels_out[pos] = single[0]
            except JudgeParseError:
                unlabelled.append(pos)
    return labels_out, unlabelled, 1


# --- sweep meta doğrulaması: yarım kalmış bir koşudan karar üretilmesin ------


def read_sweep_meta(meta_path: str | Path) -> dict:
    """`steering_sweep_meta.json`'ı oku ve gerekli alanların varlığını
    doğrula. Sorun varsa okunabilir bir Türkçe mesajla `ValueError` fırlatır
    (çağıran bunu `main()`'in genel sarmalayıcısına DEĞİL, kendi özel
    BAŞARISIZ mesajına çevirir — bkz. `_run`)."""
    meta_path = Path(meta_path)
    if not meta_path.exists():
        raise ValueError(f"{meta_path} yok.")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError(f"{meta_path} ayrıştırılamadı: {exc}") from exc
    required = {"complete", "attempted", "planned"}
    if not isinstance(meta, dict) or not required.issubset(meta):
        raise ValueError(
            f"{meta_path} beklenen alanları taşımıyor ({sorted(required)})."
        )
    return meta


def _run(argv: list[str] | None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="istek atmadan planlanan çağrı sayısını göster")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="hakem çağrısı başına öğe sayısı")
    parser.add_argument(
        "--allow-incomplete", action="store_true",
        help=(
            "steering_sweep_meta.json 'complete: false' derse bile bilerek "
            "kısmi veriyi değerlendir (karar artefaktı bunu 'incomplete_sweep: "
            "true' ile işaretler) — varsayılan davranış BUNU REDDEDER (bkz. "
            "supplement madde E2)"
        ),
    )
    args = parser.parse_args(argv)

    D = config.model_data_dir()
    sweep_path = D / "steering_sweep.jsonl"
    if not sweep_path.exists():
        print(f"BAŞARISIZ: {sweep_path} yok.\n"
              "  Önce scripts/08_steering_sweep.py çalıştırılmalı.", file=sys.stderr)
        return 2

    # E2: meta'ya HİÇ bakılmadan mevcut haliyle eksik bir sweep'ten karar
    # üretmek yanlış bir bilimsel sonuç yayınlamak demektir. Bu kontrol
    # `.jsonl`'i okumadan/gateway istemcisi kurulmadan ÖNCE yapılır — hem
    # daha ucuz hem operatöre daha erken teşhis verir.
    meta_path = D / "steering_sweep_meta.json"
    try:
        meta = read_sweep_meta(meta_path)
    except ValueError as exc:
        print(
            f"BAŞARISIZ: sweep meta'sı okunamadı.\n  {exc}\n"
            "  Önce scripts/08_steering_sweep.py TAMAMLANMIŞ olarak çalıştırılmalı.",
            file=sys.stderr,
        )
        return 2

    incomplete_sweep = not meta["complete"]
    if incomplete_sweep and not args.allow_incomplete:
        print(
            "BAŞARISIZ: sweep tamamlanmamış (complete: false) — "
            f"attempted {meta['attempted']}/{meta['planned']}.\n"
            "  B kriteri EKSİK veriden ÜRETİLEMEZ — bu yanlış bir bilimsel "
            "sonuç yayınlamak olurdu.\n"
            "  Önce scripts/08_steering_sweep.py'yi tamamlayın, ya da bilerek "
            "kısmi veriyi değerlendirmek için --allow-incomplete kullanın.",
            file=sys.stderr,
        )
        return 2
    if incomplete_sweep:
        print(
            "UYARI: KISMİ SWEEP DEĞERLENDİRİLİYOR (--allow-incomplete) — "
            f"attempted {meta['attempted']}/{meta['planned']}. Sonuç "
            "criterion_b.json içinde 'incomplete_sweep: true' olarak "
            "işaretlenecek.",
            file=sys.stderr,
        )

    records = []
    for number, line in enumerate(
        sweep_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except ValueError as exc:
            print(f"BAŞARISIZ: {sweep_path} satır {number} bozuk: {exc}", file=sys.stderr)
            return 2

    groups = group_by_layer_strength(records)
    # "üst sınır": böl-ve-kurtar hücre başına EK gönderim harcayabilir (bkz.
    # `_classify_batch`) — E4'ün +10 payı bunun için. Bu sayı retry/bölünme
    # OLMADAN gereken minimum çağrıdır, gerçek harcamanın kesin üst sınırı
    # DEĞİLDİR.
    planned = sum((len(v) + args.batch_size - 1) // args.batch_size for v in groups.values())

    try:
        client = build_default_client()
    except RuntimeError as exc:
        print(f"BAŞARISIZ: gateway istemcisi kurulamadı.\n  {exc}", file=sys.stderr)
        return 2

    stage_left, global_left = client.remaining_budget(STAGE)
    print(f"Grup sayısı: {len(groups)}   planlanan çağrı (üst sınır): {planned}")
    print(f"Aşama kalan: {stage_left}   global kalan: {global_left}")
    if args.dry_run:
        if planned > stage_left or planned > global_left:
            print("HATA: plan kalan bütçeye sığmıyor.", file=sys.stderr)
            return 2
        return 0
    if planned > stage_left or planned > global_left:
        print("BAŞARISIZ: plan kalan bütçeye sığmıyor — koşu başlatılmadı.", file=sys.stderr)
        return 2

    labels_path = D / "steering_labels.json"
    # Artımlı kalıcılığın OKUMA ucu: önceki bir koşudan kalma etiketler
    # varsa yükle — bu hücreler/konumlar aşağıdaki döngüde TEKRAR
    # sorulmayacak. Dosya bozuksa (sıfırdan başlamak önceden ÖDENMİŞ
    # etiketleri sessizce silmek demektir) BAŞARISIZ olunur.
    try:
        group_state, unlabelled_state = load_group_labels(labels_path)
    except (ValueError, KeyError) as exc:
        print(
            f"BAŞARISIZ: {labels_path} bozuk — önceki koşudan kalma etiketler "
            "okunamıyor.\n"
            f"  Ayrıntı: {exc}\n"
            "  Dosyayı elle onarın ya da (var olan etiketleri kaybetmeyi göze "
            "alarak) silin.",
            file=sys.stderr,
        )
        return 2

    def _group_done(key: tuple[int, float]) -> bool:
        total = len(groups[key])
        done = len(group_state.get(key, {})) + len(unlabelled_state.get(key, set()))
        return done >= total

    already_done = sum(1 for key in groups if _group_done(key))
    if already_done:
        print(
            f"{already_done}/{len(groups)} grup diskten yüklendi ({labels_path}) — "
            "bu gruplar tekrar hakeme sorulmayacak."
        )

    batches_split_total = 0
    try:
        for key in sorted(groups):
            items_all = [(r["question"], r["answer"]) for r in groups[key]]
            group_state.setdefault(key, {})
            unlabelled_state.setdefault(key, set())
            pending = [
                pos for pos in range(len(items_all))
                if pos not in group_state[key] and pos not in unlabelled_state[key]
            ]
            for start in range(0, len(pending), args.batch_size):
                chunk_positions = pending[start : start + args.batch_size]
                chunk_items = [items_all[pos] for pos in chunk_positions]
                chunk_labels, chunk_unlabelled, split = _classify_batch(
                    client, positions=chunk_positions, items=chunk_items, stage=STAGE
                )
                group_state[key].update(chunk_labels)
                unlabelled_state[key].update(chunk_unlabelled)
                batches_split_total += split
                # Artımlı kalıcılığın YAZMA ucu: HER hakem batch'inden sonra
                # diske yaz — bir çökme/kesinti bu yüzden en fazla BİR
                # batch'lik (<=`args.batch_size` öğe) işi kaybettirir.
                save_group_labels(labels_path, group_state, unlabelled_state)
            done_now = sum(1 for k in groups if _group_done(k))
            print(f"\r  {done_now}/{len(groups)} grup", end="", flush=True)
    except (BudgetExceeded, CircuitOpen, BudgetCorrupted) as exc:
        print(f"\nDURDURULDU: {exc}", file=sys.stderr)
        print(
            f"  Bu ana kadar toplanan etiketler {labels_path}'e kalıcı olarak "
            "yazıldı — tekrar koşuda bu hücreler/konumlar tekrar sorulmayacak.",
            file=sys.stderr,
        )
        return 2
    except GatewayError as exc:
        # `JudgeParseError` ARTIK bu grupta DEĞİL (06'daki AYNI düzeltme,
        # bkz. supplement madde E1): `_classify_batch` onu tek başına düşen
        # bir öğe için bile hep KENDİ İÇİNDE yakalayıp "etiketlenemedi"
        # sayar, hiçbir zaman burada YÜKSELMEZ.
        print(f"\nBAŞARISIZ: {exc}", file=sys.stderr)
        print(
            f"  Bu ana kadarki ilerleme {labels_path}'e kalıcı olarak yazıldı.",
            file=sys.stderr,
        )
        return 2

    print()

    labels_by_group: dict[tuple[int, float], list[str]] = {
        key: list(group_state.get(key, {}).values()) for key in groups
    }
    unlabelled_by_group: dict[tuple[int, float], int] = {
        key: len(unlabelled_state.get(key, set())) for key in groups
    }
    total_unlabelled = sum(unlabelled_by_group.values())
    if total_unlabelled:
        print(
            f"UYARI: toplam {total_unlabelled} öğe hakem tarafından (tek tek "
            "sorulsa bile) etiketlenemedi — hücre başına döküm "
            "criterion_b.json içindeki 'unlabelled_by_group' alanında.",
            file=sys.stderr,
        )

    rates = rates_by_layer(labels_by_group)
    verdicts = evaluate_all_layers(rates)

    out_dir = config.model_results_dir() / "steering"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rate_by_strength.json").write_text(json.dumps(
        {str(l): {str(s): r for s, r in by.items()} for l, by in rates.items()},
        ensure_ascii=False, indent=2), encoding="utf-8")

    criterion_payload: dict = {
        "model": config.TARGET_MODEL,
        "layers": {str(l): v for l, v in verdicts.items()},
        "unlabelled_by_group": {
            _group_key(key): unlabelled_by_group[key] for key in sorted(groups)
        },
        "note": (
            "Kriter KATMAN BAŞINA değerlendirildi. Eşik ve taban ön-tescildeki "
            "hâliyle sabit: en negatif güçteki Assistant-dışı oran, 0.0 "
            "gücündeki orandan en az 25 puan yüksek olmalı. PAYDA: her "
            "hücrenin oranı yalnızca hakemin ETİKETLEYEBİLDİĞİ öğeler "
            "üzerinden hesaplanır — hakemin (tek tek sorulsa bile) "
            "ayrıştıramadığı öğeler o hücrenin paydasından DÜŞER; kaç öğenin "
            "düştüğü 'unlabelled_by_group'ta hücre başına kayıtlıdır."
        ),
    }
    if incomplete_sweep:
        criterion_payload["incomplete_sweep"] = True
        criterion_payload["sweep_attempted"] = meta["attempted"]
        criterion_payload["sweep_planned"] = meta["planned"]
    (out_dir / "criterion_b.json").write_text(
        json.dumps(criterion_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print()
    for layer in sorted(verdicts):
        v = verdicts[layer]
        print(f"L{layer}: taban {v['baseline_rate']:.3f} → "
              f"en uzak ({v['far_strength']}) {v['far_rate']:.3f}   "
              f"artış {100*v['delta']:+.1f} puan   "
              f"{'GEÇTİ' if v['passed'] else 'DÜŞTÜ'}")
        if not v["passed"]:
            print(f"     {v['reason']}")
    print()
    code = overall_exit_code(verdicts)
    if code == 0:
        print("B KRİTERİ: HER KATMANDA GEÇTİ")
    elif code == 1:
        print("B KRİTERİ: EN AZ BİR KATMANDA DÜŞTÜ")
    else:
        print("B KRİTERİ: KARAR ÜRETİLEMEDİ (hiçbir katman değerlendirilemedi)")
    print(f"Bölünüp kurtarılmaya çalışılan batch: {batches_split_total}")
    print(f"Gönderilen istek: {client.sends_made}")
    return code


def main(argv: list[str] | None = None) -> int:
    """Tanı sarmalayıcısı — `07_extract_axis.py:609-637` / `08_steering_
    sweep.py::main` ile AYNI desen: gerçek gövde `_run()`'da, burada
    yalnızca ÖNGÖRÜLMEMİŞ bir istisnayı (ör. `evaluate_criterion_b`'nin
    guard `ValueError`'larından biri, ya da bir I/O hatası) yakalayıp temiz
    bir Türkçe teşhisle çıkış kodu 2 döndürür.

    Bu, `susceptibility.py`'nin guard'larının (boş etiket listesi, 0.0
    taban eksik, sonlu olmayan oran, negatif güç ölçümü yok) HİÇBİRİNİN
    çıplak bir traceback'e ya da — daha kötüsü — çıkış kodu 1'e ("B KRİTERİ
    DÜŞTÜ" anlamına gelir) DÖNÜŞMEMESİNİ garanti eder: exit 1 bu script'te
    YALNIZCA "kriter fiilen değerlendirildi ve düştü" demektir, bir kurulum/
    girdi hatası değil.

    `KeyboardInterrupt` BİLEREK ayrı tutulur ve yeniden fırlatılır:
    operatörün Ctrl-C'si bir "BAŞARISIZ" tanısına dönüşmemeli.
    """
    try:
        return _run(argv)
    except KeyboardInterrupt:
        raise
    except Exception as exc:  # noqa: BLE001 — kasıtlı geniş yakalama, gerekçe docstring'de
        print(f"BAŞARISIZ: beklenmeyen hata — bu bir B kriteri kararı DEĞİLDİR.\n"
              f"  {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
