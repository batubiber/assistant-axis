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

Etiket dosyasının sweep'e bağlanması (2026-08-10 Fix Round 1; bkz.
`.superpowers/sdd/p3-task-5-fix1-brief.md`, madde F1): `steering_labels.json`
artık sweep'in KİMLİĞİNİ üç ayrı alanla taşır — `sweep_sha256` (dosya
baytlarının sha256'sı), `axis_run_id` (meta'dan, köken kaydı — TEK BAŞINA
YETMEZ: aktivasyon indeksinden gelir, sweep'in KENDİSİNDEN değil; aynı
eksenle YENİDEN üretilen bir sweep aynı `axis_run_id`'yi taşır ama üretim
`do_sample=True, temperature=1.0` olduğu için TÜM yanıtlar yeni metindir) ve
`record_counts` (hücre başına kayıt sayısı). Yükleme anında üçü de
DOĞRULANIR; herhangi biri uyuşmuyorsa (ya da dosya bu üç alanı hiç
taşımıyorsa — eski şema) yüklenen durum KULLANILMAZ: dosya `.stale`
uzantısıyla kenara alınır (sessizce SİLİNMEZ), stderr'e büyük harfli bir
UYARI basılır, ve koşu SIFIRDAN başlar. `_group_done` artık SAYIYA değil
KAPSAMA bakar (`set(labels) | set(unlabelled) >= set(range(total))`) ve
oranın hesaplandığı `labels_by_group`/`unlabelled_by_group` o hücrenin
`range(len(items_all))` aralığına KIRPILIR — bayat/sızmış bir pozisyon
diskte kalsa bile orana giremez.
"""
from __future__ import annotations

import argparse
import hashlib
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


# F1 (Fix Round 1): `steering_labels.json` artık sweep'in KİMLİĞİNİ taşır —
# `_MISSING`, "bu alan JSON'da hiç yoktu" ile "bu alanın değeri `None`'dı"
# ayrımını yapabilmek için ayrı bir sentinel (payload.get(key, None) ikisini
# de `None` yapardı; eski şema tespiti bu ayrıma dayanır).
_MISSING = object()


def save_group_labels(
    path: str | Path,
    group_state: dict[tuple[int, float], dict[int, str]],
    unlabelled_state: dict[tuple[int, float], set[int]],
    *,
    sweep_sha256: str,
    axis_run_id,
    record_counts: dict[tuple[int, float], int],
) -> None:
    """`steering_labels.json`'ı ATOMİK yaz — `06_label_and_train_probe.py::
    save_labels`'ın AYNI deseni (tempfile + `os.replace`). Çağıran taraf bunu
    HER hakem batch'inden sonra çağırır: bir çökme/kesinti en fazla BİR
    batch'lik işi kaybettirir, o ana kadar toplanan etiketlerin TAMAMINI
    değil. Yazım ortasında kesilme dosyayı KIRPMAZ.

    F1: `sweep_sha256`/`axis_run_id`/`record_counts` dosyaya birlikte
    gömülür — bunlar `load_group_labels`'ın bir SONRAKİ koşuda bu durumun
    HANGİ sweep'e ait olduğunu doğrulayabilmesi için gereken parmak izi.
    `axis_run_id` TEK BAŞINA yetmez (aktivasyon indeksinden gelir, sweep'in
    kendisinden değil); asıl doğrulama `sweep_sha256`'dır."""
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
        "sweep_sha256": sweep_sha256,
        "axis_run_id": axis_run_id,
        "record_counts": {
            _group_key(key): count for key, count in record_counts.items()
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


def load_group_labels(path: str | Path) -> tuple[
    dict[tuple[int, float], dict[int, str]],
    dict[tuple[int, float], set[int]],
    dict | None,
]:
    """Var olan `steering_labels.json`'ı yükle — bu, artımlı kalıcılığın OKUMA
    ucudur (06'nın `load_existing_labels`'ıyla AYNI rol): önceki bir koşu
    kesintiye uğradıysa tamamlanmış hücreler/konumlar TEKRAR hakeme
    SORULMAZ. Dosya yoksa (ilk koşu) `(({}, {}, None)` döner — `None`
    üçüncü öğe, "karşılaştırılacak bir parmak izi yok" anlamına gelir.

    Dosya VARSA ama ayrıştırılamıyorsa ya da beklenen `labels` anahtarını
    taşımıyorsa istisna SARMALANMADAN çağırana yükselir — sessizce sıfırdan
    başlamak, zaten ÖDENMİŞ etiketleri sessizce silip aynı öğeleri yeniden
    ücretlendirmek demektir (06 ile aynı gerekçe). Bu, F1'in parmak izi
    UYUŞMAZLIĞINDAN (bozuk değil, YANLIŞ sweep'e ait) AYRI bir durumdur —
    o, çağıran tarafından `_fingerprint_mismatch_reason` ile ayrıca ele
    alınır, burada değil.

    Dönüşün üçüncü öğesi (`fingerprint`) dosyadaki ham `sweep_sha256`/
    `axis_run_id`/`record_counts` alanlarını taşır; bir alan JSON'da hiç
    yoksa (eski şema) değeri `_MISSING` olur — `None` ile KARIŞTIRILMAZ,
    çünkü `axis_run_id`'nin GERÇEK değeri de `None` olabilir (meta'da hiç
    yazılmamışsa)."""
    path = Path(path)
    if not path.exists():
        return {}, {}, None
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
    raw_counts = payload.get("record_counts", _MISSING)
    record_counts = (
        {_parse_group_key(k): v for k, v in raw_counts.items()}
        if raw_counts is not _MISSING
        else _MISSING
    )
    fingerprint = {
        "sweep_sha256": payload.get("sweep_sha256", _MISSING),
        "axis_run_id": payload.get("axis_run_id", _MISSING),
        "record_counts": record_counts,
    }
    return group_state, unlabelled_state, fingerprint


def _fingerprint_mismatch_reason(
    fingerprint: dict | None,
    *,
    sweep_sha256: str,
    axis_run_id,
    record_counts: dict[tuple[int, float], int],
) -> str | None:
    """`fingerprint` (bkz. `load_group_labels`) gerçek sweep'in parmak
    izleriyle UYUŞUYOR mu? Uyuşuyorsa (ya da karşılaştırılacak bir dosya
    hiç yoksa) `None`, uyuşmuyorsa NEDENİNİ açıklayan büyük harfli bir Türkçe
    metin döner — çağıran bunu hem stderr'e UYARI olarak basar hem de
    kararına (durumu at, sıfırdan başla) temel yapar.

    Sıra ÖNEMLİ değil ama OKUNABİLİR olsun diye: önce eski şema, sonra
    sweep'in kendi baytları (en güçlü kanıt), sonra köken kaydı, en son
    hücre kayıt sayıları (ikinci bir savunma katmanı — bkz. F1 madde 4:
    `sweep_sha256` eşleşse bile `steering_labels.json` elle/başka bir
    yoldan bozulmuş olabilir)."""
    if fingerprint is None:
        return None
    missing = [
        name
        for name in ("sweep_sha256", "axis_run_id", "record_counts")
        if fingerprint[name] is _MISSING
    ]
    if missing:
        return f"ESKİ ŞEMA — parmak izi alanları hiç yok: {', '.join(missing)}"
    if fingerprint["sweep_sha256"] != sweep_sha256:
        return (
            "SWEEP DEĞİŞMİŞ — sweep_sha256 uyuşmuyor "
            f"(dosyada {fingerprint['sweep_sha256']!r}, gerçek dosyada {sweep_sha256!r}); "
            "steering_sweep.jsonl yeniden üretilmiş olabilir (do_sample=True — "
            "aynı axis_run_id ALTINDA bile yanıtlar TAMAMEN yeni metindir)"
        )
    if fingerprint["axis_run_id"] != axis_run_id:
        return (
            "AXIS_RUN_ID UYUŞMUYOR — "
            f"dosyada {fingerprint['axis_run_id']!r}, meta'da {axis_run_id!r}"
        )
    if fingerprint["record_counts"] != record_counts:
        return "HÜCRE KAYIT SAYILARI UYUŞMUYOR — sweep'in içeriği değişmiş"
    return None


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

    # F1: sweep'in KENDİ BAYTLARININ sha256'sı — parmak izinin en güçlü
    # parçası. Dosya ~1.7 MB, maliyet ihmal edilebilir. Baytlar bir kez
    # okunur; hem hash hem satır ayrıştırması AYNI `sweep_bytes`'tan gelir
    # (dosyayı iki kez okumaktan kaçınmak için).
    sweep_bytes = sweep_path.read_bytes()
    sweep_sha256 = hashlib.sha256(sweep_bytes).hexdigest()
    # F1: `axis_run_id` aktivasyon indeksinden gelir (bkz.
    # `08_steering_sweep.py:200`, `index.get("run_id")`) — sweep'in KENDİSİNDEN
    # değil. TEK BAŞINA köken kaydıdır, sweep'i yeniden üretmek bunu
    # DEĞİŞTİRMEZ; asıl doğrulama `sweep_sha256`'dır.
    axis_run_id = meta.get("axis_run_id")

    records = []
    for number, line in enumerate(
        sweep_bytes.decode("utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except ValueError as exc:
            print(f"BAŞARISIZ: {sweep_path} satır {number} bozuk: {exc}", file=sys.stderr)
            return 2

    groups = group_by_layer_strength(records)
    # F1: hücre başına kayıt sayısı — parmak izinin üçüncü ve son parçası
    # (bkz. `_fingerprint_mismatch_reason`'ın madde 4 gerekçesi: sha256
    # eşleşse bile `steering_labels.json` ayrıca bozulmuş/elle değiştirilmiş
    # olabilir; bu ikinci, bağımsız bir savunma katmanıdır).
    record_counts = {key: len(v) for key, v in groups.items()}

    labels_path = D / "steering_labels.json"
    # F2: OKUMA ucu artık bütçe kapısının ÜSTÜNDE — diskte zaten hazır
    # (ve maliyeti 0 olan) hücreler/konumlar `planned` hesabına HİÇ
    # girmemeli, aksi hâlde her resume denemesi tüm sweep'in bütçesini
    # yeniden istermiş gibi davranıp gereksiz yere BAŞARISIZ olur (bkz.
    # fix1 brief, madde F2). Dosya bozuksa (sıfırdan başlamak önceden
    # ÖDENMİŞ etiketleri sessizce silmek demektir) BAŞARISIZ olunur — bu,
    # F1'in parmak izi UYUŞMAZLIĞINDAN (aşağıda) AYRI bir durum.
    try:
        group_state, unlabelled_state, fingerprint = load_group_labels(labels_path)
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

    # F1: yükleme zamanı doğrulama — parmak izlerinden HERHANGİ biri
    # uyuşmuyorsa (ya da dosya bunları hiç taşımıyorsa — eski şema) yüklenen
    # durum KULLANILMAZ. Eski dosya sessizce SİLİNMEZ, `.stale` uzantısıyla
    # kenara alınır; koşu sıfırdan başlar (BAŞARISIZ DEĞİL — bu kurtarılabilir
    # bir durum, yalnızca ekstra hakem çağrısına mal olur).
    mismatch_reason = _fingerprint_mismatch_reason(
        fingerprint,
        sweep_sha256=sweep_sha256,
        axis_run_id=axis_run_id,
        record_counts=record_counts,
    )
    if mismatch_reason is not None:
        stale_path = labels_path.with_name(labels_path.name + ".stale")
        labels_path.replace(stale_path)
        print(
            "UYARI: ESKİ ETİKETLER SWEEP İLE UYUŞMUYOR — durum atıldı, "
            "SIFIRDAN BAŞLANACAK.\n"
            f"  Sebep: {mismatch_reason}\n"
            f"  Eski dosya SİLİNMEDİ, {stale_path}'e taşındı.",
            file=sys.stderr,
        )
        group_state, unlabelled_state = {}, {}

    def _pending_positions(key: tuple[int, float]) -> list[int]:
        total = len(groups[key])
        done = set(group_state.get(key, {})) | set(unlabelled_state.get(key, set()))
        return [pos for pos in range(total) if pos not in done]

    # F2: `planned` artık YALNIZCA bekleyen (henüz etiketlenmemiş/
    # etiketlenemez damgalanmamış) pozisyonlar üzerinden hesaplanır — diskte
    # hazır olan hiçbir şey bütçe kapısına girmez. "üst sınır": böl-ve-kurtar
    # hücre başına EK gönderim harcayabilir (bkz. `_classify_batch`) — E4'ün
    # +10 payı bunun için. Bu sayı retry/bölünme OLMADAN gereken minimum
    # çağrıdır, gerçek harcamanın kesin üst sınırı DEĞİLDİR.
    planned = sum(
        (len(_pending_positions(key)) + args.batch_size - 1) // args.batch_size
        for key in groups
    )
    # Yalnızca EKRANDA göstermek için: resume'suz (tüm sweep) toplam plan —
    # operatör bekleyenle toplam arasındaki farktan ne kadarının diskten
    # geldiğini görebilsin.
    total_planned = sum(
        (len(v) + args.batch_size - 1) // args.batch_size for v in groups.values()
    )

    try:
        client = build_default_client()
    except RuntimeError as exc:
        print(f"BAŞARISIZ: gateway istemcisi kurulamadı.\n  {exc}", file=sys.stderr)
        return 2

    stage_left, global_left = client.remaining_budget(STAGE)
    print(f"Grup sayısı: {len(groups)}   toplam planlanan çağrı (üst sınır): {total_planned}"
          f"   bekleyen (bütçe kontrolü buna göre): {planned}")
    print(f"Aşama kalan: {stage_left}   global kalan: {global_left}")
    if args.dry_run:
        if planned > stage_left or planned > global_left:
            print("HATA: plan kalan bütçeye sığmıyor.", file=sys.stderr)
            return 2
        return 0
    if planned > stage_left or planned > global_left:
        print("BAŞARISIZ: plan kalan bütçeye sığmıyor — koşu başlatılmadı.", file=sys.stderr)
        return 2

    def _group_done(key: tuple[int, float]) -> bool:
        # F1 madde 3: SAYIYA değil KAPSAMA bak — `>=` bir SAYI karşılaştırması
        # olduğunda, yanlış hücreye ait (ama sayıca yeterli) bayat etiketler
        # o hücreyi de "tamam" ilan edebilirdi. Küme karşılaştırması bunu
        # ÖNLER: yalnızca gerçekten `range(total)`'ın HER pozisyonu
        # kapsanmışsa `True` döner.
        total = len(groups[key])
        covered = set(group_state.get(key, {})) | set(unlabelled_state.get(key, set()))
        return covered >= set(range(total))

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
            pending = _pending_positions(key)
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
                # batch'lik (<=`args.batch_size` öğe) işi kaybettirir. F1:
                # her yazımda GÜNCEL parmak izi de gömülür (sabit — sweep
                # koşu boyunca değişmez), böylece bir sonraki koşu bu
                # dosyanın HANGİ sweep'e ait olduğunu doğrulayabilir.
                save_group_labels(
                    labels_path, group_state, unlabelled_state,
                    sweep_sha256=sweep_sha256, axis_run_id=axis_run_id,
                    record_counts=record_counts,
                )
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

    # F1 madde 4: KIRPMA — `group_state`/`unlabelled_state` fingerprint
    # eşleşse bile (ya da eski bir koşudan kalan bir hücrede) `range(total)`
    # DIŞINDA bir pozisyon barındırabilir; bu döngü o pozisyonları oranın
    # PAYDASINA hiç sokmadan eler. `0 <= pos < total` — pozisyonlar hep
    # `range(len(items_all))`'dan geldiği için negatif olmaz, ama savunma
    # yine de açıkça yazılır.
    labels_by_group: dict[tuple[int, float], list[str]] = {}
    unlabelled_by_group: dict[tuple[int, float], int] = {}
    for key in groups:
        total = len(groups[key])
        labels_by_group[key] = [
            label for pos, label in group_state.get(key, {}).items() if 0 <= pos < total
        ]
        unlabelled_by_group[key] = sum(
            1 for pos in unlabelled_state.get(key, set()) if 0 <= pos < total
        )
    # F3: hücre başına PAYDA — kaç öğenin oranı hesaplamaya girdiği (kırpmadan
    # SONRAKİ gerçek sayı). `criterion_b.json`'a yazılmadan bu alan olmadan,
    # 250 öğeden gelen bir oran ile 4 öğeden gelen bir oran artefaktta bayt
    # bayt AYNI görünürdü.
    labelled_by_group: dict[tuple[int, float], int] = {
        key: len(labels_by_group[key]) for key in groups
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

    # F5: karar artefaktları yalnızca EN AZ BİR katman fiilen
    # değerlendirildiyse yazılır. Sıfır kayıtlı bir sweep (ya da hiçbir
    # katmanın değerlendirilemediği başka bir durum) `verdicts`'i boş
    # bırakır; bu artık önceki GERÇEK `criterion_b.json`'ı EZMEZ — script'in
    # kendi konvansiyonuyla (TÜM diğer çıkış-2 yolları yazımlardan ÖNCE
    # döner) tutarlı hâle gelir.
    if verdicts:
        out_dir = config.model_results_dir() / "steering"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "rate_by_strength.json").write_text(json.dumps(
            {str(l): {str(s): r for s, r in by.items()} for l, by in rates.items()},
            ensure_ascii=False, indent=2), encoding="utf-8")

        criterion_payload: dict = {
            "model": config.TARGET_MODEL,
            "layers": {str(l): v for l, v in verdicts.items()},
            "labelled_by_group": {
                _group_key(key): labelled_by_group[key] for key in sorted(groups)
            },
            "unlabelled_by_group": {
                _group_key(key): unlabelled_by_group[key] for key in sorted(groups)
            },
            "note": (
                "Kriter KATMAN BAŞINA değerlendirildi. Eşik ve taban "
                "ön-tescildeki hâliyle sabit: en negatif güçteki "
                "Assistant-dışı oran, 0.0 gücündeki orandan en az 25 puan "
                "yüksek olmalı. PAYDA: her hücrenin oranı yalnızca hakemin "
                "ETİKETLEYEBİLDİĞİ öğeler üzerinden hesaplanır; bu payda "
                "hücre başına 'labelled_by_group' alanında AÇIKÇA "
                "kayıtlıdır — hakemin (tek tek sorulsa bile) "
                "ayrıştıramadığı öğeler paydadan DÜŞER, kaç öğenin düştüğü "
                "'unlabelled_by_group'ta hücre başına kayıtlıdır."
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
