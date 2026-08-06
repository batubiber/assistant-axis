#!/usr/bin/env python3
"""Aşama 2 — hakem etiketleri topla, probe eğit, 16.000 rollout'u etiketle.

Kullanım:
    uv run python scripts/06_label_and_train_probe.py --dry-run
    uv run --extra ml python scripts/06_label_and_train_probe.py
    uv run --extra ml python scripts/06_label_and_train_probe.py --allow-pilot  # bilinçli pilot

ÇIKIŞ KODLARI:
    0  probe güvenilir, role_expression.json yazıldı
    1  BULGU: probe held-out uyumu eşiğin altında — güvenilmez, geri çekilme
       kuralı devreye girer (bu bir ÖLÇÜM sonucudur, bir çalıştırma hatası değil)
    2  koşulamadı: ön koşul/kurulum hatası, bütçe, gateway, --dry-run planı
       kalan bütçeye sığmıyor, rollouts.jsonl eksik/bozuk/PİLOT (--allow-pilot yoksa)

`05_capture_activations.py` ile aynı `--allow-pilot` deseni: `rollouts.jsonl`
`04`'ün bir `--limit` koşusundan geliyorsa (`rollouts_meta.json` künyesi
bunu taşır), bu script varsayılan olarak REDDEDER — aksi hâlde hakem
harcamasının (~200 çağrı, aşama bütçesinin çoğu) bir duman testi üzerinde
yapılması hiçbir şeyle engellenmezdi.

`1` bilinçli olarak TEK bir anlama ayrılmıştır. Eskiden `--dry-run`'ın bütçe
reddi de 1 döndürüyordu; bir kabuk pipeline'ı "probe güvenilmez" ile "koşu
hiç başlamadı"yı ayırt edemiyordu.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

from aax import config
from aax.gateway import BudgetExceeded, CircuitOpen, GatewayError, build_default_client
from aax.judge import (
    SCORE_MAX_TOKENS,
    SCORE_TEMPERATURE,
    JudgeParseError,
    build_role_score_prompts,
    score_role_expression,
)
from aax.probe import RoleExpressionProbe, embed_answers, stratified_sample
from aax.prompts import load_role_catalog
from aax.rollouts import load_rollouts_meta, read_rollouts, rollouts_run_id

STAGE = "stage2_probe_labels"
SEED = 20260806
LABEL_SAMPLE_SIZE = 2000
LABELS_PATH = config.DATA_DIR / "probe_labels.json"
OUT_PATH = config.DATA_DIR / "role_expression.json"


def collapse(score: int) -> str:
    if score == 3:
        return "fully"
    if score == 2:
        return "somewhat"
    if score in (0, 1):
        return "no"
    raise ValueError(f"Puan 0-3 aralığı dışında: {score!r}")


def run_dry_run(client, by_role: dict[str, list[int]], records: list[dict], catalog: dict) -> int:
    """İstek atmadan planı KALAN bütçeyle kıyasla.

    `scripts/00_generate_role_data.py::run_dry_run` ile AYNI desen. Eski
    sürüm iki ayrı hata yapıyordu:

    1. Plan, aşama TAVANIYLA (`config.STAGE_BUDGETS[STAGE]`) kıyaslanıyordu.
       Tavanın çoğunu önceki bir koşuda harcamış bir operatör temiz bir `0`
       görüp koşuyu başlatır, bütçe biter bitmez ortasında kesilirdi.
       `GatewayClient.remaining_budget()` tam bu iş için var ve docstring'i
       bunu açıkça söylüyor.
    2. Plan cache'i yok sayıyordu (`(len(rows) + 9) // 10`). `would_call()`
       cache'te olanı sayMAZ — gerçekten HARCANACAK çağrı sayısı budur.

    BİRİM UYARISI korunuyor: bütçe sayacı **HTTP gönderimi** sayar, mantıksal
    çağrı değil (bkz. `config.STAGE_BUDGETS` yorumu). Bir mantıksal çağrı
    retry'larla 2-3 gönderim harcayabilir; 300'lük tavanın 50'si zaten bu pay
    içindir. Yani `planlanan <= kalan` GEREK koşuldur, YETER koşul değildir —
    çıktı bunu açıkça yazar.
    """
    planned = 0
    cached = 0
    for role, rows in by_role.items():
        items = [(records[i]["question"], records[i]["answer"]) for i in rows]
        for prompt in build_role_score_prompts(
            role=role, description=catalog[role], items=items
        ):
            if client.would_call(
                [{"role": "user", "content": prompt}],
                temperature=SCORE_TEMPERATURE,
                max_tokens=SCORE_MAX_TOKENS,
            ):
                planned += 1
            else:
                cached += 1

    stage_remaining, global_remaining = client.remaining_budget(STAGE)
    stage_cap = config.STAGE_BUDGETS[STAGE]

    print(f"Planlanan çağrı:      {planned} (cache'te: {cached})")
    print(f"Aşama bütçesi:        {stage_cap} (kalan: {stage_remaining})")
    print(f"Global tavan:         {config.GLOBAL_BUDGET} (kalan: {global_remaining})")
    print(
        "Not: bütçe sayacı HTTP gönderimi sayar, mantıksal çağrı değil — "
        "planın kalana sığması gerek koşuldur, yeter koşul değil (retry payı)."
    )

    sorunlar = []
    if planned > stage_remaining:
        sorunlar.append(
            f"aşama bütçesi: {planned} planlandı, yalnızca {stage_remaining} kaldı "
            f"({stage_cap} tavanın {stage_cap - stage_remaining}'i harcanmış)"
        )
    if planned > global_remaining:
        sorunlar.append(
            f"global tavan: {planned} planlandı, yalnızca {global_remaining} kaldı "
            f"({config.GLOBAL_BUDGET} tavanın {config.GLOBAL_BUDGET - global_remaining}'i "
            "harcanmış)"
        )
    if sorunlar:
        for sorun in sorunlar:
            print(f"HATA: {sorun}", file=sys.stderr)
        print(
            "Koşu başlatılmadı — --sample-size küçült. Tavan yükseltilmez.",
            file=sys.stderr,
        )
        # Çıkış 2: bu bir ÖN KOŞUL hatasıdır, bir bulgu değil. Çıkış 1 bu
        # script'te tek bir şey demek: "probe güvenilmez" (bkz. modül
        # docstring'i).
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sample-size", type=int, default=LABEL_SAMPLE_SIZE)
    parser.add_argument(
        "--allow-pilot",
        action="store_true",
        help=(
            "rollouts.jsonl bir --limit koşusundan gelse bile devam et "
            "(05_capture_activations.py'nin --allow-pilot'ıyla aynı bilinçli geçersiz kılma)"
        ),
    )
    args = parser.parse_args(argv)

    rollouts_path = config.DATA_DIR / "rollouts.jsonl"
    # `05_capture_activations.py::main` ile aynı desen: çıplak `read_rollouts`
    # çağrısı `FileNotFoundError`/`ValueError`'ı sarmasız bırakıyordu — biri
    # yorumlayıcıyı çıplak bir traceback'le, ikisi de görev tanımının
    # istediği "traceback değil temiz tanı" koşulunu ihlal ederek çıkış 1
    # (bu script'te "probe güvenilmez" anlamına gelen kod) ile döndürüyordu.
    try:
        records = read_rollouts(rollouts_path)
    except FileNotFoundError:
        print(
            f"BAŞARISIZ: {rollouts_path} yok.\n"
            "  Bu dosya Aşama 1 üretiminin çıktısıdır — önce "
            "scripts/04_generate_rollouts.py çalıştırılmalı.",
            file=sys.stderr,
        )
        return 2
    except ValueError as exc:
        print(f"BAŞARISIZ: {rollouts_path} okunamadı.\n  {exc}", file=sys.stderr)
        return 2

    # Önemli 5: `05`'e bağlı olmadan `rollouts.jsonl`'ı DOĞRUDAN okuyordu —
    # künye kontrolü yalnızca `05`'e bağlıydı. Bu, hakem harcamasının
    # (~200 çağrı, 300'lük aşama bütçesinin çoğu) bir PİLOT künye üzerinde de
    # yapılabileceği anlamına geliyordu; hiçbir şey bunu harcamadan önce
    # reddetmiyordu. `05`'in `--allow-pilot` deseninin AYNISI.
    try:
        meta = load_rollouts_meta(
            config.DATA_DIR / "rollouts_meta.json", records, allow_pilot=args.allow_pilot
        )
    except ValueError as exc:
        print(f"BAŞARISIZ: rollout kümesi kanonik değil.\n  {exc}", file=sys.stderr)
        return 2
    if meta.get("limit") is not None:
        print(
            f"UYARI: PİLOT rollout kümesi (--limit={meta['limit']}, {meta['n']} kayıt) "
            "--allow-pilot ile kabul edildi. Bu etiketler A kriteri için kullanılamaz."
        )

    role_rows = [i for i, r in enumerate(records) if r["kind"] == "role"]
    role_records = [records[i] for i in role_rows]

    # Kurulum aşamasındaki her hata `build_default_client()` çevresindeki
    # sarmalayıcıyla aynı desende ele alınır: çıplak bir traceback yerine
    # anlaşılır bir Türkçe tanı ve sıfırdan farklı bir çıkış kodu (2).
    try:
        chosen_local = stratified_sample(role_records, n=args.sample_size, seed=SEED)
    except ValueError as exc:
        print(
            "BAŞARISIZ: örnekleme kurulamadı.\n"
            f"  İstenen örnek boyutu {args.sample_size}, mevcut rol satırı sayısı "
            f"{len(role_records)}.\n"
            "  --sample-size ile mevcut popülasyona sığan daha küçük bir değer verin.\n"
            f"  Ayrıntı: {exc}",
            file=sys.stderr,
        )
        return 2
    chosen = [role_rows[i] for i in chosen_local]

    # load_role_catalog üzerinden: kısmi/pilot bir katalogla etiketleme yapmak,
    # yanlış rol kümesi üzerinde probe eğitmek demek olurdu.
    try:
        catalog = {
            r["role"]: r["description"]
            for r in load_role_catalog(config.DATA_DIR / "roles.json")
        }
    except ValueError as exc:
        print(
            "BAŞARISIZ: rol kataloğu kanonik değil.\n"
            f"  {exc}\n"
            "  Aşama 0'ı (scripts/00_generate_role_data.py) --allow-partial "
            "OLMADAN tamamlayıp tekrar deneyin.",
            file=sys.stderr,
        )
        return 2

    by_role: dict[str, list[int]] = defaultdict(list)
    for row in chosen:
        by_role[records[row]["role"]].append(row)

    # Fail-closed: yukarıdaki `load_role_catalog` kanoniklik doğrulaması rol
    # KÜMESİNİ değil rol İSİMLERİNİN eksiksizliğini garantiler; `rollouts.jsonl`
    # farklı (ör. daha eski/pilot) bir katalogdan üretilmiş olabilir. Örneklenen
    # bir rol katalogda yoksa sessizce jenerik bir açıklama uydurmak (eskiden:
    # `catalog.get(role, f"the role of a {role}")`) yanlış rol kümesi üzerinde
    # hakemlik yapmak demektir — bu, hemen üstteki yorumun reddettiği tam olarak
    # aynı hata sınıfıdır.
    missing_roles = sorted(role for role in by_role if role not in catalog)
    if missing_roles:
        print(
            f"BAŞARISIZ: örneklenen {len(missing_roles)} rol kanonik katalogda yok: "
            f"{missing_roles}.\n"
            "  rollouts.jsonl ile roles.json aynı koşudan gelmiyor olabilir — "
            "Aşama 0 ve Aşama 1'i aynı kanonik katalogla tekrar çalıştırın.",
            file=sys.stderr,
        )
        return 2

    # 00_generate_role_data.py ve 01_smoke_gateway.py ile aynı desen: eksik
    # `APP_KEY_JAILBREAK` çıplak bir traceback yerine anlaşılır bir Türkçe tanı
    # ve sıfırdan farklı bir çıkış koduna (2) çevrilir. Brief'in Adım 5 kod
    # bloğunda bu sarmalayıcı yoktu, ama Adım 6 "anahtar yoksa temiz tanı ve
    # çıkış kodu 2" bekliyor — bu iki ifadeyi tutarlı kılmak için eklendi.
    try:
        client = build_default_client()
    except RuntimeError as exc:
        print(f"BAŞARISIZ: gateway istemcisi kurulamadı.\n  {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        return run_dry_run(client, by_role, records, catalog)

    labels: dict[int, str] = {}
    try:
        for role, rows in by_role.items():
            items = [(records[i]["question"], records[i]["answer"]) for i in rows]
            scores = score_role_expression(
                client,
                role=role,
                description=catalog[role],
                items=items,
                stage=STAGE,
            )
            for row, score in zip(rows, scores):
                labels[row] = collapse(score)
            print(f"\r{len(labels)}/{len(chosen)} etiketlendi", end="")
    except (BudgetExceeded, CircuitOpen) as exc:
        print(f"\nDURDURULDU: {exc}", file=sys.stderr)
        return 2
    except (GatewayError, JudgeParseError) as exc:
        print(f"\nBAŞARISIZ: {exc}", file=sys.stderr)
        return 2

    print()
    LABELS_PATH.write_text(
        json.dumps({"seed": SEED, "labels": {str(k): v for k, v in labels.items()}}, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Yazıldı: {LABELS_PATH} ({len(labels)} etiket), gönderilen istek: {client.sends_made}")

    # Tüm rol yanıtlarını TEK GEÇİŞTE embed et. Hakem etiketli ~2.000 satır
    # rol yanıtlarının (~16.000) bir alt kümesi olduğu için burada AYRICA
    # `embed_answers([records[i]["answer"] for i in rows])` çağırmak aynı
    # ~2.000 cümleyi ikinci kez embed ederdi (16.000 satır için 18.000
    # embedding) VE `embed_answers` içeride `SentenceTransformer('BAAI/bge-m3')`
    # kurduğu için birkaç GB'lık modeli diskten ikinci kez yüklerdi. Bunun
    # yerine TEK bir dizi hesaplanır; hakem etiketli satırların embedding'leri
    # bu dizinin içinden `role_rows`'taki KONUMLARINA (global `row` değil)
    # göre indekslenir.
    all_role_answers = [records[i]["answer"] for i in role_rows]
    all_embeddings = embed_answers(all_role_answers)

    rows = sorted(labels)
    role_row_position = {row: position for position, row in enumerate(role_rows)}
    label_positions = [role_row_position[row] for row in rows]
    embeddings = all_embeddings[label_positions]

    probe = RoleExpressionProbe(seed=SEED)
    try:
        probe.fit(embeddings, [labels[i] for i in rows])
    except ValueError as exc:
        print(
            "BAŞARISIZ: probe eğitilemedi.\n"
            f"  {exc}\n"
            "  Olası neden: nadir bir kategoriden (ör. 'somewhat') 2'den az örnek "
            "var — train_test_split sınıf başına en az 2 örnek ister. "
            "--sample-size'ı artırıp tekrar deneyin.",
            file=sys.stderr,
        )
        return 2
    print(f"Probe held-out uyumu: {probe.holdout_agreement:.1%} (eşik %85)")

    if not probe.is_trustworthy:
        print(
            f"PROBE GÜVENİLİR DEĞİL: held-out uyum %{probe.holdout_agreement * 100:.1f}, "
            "eşik %85.\n"
            "  Spec'in geri çekilme kuralı (Bölüm 5, Aşama 2) rol düzeyinde tut/at "
            "filtresine dönmeyi söyler: rol başına 15 rollout hakeme sorulur (~180 çağrı).\n"
            "  BU OTOMATİK GERİ ÇEKİLME UYGULANMADI — bu script onu koşmaz, elle bir "
            "karar gerekir. İki seçenek var:\n"
            "    1) Daha büyük bir --sample-size ile tekrar koş: probe'un uyumu çoğunlukla "
            "etiket sayısıyla sınırlıdır. Bunun bütçe maliyeti aşağıdaki KALAN'a "
            "sığmalıdır.\n"
            "    2) 'somewhat' vektörleriyle devam et (spec Bölüm 9'un '<40 rol' çıkış "
            "yolu) ve probe'un güvenilmez olduğunu sonuçlarda AÇIKÇA raporla.\n"
            f"  Bu koşuda harcanan: {client.sends_made} gönderim "
            f"(aşama tavanı {config.STAGE_BUDGETS[STAGE]}); "
            f"aşamada kalan: {client.remaining_budget(STAGE)[0]}, "
            f"global kalan: {client.remaining_budget(STAGE)[1]}.\n"
            f"  Hakem etiketleri {LABELS_PATH} içinde duruyor — tekrar koşuda cache'ten "
            "gelirler, aynı örnekler ikinci kez ücretlendirilmez.",
            file=sys.stderr,
        )
        return 1

    predicted = probe.predict(all_embeddings)

    expression = {str(row): labels.get(row, pred) for row, pred in zip(role_rows, predicted)}
    OUT_PATH.write_text(
        json.dumps(
            {
                # `05_capture_activations.py`'nin `activations_index.json`'a
                # yazdığı kimliğin AYNISI (ikisi de `rollouts.jsonl`
                # kayıtlarından türetilir). `07_extract_axis.py` ikisinin
                # eşit olmasını ŞART koşar: onsuz, Aşama 1'in FARKLI bir rol
                # kümesiyle ama aynı satır sayısı ve sırasıyla yeniden
                # koşturulması, 07'nin sayı+kapsama kontrollerinin ikisini de
                # geçiyordu ve fully/somewhat ayrımı sessizce kayıyordu.
                "run_id": rollouts_run_id(records),
                "holdout_agreement": probe.holdout_agreement,
                "n_judge_labels": len(labels),
                "n_probe_labels": len(role_rows) - len(labels),
                "expression": expression,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    counts = {c: list(expression.values()).count(c) for c in ("fully", "somewhat", "no")}
    print(f"Yazıldı: {OUT_PATH}")
    print(f"Dağılım: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
