#!/usr/bin/env python3
"""Aşama 0: rol başına açıklama + 3 sistem promptu + 40 soru üret.

Rol başına tek gateway çağrısı. Cache sayesinde yeniden koşmak bedava.
Önce mutlaka --dry-run ile çağrı sayısını doğrula.

Kullanım:
    uv run python scripts/00_generate_role_data.py --dry-run
    uv run python scripts/00_generate_role_data.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
from pathlib import Path

from aax import config
from aax.gateway import BudgetExceeded, CircuitOpen, GatewayError, build_default_client
from aax.judge import JudgeParseError
from aax.roles import ROLE_NAMES, build_generation_prompt, parse_generation_response

STAGE = "stage0_roles"
SHARED_QUESTION_COUNT = 40

# Cache anahtarının parçası olan çağrı parametreleri TEK yerde tanımlıdır.
# `run_generation_loop` (chat) ile `run_dry_run` (would_call) bu sabitleri
# paylaşmak zorunda: değerler ayrışırsa iki yol farklı cache anahtarı üretir
# ve --dry-run cache'te hazır duran kayıtları göremeyip her şeyi "planlanan"
# sayar. Eskiden ikisinde de elle yazılıydı ve bu kırılganlık bir yorumla
# işaretlenmişti; artık yapıyla bağlı ve testle sabitleniyor.
CHAT_TEMPERATURE = 0.0
CHAT_MAX_TOKENS = 4096

# Sabit tohum: ortak soru kümesi her koşuda aynı çıkmalı, yoksa farklı
# koşulardan gelen rol vektörleri birbiriyle kıyaslanabilir olmaktan çıkar.
#
# DETERMİNİZM KOŞULLUDUR. Tohum sabittir ama örnekleme havuzu `records`'tan
# kurulur, yani hangi rollerin BAŞARDIĞINA bağlıdır. Aynı tohum + farklı rol
# kümesi = farklı 40 soru. Bu yüzden her iki artifact da içerikten türetilen
# bir `run_id` ile damgalanır (bkz. `compute_run_id`): Aşama 1'in 14.400
# rollout'u `questions.json`'a bağlıdır, sessiz bir takas diskteki işi
# geçersiz kılardı.
SEED = 20260804

# Üst üste bu kadar rolde ayrıştırma hatası olursa koşu durur. `hakem-llm`'nin
# 40 soruluk JSON'u üretip üretemediği tam da Aşama 0.5 kapısının test ettiği
# belirsizlik. Gövdesi ayrışan ama kullanılamayan bir 200 devre kesiciyi
# SIFIRLAR (taşıma katmanı açısından başarılıdır), bu yüzden gateway burada
# hiçbir şey yapmaz: aşama bütçesinin tamamını yakıp sıfır kayıt üretmeyi
# engelleyen tek şey bu script düzeyindeki sayaçtır.
MAX_CONSECUTIVE_PARSE_FAILURES = 10


def compute_run_id(records: list[dict]) -> str:
    """Üretilen rol adlarından (katalog sırasıyla) türetilen koşu kimliği.

    Saatten değil İÇERİKTEN türetilir: bu repoda saate bağlı hiçbir kimlik
    yok ve yeniden üretilebilirlik asıl mesele. Aynı roller başarırsa aynı
    kimlik çıkar, yani aynı `shared_questions` da çıkar.

    Tüketici tarafı için sözleşme: `roles.json` ve `questions.json` aynı
    `run_id`'yi taşıyorsa aynı koşudandırlar. `questions.json`'ın `run_id`'si
    bir Aşama 1 rollout setinin beklediğinden farklıysa soru kümesi değişmiş
    demektir ve o rollout'lar artık o soru kümesine ait değildir.
    """
    blob = "\n".join(record["role"] for record in records)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def build_roles_payload(
    records: list[dict],
    failures: list[tuple[str, str]],
    requested: int,
    attempted: int,
    not_attempted: list[str],
) -> dict:
    """`roles.json`/`roles.partial.json` içeriği — bare list yerine zarf.

    Üç sayaç bilerek birbirinden farklıdır:

    * `requested`  — istenen batch büyüklüğü (`select_roles()` çıktısının
      uzunluğu); koşu hiç başlamasa bile sabittir.
    * `attempted`  — döngünün fiilen ulaştığı rol sayısı. Bütçe/devre kesici
      koşuyu erken durdurursa `requested`'tan küçük kalır.
    * `produced`   — başarıyla ayrıştırılan kayıt sayısı (`len(records)`).

    Değişmez: `requested == produced + len(failed) + len(not_attempted)` —
    denenen her rol ya bir kayıt ya bir hata üretir, denenmeyen her rol
    `not_attempted`'tadır.

    `complete` alanı `produced == requested` olup olmadığını taşır; downstream
    bir script'in kısmi bir katalogu tam sanmasını engellemek için bu zarf
    gereklidir. `not_attempted`, döngünün hiç ulaşamadığı rolleri katalog
    sırasıyla listeler — bir operatörün "hangi roller kaldı?" sorusuna
    `ROLE_NAMES` ile set-diff almaya gerek kalmadan doğrudan bu dosyadan cevap
    bulabilmesi için. `run_id`, `questions.json` ile eşleştirme içindir.
    """
    produced = len(records)
    return {
        "run_id": compute_run_id(records),
        "complete": produced == requested,
        "requested": requested,
        "attempted": attempted,
        "produced": produced,
        "not_attempted": not_attempted,
        "failed": [{"role": role, "reason": reason} for role, reason in failures],
        "roles": records,
    }


def sample_shared_questions(records: list[dict]) -> list[str]:
    """Üretilen tüm sorulardan `SEED` ile deterministik bir ortak alt küme seç.

    Determinizm KOŞULLUDUR: havuz `records`'tan kurulur, yani aynı rollerin
    başarmasına bağlıdır (bkz. `SEED` notu).
    """
    pool = [question for record in records for question in record["questions"]]
    rng = random.Random(SEED)
    return rng.sample(pool, min(SHARED_QUESTION_COUNT, len(pool)))


def build_questions_payload(records: list[dict], requested: int, attempted: int) -> dict:
    """`questions.json`/`questions.partial.json` içeriği.

    Sayaç anlamları `build_roles_payload` ile birebir aynıdır (bkz. orada).

    Örnekleme girdileri (`seed`, `role_count`, `pool_size`) ve `run_id` de
    yazılır. Neden: `--allow-partial` ile koşan bir koşu kısmi bir havuzdan
    kanonik `questions.json`'ı yazabilir; sonraki tam bir koşu onu FARKLI 40
    soruyla ezer. Spec Aşama 1 bu dosyaya bağlı 14.400 rollout üretiyor —
    sessiz takas diskteki işi geçersiz kılardı. Bu alanlar sayesinde tüketici
    soru kümesinin değiştiğini görebilir.
    """
    produced = len(records)
    pool = [question for record in records for question in record["questions"]]
    return {
        "run_id": compute_run_id(records),
        "complete": produced == requested,
        "requested": requested,
        "attempted": attempted,
        "produced": produced,
        "seed": SEED,
        "role_count": produced,
        "pool_size": len(pool),
        "shared_questions": sample_shared_questions(records),
    }


def resolve_artifact_paths(
    data_dir: Path, complete: bool, allow_partial: bool
) -> tuple[Path, Path]:
    """Kanonik dosya adlarını yalnızca koşu tamsa (ya da bilerek bypass
    edildiyse) döndür.

    Kapalı yönde (fail-closed) davranış: tamlığı kanıtlanamayan bir koşu
    `data/roles.json` / `data/questions.json`'a asla dokunmaz — bunun
    yerine `.partial.json` adlarına yazılır ve mevcut tam katalog olduğu
    gibi kalır. `--allow-partial` bu korumayı bilinçli olarak devre dışı
    bırakan görünür kaçış kapısıdır.
    """
    if complete or allow_partial:
        return data_dir / "roles.json", data_dir / "questions.json"
    return data_dir / "roles.partial.json", data_dir / "questions.partial.json"


def _stage_temp_file(path: Path, text: str) -> str:
    """`path`'in yanında tam yazılmış, fsync'lenmiş geçici dosya bırak."""
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}-", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return tmp_name


def _publish_atomically(payloads: list[tuple[Path, dict]]) -> None:
    """İki artifact'ı geçici dosya + `os.replace` ile yayımla.

    `gateway.py`'nin bütçe dosyası için kullandığı standardın aynısı: yazım
    ortasında bir çökme yarım bir `roles.json` bırakamaz. Ek olarak HER İKİ
    geçici dosya da diske tam yazılmadan hiçbiri yerine konmaz, yani "roles
    yazıldı ama questions yarım kaldı" durumu oluşmaz.

    Dürüst sınır: iki `os.replace` ardışıktır, aralarında (mikrosaniyelik) bir
    pencere vardır. POSIX'te iki dosyayı tek işlemde takas etmenin yolu yok.
    O pencerede çökülürse eski `questions.json` yeni `roles.json` ile eşleşir —
    ama ikisi farklı `run_id` taşıyacağı için tüketici bunu GÖREBİLİR
    (bkz. `compute_run_id`). Yarım/kırpık dosya hiçbir durumda oluşmaz.
    """
    staged: list[tuple[str, Path]] = []
    try:
        for path, payload in payloads:
            text = json.dumps(payload, ensure_ascii=False, indent=2)
            staged.append((_stage_temp_file(path, text), path))
        for tmp_name, path in staged:
            os.replace(tmp_name, path)
    except BaseException:
        for tmp_name, _ in staged:
            Path(tmp_name).unlink(missing_ok=True)
        raise


def write_artifacts(
    data_dir: Path,
    roles: tuple[str, ...],
    records: list[dict],
    failures: list[tuple[str, str]],
    attempted: int,
    allow_partial: bool,
) -> tuple[int, Path, Path, dict, dict]:
    """Kayıtları diske yaz, dosya adını tamlık durumuna göre seç.

    `roles` istenen (requested) tam batch'tir — `roles[attempted:]` döngünün
    hiç ulaşamadığı kuyruktur (`not_attempted`), çünkü `run_generation_loop`
    rolleri her zaman katalog sırasıyla ve aradan atlamadan işler.

    Döndürür: `(exit_code, roles_path, questions_path, roles_payload,
    questions_payload)`. `exit_code` koşu eksik kaldıysa ve
    `allow_partial` verilmediyse `1`'dir — bir kabuk pipeline'ının
    devam etmek yerine durması için.
    """
    requested = len(roles)
    not_attempted = list(roles[attempted:])

    roles_payload = build_roles_payload(records, failures, requested, attempted, not_attempted)
    questions_payload = build_questions_payload(records, requested, attempted)
    complete = roles_payload["complete"]

    roles_path, questions_path = resolve_artifact_paths(data_dir, complete, allow_partial)
    data_dir.mkdir(parents=True, exist_ok=True)
    _publish_atomically(
        [(roles_path, roles_payload), (questions_path, questions_payload)]
    )

    if allow_partial and not complete:
        # `--allow-partial` eski bir tam artifact'ı sessizce ezmesin: bu
        # bilinçli terfiyi operatöre stderr'de açıkça söyle.
        print(
            f"UYARI: kısmi koşu ({roles_payload['produced']}/{roles_payload['requested']} "
            f"rol) kanonik dosya adlarına terfi ettirildi — {roles_path.name} / "
            f"{questions_path.name}. Önceki tam artifact varsa üzerine yazıldı. "
            f"Yeni run_id: {roles_payload['run_id']} — bu koşunun ortak soru "
            f"kümesi öncekinden FARKLI olabilir.",
            file=sys.stderr,
        )

    exit_code = 0 if (complete or allow_partial) else 1
    return exit_code, roles_path, questions_path, roles_payload, questions_payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="istek atmadan çağrı say")
    parser.add_argument("--limit", type=int, default=None, help="ilk N rol (pilot için)")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "eksik kalan koşuyu kanonik roles.json/questions.json dosyalarına "
            "yaz (varsayılan: kapalı — fail-closed, .partial.json'a yazılır)"
        ),
    )
    parser.add_argument(
        "--max-parse-failures",
        type=int,
        default=MAX_CONSECUTIVE_PARSE_FAILURES,
        help=(
            "üst üste bu kadar ayrıştırma hatasında koşuyu durdur "
            f"(varsayılan: {MAX_CONSECUTIVE_PARSE_FAILURES})"
        ),
    )
    return parser


def select_roles(limit: int | None) -> tuple[str, ...]:
    """`--limit` uygulanmış rol listesi.

    `0` "sınırsız" değil "sıfır rol" demektir — bu yüzden falsy kontrolü
    (`if limit`) değil `is not None` kullanılır. `0` bir Python int'i olarak
    falsy'dir; `if args.limit:` ile yazılsaydı `--limit 0` sessizce tüm 120
    rolü çalıştırırdı.
    """
    return ROLE_NAMES[:limit] if limit is not None else ROLE_NAMES


def run_generation_loop(
    roles: tuple[str, ...],
    client,
    *,
    stage: str = STAGE,
    max_tokens: int = CHAT_MAX_TOKENS,
    max_consecutive_parse_failures: int = MAX_CONSECUTIVE_PARSE_FAILURES,
) -> tuple[list[dict], list[tuple[str, str]], int, str | None]:
    """`roles`'u sırayla gateway'e gönder, yanıtları ayrıştır.

    Döndürür: `(records, failures, attempted, stop_reason)`. `attempted`,
    döngünün fiilen ulaştığı rol sayısıdır — `len(roles)` (istenen batch
    büyüklüğü) ile KARIŞTIRILMAMALI: koşu erken durursa küçük kalır.
    `stop_reason` koşuyu durduran nedendir, durmadıysa `None`.

    **Hiçbir istisna bu döngüden dışarı sızmaz.** Bu bilinçli: sızan bir hata
    `main()`'in `write_artifacts`'e hiç ulaşamamasına, yani o ana kadar
    üretilmiş TÜM kayıtların çöpe gitmesine yol açıyordu. `BudgetCorrupted`,
    bilinmeyen aşama `ValueError`'ı ve `KeyboardInterrupt` — yani en olası
    başarısızlık kipleri — `.partial.json` makinesini tamamen atlıyordu.
    Artık ne çıkarsa çıksın `failures`'a kaydedilir, döngü kırılır ve o ana
    kadarki iş diske yazılır.

    Durdurma nedenleri üç türlüdür ve `failed` içinde ayırt edilebilir kalır:

    * bütçe / devre kesici tetikleyicisi,
    * üst üste `max_consecutive_parse_failures` ayrıştırma hatası,
    * beklenmeyen istisna (kesinti dahil).

    Değişmez: denenen her rol için TAM OLARAK bir sonuç kaydedilir — ya
    `records`'a bir kayıt ya `failures`'a bir satır. Durdurma anında ikinci
    bir `failures` satırı EKLENMEZ, mevcut satırın nedeni zenginleştirilir.
    """
    records: list[dict] = []
    failures: list[tuple[str, str]] = []
    attempted = 0
    stop_reason: str | None = None
    consecutive_parse_failures = 0

    for index, role in enumerate(roles, start=1):
        attempted = index
        try:
            prompt = build_generation_prompt(role)
            raw = client.chat(
                [{"role": "user", "content": prompt}],
                stage=stage,
                temperature=CHAT_TEMPERATURE,
                max_tokens=max_tokens,
            )
            records.append(parse_generation_response(role, raw))
            consecutive_parse_failures = 0
        except JudgeParseError as exc:
            consecutive_parse_failures += 1
            if consecutive_parse_failures >= max_consecutive_parse_failures:
                stop_reason = (
                    f"DURDURULDU — üst üste {consecutive_parse_failures} rolde "
                    f"ayrıştırma hatası (sınır: {max_consecutive_parse_failures}). "
                    "hakem-llm istenen JSON'u üretemiyor olabilir; aşama bütçesini "
                    f"sıfır kayıt için yakmamak adına durduruldu. Son hata: {exc}"
                )
                print(f"\n{stop_reason}", file=sys.stderr)
                failures.append((role, stop_reason))
                break
            failures.append((role, str(exc)))
        except (BudgetExceeded, CircuitOpen) as exc:
            stop_reason = (
                f"DURDURULDU — koşuyu durduran bütçe/devre kesici tetikleyicisi: {exc}"
            )
            print(f"\nDURDURULDU: {exc}", file=sys.stderr)
            failures.append((role, stop_reason))
            break
        except GatewayError as exc:
            failures.append((role, str(exc)))
        except BaseException as exc:  # noqa: BLE001 — bkz. docstring
            stop_reason = (
                f"DURDURULDU — beklenmeyen hata ({type(exc).__name__}): {exc}. "
                "O ana kadar üretilen kayıtlar kısmi artifact olarak yazıldı."
            )
            print(f"\n{stop_reason}", file=sys.stderr)
            failures.append((role, stop_reason))
            break
        print(f"\r{index}/{len(roles)} — {role:<16} hata: {len(failures)}", end="")

    return records, failures, attempted, stop_reason


def run_dry_run(client, roles: tuple[str, ...]) -> int:
    """İstek atmadan planı KALAN bütçeyle kıyasla.

    Aşama TAVANIYLA (`STAGE_BUDGETS[STAGE]`) kıyaslamak yanıltıcıydı: tavanın
    çoğunu önceki bir koşuda harcamış bir operatör temiz bir 0 görüp koşuyu
    başlatıyor, kalan bütçe biter bitmez ortasında kesiliyordu. Spec Bölüm 6
    `--dry-run`'ı zorunlu ön kontrol yapıyor; ön kontrolün baktığı sayı tavan
    değil diskteki sayaca göre KALAN olmalı. Global tavan da ayrıca kontrol
    edilir: aşama bütçesi bol olsa bile 1500 dolmuş olabilir.
    """
    planned = sum(
        1
        for role in roles
        if client.would_call(
            [{"role": "user", "content": build_generation_prompt(role)}],
            temperature=CHAT_TEMPERATURE,
            max_tokens=CHAT_MAX_TOKENS,
        )
    )
    stage_remaining, global_remaining = client.remaining_budget(STAGE)
    stage_cap = config.STAGE_BUDGETS[STAGE]

    print(f"Planlanan çağrı:      {planned} (cache'te: {len(roles) - planned})")
    print(f"Aşama bütçesi:        {stage_cap} (kalan: {stage_remaining})")
    print(f"Global tavan:         {config.GLOBAL_BUDGET} (kalan: {global_remaining})")

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
            "Koşu başlatılmadı — batch'i --limit ile küçült. Tavan yükseltilmez.",
            file=sys.stderr,
        )
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    roles = select_roles(args.limit)
    client = build_default_client()

    if args.dry_run:
        return run_dry_run(client, roles)

    records, failures, attempted, stop_reason = run_generation_loop(
        roles, client, max_consecutive_parse_failures=args.max_parse_failures
    )

    print()
    exit_code, roles_path, questions_path, roles_payload, questions_payload = write_artifacts(
        config.DATA_DIR, roles, records, failures, attempted, args.allow_partial
    )

    pool_size = questions_payload["pool_size"]
    shared = questions_payload["shared_questions"]
    print(
        f"Yazıldı: {roles_path} "
        f"({roles_payload['produced']}/{roles_payload['requested']} rol, "
        f"complete={roles_payload['complete']}, run_id={roles_payload['run_id']})"
    )
    print(f"Ortak soru havuzu: {pool_size} → {len(shared)} seçildi → {questions_path}")
    print(f"Gönderilen istek: {client.sends_made}")
    if failures:
        print(f"\nBaşarısız {len(failures)} rol:")
        for role, reason in failures[:10]:
            print(f"  {role}: {reason[:100]}")

    if stop_reason is not None:
        # Yarıda kesilmiş bir koşu asla "başarı" değildir — `--allow-partial`
        # dosya adlarını terfi ettirse bile çıkış kodu sıfır olmamalı.
        print(f"\n{stop_reason}", file=sys.stderr)
        exit_code = exit_code or 1

    if exit_code != 0:
        print(
            f"\nHATA: koşu eksik kaldı ({roles_payload['produced']}/"
            f"{roles_payload['requested']}) — yazılan dosyalar: {roles_path.name} / "
            f"{questions_path.name}.",
            file=sys.stderr,
        )

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
