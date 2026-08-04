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
import json
import random
import sys
from pathlib import Path

from aax import config
from aax.gateway import BudgetExceeded, CircuitOpen, GatewayError, build_default_client
from aax.judge import JudgeParseError
from aax.roles import ROLE_NAMES, build_generation_prompt, parse_generation_response

STAGE = "stage0_roles"
SHARED_QUESTION_COUNT = 40
# Sabit tohum: ortak soru kümesi her koşuda aynı çıkmalı, yoksa farklı
# koşulardan gelen rol vektörleri birbiriyle kıyaslanabilir olmaktan çıkar.
SEED = 20260804


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

    `complete` alanı `produced == requested` olup olmadığını taşır; downstream
    bir script'in kısmi bir katalogu tam sanmasını engellemek için bu zarf
    gereklidir. `not_attempted`, döngünün hiç ulaşamadığı rolleri katalog
    sırasıyla listeler — bir operatörün "hangi roller kaldı?" sorusuna
    `ROLE_NAMES` ile set-diff almaya gerek kalmadan doğrudan bu dosyadan cevap
    bulabilmesi için.
    """
    produced = len(records)
    return {
        "complete": produced == requested,
        "requested": requested,
        "attempted": attempted,
        "produced": produced,
        "not_attempted": not_attempted,
        "failed": [{"role": role, "reason": reason} for role, reason in failures],
        "roles": records,
    }


def sample_shared_questions(records: list[dict]) -> list[str]:
    """Üretilen tüm sorulardan `SEED` ile deterministik bir ortak alt küme seç."""
    pool = [question for record in records for question in record["questions"]]
    rng = random.Random(SEED)
    return rng.sample(pool, min(SHARED_QUESTION_COUNT, len(pool)))


def build_questions_payload(records: list[dict], requested: int, attempted: int) -> dict:
    """`questions.json`/`questions.partial.json` içeriği.

    Sayaç anlamları `build_roles_payload` ile birebir aynıdır (bkz. orada).
    """
    produced = len(records)
    return {
        "complete": produced == requested,
        "requested": requested,
        "attempted": attempted,
        "produced": produced,
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
    roles_path.write_text(
        json.dumps(roles_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    questions_path.write_text(
        json.dumps(questions_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if allow_partial and not complete:
        # `--allow-partial` eski bir tam artifact'ı sessizce ezmesin: bu
        # bilinçli terfiyi operatöre stderr'de açıkça söyle.
        print(
            f"UYARI: kısmi koşu ({roles_payload['produced']}/{roles_payload['requested']} "
            f"rol) kanonik dosya adlarına terfi ettirildi — {roles_path.name} / "
            f"{questions_path.name}. Önceki tam artifact varsa üzerine yazıldı.",
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
    roles: tuple[str, ...], client, *, stage: str = STAGE, max_tokens: int = 4096
) -> tuple[list[dict], list[tuple[str, str]], int]:
    """`roles`'u sırayla gateway'e gönder, yanıtları ayrıştır.

    Döndürür: `(records, failures, attempted)`. `attempted`, döngünün fiilen
    ulaştığı rol sayısıdır — `len(roles)` (istenen batch büyüklüğü) ile
    KARIŞTIRILMAMALI: bütçe/devre kesici koşuyu erken durdurursa küçük kalır.

    Bütçe/devre kesici koşuyu durdurduğunda, o anki rol de "denendi VE
    başarısız oldu" sayılır — atlanmaz, nedeni açıkça tetikleyici olduğunu
    söyleyen bir mesajla `failures`'a eklenir. Böylece bir ayrıştırma/gateway
    hatasından ayırt edilebilir kalır ve `roles.partial.json` "koşuyu hangi
    rol durdurdu?" sorusuna kendi başına cevap verebilir.
    """
    records: list[dict] = []
    failures: list[tuple[str, str]] = []
    attempted = 0

    for index, role in enumerate(roles, start=1):
        attempted = index
        prompt = build_generation_prompt(role)
        try:
            raw = client.chat(
                [{"role": "user", "content": prompt}], stage=stage, max_tokens=max_tokens
            )
            records.append(parse_generation_response(role, raw))
        except JudgeParseError as exc:
            failures.append((role, str(exc)))
        except (BudgetExceeded, CircuitOpen) as exc:
            print(f"\nDURDURULDU: {exc}", file=sys.stderr)
            failures.append(
                (role, f"DURDURULDU — koşuyu durduran bütçe/devre kesici tetikleyicisi: {exc}")
            )
            break
        except GatewayError as exc:
            failures.append((role, str(exc)))
        print(f"\r{index}/{len(roles)} — {role:<16} hata: {len(failures)}", end="")

    return records, failures, attempted


def main() -> int:
    args = build_arg_parser().parse_args()

    roles = select_roles(args.limit)
    client = build_default_client()

    if args.dry_run:
        # max_tokens cache anahtarının parçası — chat() ile birebir aynı olmalı,
        # yoksa dry-run cache'teki kayıtları göremez.
        planned = sum(
            1
            for role in roles
            if client.would_call(
                [{"role": "user", "content": build_generation_prompt(role)}],
                max_tokens=4096,
            )
        )
        cap = config.STAGE_BUDGETS[STAGE]
        print(f"Planlanan çağrı: {planned} (cache'te: {len(roles) - planned})")
        print(f"Aşama bütçesi:   {cap}")
        if planned > cap:
            print("HATA: plan aşama bütçesini aşıyor — batch'i küçült.", file=sys.stderr)
            return 1
        return 0

    records, failures, attempted = run_generation_loop(roles, client)

    print()
    exit_code, roles_path, questions_path, roles_payload, questions_payload = write_artifacts(
        config.DATA_DIR, roles, records, failures, attempted, args.allow_partial
    )

    pool_size = sum(len(record["questions"]) for record in records)
    shared = questions_payload["shared_questions"]
    print(
        f"Yazıldı: {roles_path} "
        f"({roles_payload['produced']}/{roles_payload['requested']} rol, "
        f"complete={roles_payload['complete']})"
    )
    print(f"Ortak soru havuzu: {pool_size} → {len(shared)} seçildi → {questions_path}")
    print(f"Gönderilen istek: {client.sends_made}")
    if failures:
        print(f"\nBaşarısız {len(failures)} rol:")
        for role, reason in failures[:10]:
            print(f"  {role}: {reason[:100]}")

    if exit_code != 0:
        print(
            f"\nHATA: koşu eksik kaldı ({roles_payload['produced']}/"
            f"{roles_payload['requested']}) ve --allow-partial verilmedi — "
            f"kanonik dosyalara DOKUNULMADI, bunun yerine {roles_path.name} / "
            f"{questions_path.name} yazıldı.",
            file=sys.stderr,
        )

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
