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

from aax import config
from aax.gateway import BudgetExceeded, CircuitOpen, GatewayError, build_default_client
from aax.judge import JudgeParseError
from aax.roles import ROLE_NAMES, build_generation_prompt, parse_generation_response

STAGE = "stage0_roles"
SHARED_QUESTION_COUNT = 40
SEED = 20260804


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="istek atmadan çağrı say")
    parser.add_argument("--limit", type=int, default=None, help="ilk N rol (pilot için)")
    args = parser.parse_args()

    roles = ROLE_NAMES[: args.limit] if args.limit else ROLE_NAMES
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

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    failures: list[tuple[str, str]] = []

    for index, role in enumerate(roles, start=1):
        prompt = build_generation_prompt(role)
        try:
            raw = client.chat(
                [{"role": "user", "content": prompt}], stage=STAGE, max_tokens=4096
            )
            records.append(parse_generation_response(role, raw))
        except JudgeParseError as exc:
            failures.append((role, str(exc)))
        except (BudgetExceeded, CircuitOpen) as exc:
            print(f"\nDURDURULDU: {exc}", file=sys.stderr)
            break
        except GatewayError as exc:
            failures.append((role, str(exc)))
        print(f"\r{index}/{len(roles)} — {role:<16} hata: {len(failures)}", end="")

    print()
    roles_path = config.DATA_DIR / "roles.json"
    roles_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    pool = [question for record in records for question in record["questions"]]
    rng = random.Random(SEED)
    shared = rng.sample(pool, min(SHARED_QUESTION_COUNT, len(pool)))
    (config.DATA_DIR / "questions.json").write_text(
        json.dumps({"shared_questions": shared}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Yazıldı: {roles_path} ({len(records)} rol)")
    print(f"Ortak soru havuzu: {len(pool)} → {len(shared)} seçildi")
    print(f"Gönderilen istek: {client.sends_made}")
    if failures:
        print(f"\nBaşarısız {len(failures)} rol:")
        for role, reason in failures[:10]:
            print(f"  {role}: {reason[:100]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
