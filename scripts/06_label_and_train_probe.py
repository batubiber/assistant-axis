#!/usr/bin/env python3
"""Aşama 2 — hakem etiketleri topla, probe eğit, 16.000 rollout'u etiketle.

Kullanım:
    uv run python scripts/06_label_and_train_probe.py --dry-run
    uv run --extra ml python scripts/06_label_and_train_probe.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

import numpy as np

from aax import config
from aax.gateway import BudgetExceeded, CircuitOpen, GatewayError, build_default_client
from aax.judge import JudgeParseError, score_role_expression
from aax.probe import RoleExpressionProbe, embed_answers, stratified_sample
from aax.prompts import load_role_catalog
from aax.rollouts import read_rollouts

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sample-size", type=int, default=LABEL_SAMPLE_SIZE)
    args = parser.parse_args()

    records = read_rollouts(config.DATA_DIR / "rollouts.jsonl")
    role_rows = [i for i, r in enumerate(records) if r["kind"] == "role"]
    role_records = [records[i] for i in role_rows]

    chosen_local = stratified_sample(role_records, n=args.sample_size, seed=SEED)
    chosen = [role_rows[i] for i in chosen_local]

    # load_role_catalog üzerinden: kısmi/pilot bir katalogla etiketleme yapmak,
    # yanlış rol kümesi üzerinde probe eğitmek demek olurdu.
    catalog = {r["role"]: r["description"] for r in load_role_catalog(config.DATA_DIR / "roles.json")}

    by_role: dict[str, list[int]] = defaultdict(list)
    for row in chosen:
        by_role[records[row]["role"]].append(row)

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
        planned = sum(
            (len(rows) + 9) // 10 for rows in by_role.values()
        )
        cap = config.STAGE_BUDGETS[STAGE]
        print(f"Planlanan çağrı (üst sınır, cache yok sayılarak): {planned}")
        print(f"Aşama bütçesi: {cap}")
        if planned > cap:
            print("HATA: plan aşama bütçesini aşıyor — --sample-size küçült.", file=sys.stderr)
            return 1
        return 0

    labels: dict[int, str] = {}
    try:
        for role, rows in by_role.items():
            items = [(records[i]["question"], records[i]["answer"]) for i in rows]
            scores = score_role_expression(
                client,
                role=role,
                description=catalog.get(role, f"the role of a {role}"),
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

    rows = sorted(labels)
    embeddings = embed_answers([records[i]["answer"] for i in rows])
    probe = RoleExpressionProbe(seed=SEED)
    probe.fit(embeddings, [labels[i] for i in rows])
    print(f"Probe held-out uyumu: {probe.holdout_agreement:.1%} (eşik %85)")

    if not probe.is_trustworthy:
        print(
            "PROBE GÜVENİLİR DEĞİL — spec'in geri çekilme kuralı devreye giriyor.\n"
            "  Rol düzeyinde tut/at filtresine dön ve bunu sonuçlarda raporla.",
            file=sys.stderr,
        )
        return 1

    all_role_answers = [records[i]["answer"] for i in role_rows]
    all_embeddings = embed_answers(all_role_answers)
    predicted = probe.predict(all_embeddings)

    expression = {str(row): labels.get(row, pred) for row, pred in zip(role_rows, predicted)}
    OUT_PATH.write_text(
        json.dumps(
            {
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
