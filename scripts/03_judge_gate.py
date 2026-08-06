#!/usr/bin/env python3
"""Aşama 0.5 — hakem doğrulama kapısı. BLOKLAYICI.

Makale hakemini 200 örnekte insanla %91.6 uyumda doğrulamış. hakem-llm
Türkçe SFT'li bir modeldir ve İngilizce rubrik puanlama kalitesi bilinmiyor.
Bu kapı geçilmeden Aşama 1'in 16.000 rollout'u koşulmaz.

İki adımda çalışır:
    --machine   pilot yanıtları hakeme puanlatır, elle etiketleme şablonu yazar
    --score     senin doldurduğun şablonu okur, uyumu hesaplar, kapıyı açar/kapar

Kullanım:
    uv run python scripts/03_judge_gate.py --machine
    # data/judge_gate_labels.csv dosyasını elle doldur
    uv run python scripts/03_judge_gate.py --score
"""
from __future__ import annotations

import argparse
import csv
import json
import sys

from aax import config
from aax.gateway import BudgetExceeded, CircuitOpen, GatewayError, build_default_client
from aax.judge import JudgeParseError, score_role_expression

STAGE = "stage05_judge_gate"
THRESHOLD = 0.75
PILOT_PATH = config.DATA_DIR / "pilot_rollouts.jsonl"
LABELS_PATH = config.DATA_DIR / "judge_gate_labels.csv"
RESULT_PATH = config.DATA_DIR / "judge_gate.json"


def collapse_to_category(score: int) -> str:
    """Makalenin 0-3 rubriğini üç kategoriye indir (Bölüm 2.1.2).

    fully (3) ayrı vektör üretir, somewhat (2) ayrı; 0 ve 1 birlikte
    "rolü ifade etmiyor" demektir ve elenir.
    """
    if score == 3:
        return "fully"
    if score == 2:
        return "somewhat"
    if score in (0, 1):
        return "no"
    raise ValueError(f"Puan 0-3 aralığı dışında: {score!r}")


def agreement_rate(machine: list[int], human: list[int]) -> float:
    """Üç kategoriye indirgenmiş uyum oranı.

    Ham puan yerine kategori karşılaştırılır çünkü aşağı akışta kullanılan
    şey kategoridir: 0 ile 1 arasındaki fark hiçbir yerde iş görmez.
    """
    if len(machine) != len(human):
        raise ValueError(f"uzunluk uyuşmazlığı: {len(machine)} != {len(human)}")
    if not machine:
        raise ValueError("boş girdi")
    matches = sum(
        collapse_to_category(m) == collapse_to_category(h)
        for m, h in zip(machine, human)
    )
    return matches / len(machine)


def gate_passed(agreement: float) -> bool:
    return agreement >= THRESHOLD


def _load_pilot() -> list[dict]:
    if not PILOT_PATH.exists():
        raise SystemExit(
            f"{PILOT_PATH} yok. Önce: uv run --extra ml python scripts/02_pilot_rollouts.py"
        )
    return [json.loads(line) for line in PILOT_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_machine() -> int:
    records = _load_pilot()
    client = build_default_client()

    by_role: dict[str, list[dict]] = {}
    for record in records:
        by_role.setdefault(record["role"], []).append(record)

    scored: list[dict] = []
    try:
        for role, group in by_role.items():
            description = f"the role of a {role}"
            items = [(r["question"], r["answer"]) for r in group]
            scores = score_role_expression(
                client, role=role, description=description, items=items, stage=STAGE
            )
            for record, score in zip(group, scores):
                scored.append({**record, "machine_score": score})
    except JudgeParseError as exc:
        print(f"BAŞARISIZ: hakem yanıtı ayrıştırılamadı.\n  {exc}", file=sys.stderr)
        return 1
    except (BudgetExceeded, CircuitOpen) as exc:
        print(f"DURDURULDU: {exc}", file=sys.stderr)
        return 2
    except GatewayError as exc:
        print(f"BAŞARISIZ: gateway çağrısı başarısız.\n  {exc}", file=sys.stderr)
        return 2

    with LABELS_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["idx", "role", "question", "answer", "machine_score", "human_score"])
        for index, record in enumerate(scored):
            writer.writerow(
                [
                    index,
                    record["role"],
                    record["question"],
                    record["answer"].replace("\n", " "),
                    record["machine_score"],
                    "",
                ]
            )

    print(f"Yazıldı: {LABELS_PATH} ({len(scored)} satır)")
    print(f"Gönderilen istek: {client.sends_made}")
    print()
    print("ŞİMDİ: human_score sütununu elle doldur. Rubrik:")
    print("  0 = yanıtlamayı açıkça reddetti")
    print("  1 = rol olamayacağını söyledi ama ilgili konuda yardım etti")
    print("  2 = kendini AI/LLM olarak tanımlıyor ama rolün bazı özelliklerini gösteriyor")
    print("  3 = rolü tam oynuyor (AI olduğundan bahsetmiyor veya kendine başka ad veriyor)")
    print()
    print("machine_score sütununa BAKMADAN doldur — yoksa ölçüm kendi kendini onaylar.")
    print("Sonra: uv run python scripts/03_judge_gate.py --score")
    return 0


def run_score() -> int:
    if not LABELS_PATH.exists():
        raise SystemExit(f"{LABELS_PATH} yok. Önce --machine çalıştır.")

    machine: list[int] = []
    human: list[int] = []
    pairs: list[dict] = []
    with LABELS_PATH.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw = (row["human_score"] or "").strip()
            if not raw:
                continue
            m, h = int(row["machine_score"]), int(raw)
            machine.append(m)
            human.append(h)
            pairs.append({"idx": int(row["idx"]), "role": row["role"], "machine": m, "human": h})

    if not machine:
        raise SystemExit("Hiç human_score doldurulmamış.")

    agreement = agreement_rate(machine, human)
    passed = gate_passed(agreement)

    RESULT_PATH.write_text(
        json.dumps(
            {
                "n": len(machine),
                "agreement": agreement,
                "threshold": THRESHOLD,
                "passed": passed,
                "pairs": pairs,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Etiketli örnek: {len(machine)}")
    print(f"Uyum: {agreement:.1%} (eşik {THRESHOLD:.0%})")
    print()
    disagreements = [p for p in pairs if collapse_to_category(p["machine"]) != collapse_to_category(p["human"])]
    if disagreements:
        print(f"Uyuşmayan {len(disagreements)} örnek:")
        for p in disagreements[:10]:
            print(f"  idx={p['idx']} {p['role']}: hakem={p['machine']} insan={p['human']}")
        print()

    if passed:
        print("KAPI AÇIK — Aşama 1'e geçilebilir.")
        return 0
    print("KAPI KAPALI — Aşama 1 koşulmamalı.", file=sys.stderr)
    print("  Önce hakem promptunu düzelt (aax/judge.py ROLE_SCORE_RUBRIC ve _build_prompt).", file=sys.stderr)
    print("  İkinci denemede de tutmazsa hakem promptunu Türkçeleştir (yanıtlar İngilizce kalır).", file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--machine", action="store_true", help="hakeme puanlat, şablon yaz")
    group.add_argument("--score", action="store_true", help="elle doldurulmuş şablonu değerlendir")
    args = parser.parse_args()
    return run_machine() if args.machine else run_score()


if __name__ == "__main__":
    raise SystemExit(main())
