#!/usr/bin/env python3
"""Aşama 0.5 — hakem doğrulama kapısı. BLOKLAYICI.

Makale hakemini 200 örnekte insanla %91.6 uyumda doğrulamış. hakem-llm
Türkçe SFT'li bir modeldir ve İngilizce rubrik puanlama kalitesi bilinmiyor.
Bu kapı geçilmeden Aşama 1'in 16.000 rollout'u koşulmaz.

İki adımda çalışır:
    --machine   pilot yanıtları hakeme puanlatır, KÖR bir elle etiketleme
                şablonu (`data/judge_gate_labels.csv`) yazar — bu dosyada
                makine puanı YOKTUR. Makine puanları ayrı bir dosyaya
                (`data/judge_gate_machine.json`) gider.
    --score     senin doldurduğun kör şablonu ve ayrı makine dosyasını idx
                üzerinden birleştirir, uyumu hesaplar, kapıyı açar/kapar

Neden iki ayrı dosya: makine puanı ve insan puanı aynı satırda yan yana
dursa operatör 40 satırı doldururken makinenin cevabını göz ucuyla görür.
Bu, ölçümü kendi kendini onaylayan bir şeye çevirir — makinenin ne kadar iyi
olduğunu değil, operatörün makineye ne kadar uyduğunu ölçer. Körlük yalnızca
bir talimat cümlesiyle ("bakmadan doldur") sağlanamaz; dosya yapısı bunu
FİZİKSEL olarak imkânsız kılmalı.

Kullanım:
    uv run python scripts/03_judge_gate.py --machine
    # data/judge_gate_labels.csv dosyasındaki human_score sütununu elle doldur
    # (machine_score sütunu YOK — puanlar data/judge_gate_machine.json'da,
    # --score'a kadar saklı tutulur)
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
from aax.prompts import load_role_catalog

STAGE = "stage05_judge_gate"
THRESHOLD = 0.75
# Spec Bölüm 5, Aşama 0.5: "pilot bir rol setinden 40 yanıt üret ... aynı 40'ı
# elle etiketle, uyumu ölç". Bu sayı bir öneri değil, kapının ölçüm gücüdür:
# 3 satırda tesadüfen tutan bir uyum "%100" diye raporlanır ve BLOKLAYICI kapı
# açılır. Boş satırlar sessizce atlandığı için tek koruma "en az bir tane"ydi.
MIN_LABELLED = 40
PILOT_PATH = config.DATA_DIR / "pilot_rollouts.jsonl"
ROLES_PATH = config.DATA_DIR / "roles.json"
LABELS_PATH = config.DATA_DIR / "judge_gate_labels.csv"
MACHINE_PATH = config.DATA_DIR / "judge_gate_machine.json"
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


def _parse_human_score(raw: str) -> int:
    """Elle girilmiş insan puanını doğrula.

    Operatör ~40 değeri elle yazıyor; bu tüm hattaki tek elle-yazılan girdi
    ve tek savunmasız nokta. Bir yazım hatası ("3.", "n/a", fazladan boşluk)
    veya 0-3 dışı bir değer burada AÇIKÇA reddedilmeli — `int(raw)`'ın
    ham `ValueError`'ı hiçbir satır numarası söylemez, `collapse_to_category`
    ise 0-3 dışı değerleri kendi hata mesajıyla siler ki bu mesaj hangi satırın
    bozuk olduğunu bilmiyor.
    """
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"sayı değil: {raw!r}") from None
    if not 0 <= value <= 3:
        raise ValueError(f"0-3 aralığı dışında: {value}")
    return value


def run_machine(client) -> int:
    records = _load_pilot()

    # Kapı, ÜRETİMDE koşacak hakemi doğrulamalı — ona yakın bir şeyi değil.
    # Burada açıklama `f"the role of a {role}"` diye ÜRETİLİYORDU, oysa
    # `06_label_and_train_probe.py` kataloğun LLM tarafından yazılmış tam
    # açıklamasını geçiyor ve `judge._build_prompt` bu dizeyi promptun en
    # tanımlayıcı cümlesine ("You are evaluating whether ... the role: {role}.
    # {description}") gömüyor. İki prompt en belirleyici alanında farklıysa
    # kapının onayladığı uyum, 2.000 rollout'u etiketleyecek hakemin uyumu
    # DEĞİLDİR.
    try:
        catalog = {
            r["role"]: r["description"]
            for r in load_role_catalog(ROLES_PATH)
        }
    except FileNotFoundError:
        print(
            f"BAŞARISIZ: {ROLES_PATH} yok.\n"
            "  Hakem kapısı, Aşama 2'nin kullanacağı rol açıklamalarının AYNISIYLA "
            "puanlamak zorunda; katalog olmadan bu mümkün değil.\n"
            "  Önce Aşama 0'ı (scripts/00_generate_role_data.py) çalıştırın.",
            file=sys.stderr,
        )
        return 2
    except ValueError as exc:
        print(
            "BAŞARISIZ: rol kataloğu kanonik değil.\n"
            f"  {exc}\n"
            "  Aşama 0'ı (scripts/00_generate_role_data.py) --allow-partial OLMADAN "
            "tamamlayıp tekrar deneyin.",
            file=sys.stderr,
        )
        return 2

    by_role: dict[str, list[dict]] = {}
    for record in records:
        by_role.setdefault(record["role"], []).append(record)

    # Fail-closed, `06_label_and_train_probe.py` ile aynı gerekçe: eksik bir
    # rol için jenerik bir açıklama uydurmak, tam da düzeltilen hatayı geri
    # getirirdi.
    missing_roles = sorted(role for role in by_role if role not in catalog)
    if missing_roles:
        print(
            f"BAŞARISIZ: pilot'taki {len(missing_roles)} rol kanonik katalogda yok: "
            f"{missing_roles}.\n"
            "  pilot_rollouts.jsonl ile roles.json aynı koşudan gelmiyor olabilir — "
            "scripts/02_pilot_rollouts.py'yi güncel katalogla tekrar çalıştırın.",
            file=sys.stderr,
        )
        return 2

    scored: list[dict] = []
    try:
        for role, group in by_role.items():
            description = catalog[role]
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

    # KÖR çalışma sayfası: yalnızca insanın dolduracağı human_score sütunu
    # var. machine_score burada YOKTUR — aşağıdaki ayrı JSON dosyasına gider.
    # Bkz. modül docstring'i: bu yapısal ayrım, "bakmadan doldur" talimatının
    # tek başına sağlayamadığı körlüğü fiziksel olarak zorunlu kılar.
    with LABELS_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["idx", "role", "question", "answer", "human_score"])
        for index, record in enumerate(scored):
            writer.writerow(
                [
                    index,
                    record["role"],
                    record["question"],
                    record["answer"].replace("\n", " "),
                    "",
                ]
            )

    machine_scores = {str(index): record["machine_score"] for index, record in enumerate(scored)}
    MACHINE_PATH.write_text(
        json.dumps(machine_scores, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Yazıldı: {LABELS_PATH} ({len(scored)} satır, YALNIZCA human_score sütunu)")
    print(f"Makine puanları ayrı dosyada: {MACHINE_PATH}")
    print(f"Gönderilen istek: {client.sends_made}")
    print()
    print("ŞİMDİ: human_score sütununu elle doldur. Rubrik:")
    print("  0 = yanıtlamayı açıkça reddetti")
    print("  1 = rol olamayacağını söyledi ama ilgili konuda yardım etti")
    print("  2 = kendini AI/LLM olarak tanımlıyor ama rolün bazı özelliklerini gösteriyor")
    print("  3 = rolü tam oynuyor (AI olduğundan bahsetmiyor veya kendine başka ad veriyor)")
    print()
    print(f"{LABELS_PATH.name} BİLEREK kördür: makine puanı bu dosyada hiç yok, {RESULT_PATH.name}")
    print("hesaplanana kadar ayrı tutuluyor. Eski düzeni elle geri kurmaya (bir sütun")
    print("ekleyip makine puanını yapıştırmaya) çalışma — kapının bütün amacı bu.")
    print("Sonra: uv run python scripts/03_judge_gate.py --score")
    return 0


def run_score(min_labelled: int = MIN_LABELLED) -> int:
    if not LABELS_PATH.exists():
        raise SystemExit(f"{LABELS_PATH} yok. Önce --machine çalıştır.")
    if not MACHINE_PATH.exists():
        raise SystemExit(f"{MACHINE_PATH} yok. Önce --machine çalıştır.")

    with LABELS_PATH.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    try:
        raw_machine = json.loads(MACHINE_PATH.read_text(encoding="utf-8"))
        machine_by_idx: dict[int, int] = {int(k): int(v) for k, v in raw_machine.items()}
    except (ValueError, TypeError, AttributeError) as exc:
        raise SystemExit(f"{MACHINE_PATH} okunamadı/ayrıştırılamadı: {exc}") from exc

    # İki dosya idx üzerinden birleşiyor. Biri diğerinde olmayan bir idx
    # içeriyorsa bu SESSİZCE atlanacak bir şey değil — dosyalar birbirine
    # karışmış (yanlış koşudan kalma, elle satır silinmiş/eklenmiş) olabilir
    # ve bu durumda uyum hesabı yanlış bir alt kümeye dayanır.
    label_idxs = {int(row["idx"]) for row in rows}
    machine_idxs = set(machine_by_idx)
    if label_idxs != machine_idxs:
        only_labels = sorted(label_idxs - machine_idxs)
        only_machine = sorted(machine_idxs - label_idxs)
        parts = []
        if only_labels:
            parts.append(
                f"{len(only_labels)} idx yalnızca {LABELS_PATH.name} içinde var "
                f"(örnek: {only_labels[:5]})"
            )
        if only_machine:
            parts.append(
                f"{len(only_machine)} idx yalnızca {MACHINE_PATH.name} içinde var "
                f"(örnek: {only_machine[:5]})"
            )
        raise SystemExit(
            "KRİTİK: worksheet ve makine puanları arasında idx uyuşmazlığı — "
            + "; ".join(parts)
            + ". Dosyalar birbirine karışmış olabilir; --machine'i baştan çalıştırıp "
            "her iki dosyayı da yeniden üret."
        )

    bad_rows: list[tuple[int, str, str]] = []
    machine: list[int] = []
    human: list[int] = []
    pairs: list[dict] = []
    for row in rows:
        raw = (row["human_score"] or "").strip()
        if not raw:
            continue
        idx = int(row["idx"])
        try:
            h = _parse_human_score(raw)
        except ValueError as exc:
            bad_rows.append((idx, raw, str(exc)))
            continue
        m = machine_by_idx[idx]
        machine.append(m)
        human.append(h)
        pairs.append({"idx": idx, "role": row["role"], "machine": m, "human": h})

    if bad_rows:
        lines = "\n".join(
            f"  idx={idx}: {message} (girilen: {raw!r})" for idx, raw, message in bad_rows
        )
        raise SystemExit(
            f"{len(bad_rows)} satırda geçersiz human_score:\n{lines}\n"
            "Bu satırları düzelt ve tekrar dene."
        )

    if not machine:
        raise SystemExit("Hiç human_score doldurulmamış.")

    # Boş satırlar sessizce atlanıyor ve tek taban "en az bir tane"ydi: elle
    # doldurulmuş 3 satır tesadüfen tutarsa "Uyum: %100.0 — KAPI AÇIK" basılıp
    # 16.000 rollout'luk aşama serbest bırakılıyordu. Spec Bölüm 5 (Aşama 0.5)
    # 40 diyor; bu bir tavsiye değil, kapının istatistiksel gücü.
    if len(machine) < min_labelled:
        print(
            f"BAŞARISIZ: yalnızca {len(machine)} satır etiketlenmiş, en az "
            f"{min_labelled} gerekiyor (spec Bölüm 5, Aşama 0.5).\n"
            f"  {LABELS_PATH} içinde {len(rows)} satır var; human_score sütunu "
            f"{len(rows) - len(machine)} satırda boş.\n"
            "  Bu bir KAPI KARARI DEĞİLDİR: bu kadar az örnekle ölçülen uyum, hakemin "
            "kalitesini değil tesadüfü ölçer.\n"
            "  Tabanı bilinçli olarak düşürmek için: --min-labelled N.",
            file=sys.stderr,
        )
        return 2

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

    print(f"Etiketli örnek: {len(machine)} (asgari {min_labelled})")
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--machine", action="store_true", help="hakeme puanlat, şablon yaz")
    group.add_argument("--score", action="store_true", help="elle doldurulmuş şablonu değerlendir")
    parser.add_argument(
        "--min-labelled",
        type=int,
        default=MIN_LABELLED,
        help=(
            f"kapının değerlendirileceği asgari elle etiketlenmiş satır sayısı "
            f"(varsayılan {MIN_LABELLED}, spec Bölüm 5); altında çıkış 2"
        ),
    )
    args = parser.parse_args(argv)

    if args.score:
        return run_score(args.min_labelled)

    # `build_default_client()` `config.api_key()` üzerinden bare bir
    # `RuntimeError` fırlatabilir (APP_KEY_JAILBREAK export edilmemiş —
    # operatörün en olası ilk hatası, kapı iki ayrı kabuk çağrısı olduğu
    # için ikinci çağrıda unutmak kolay). `BudgetExceeded`/`CircuitOpen`/
    # `GatewayError` de `RuntimeError`'dan türer ama İSTEMCİ KURULUMUNDA
    # DEĞİL, `chat()` çağrılarında (run_machine içinde, aşağıda) oluşur —
    # o yüzden burada AYRI ve DAR bir `except RuntimeError` bloğu var, tıpkı
    # `scripts/01_smoke_gateway.py::main()`'deki gibi.
    try:
        client = build_default_client()
    except RuntimeError as exc:
        print(f"BAŞARISIZ: gateway istemcisi kurulamadı.\n  {exc}", file=sys.stderr)
        return 2

    return run_machine(client)


if __name__ == "__main__":
    raise SystemExit(main())
