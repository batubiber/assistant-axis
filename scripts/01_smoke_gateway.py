#!/usr/bin/env python3
"""Gerçek gateway'e karşı 2 çağrılık uçtan uca doğrulama.

Doğruladıkları:
  1. Bağlantı + kimlik doğrulama çalışıyor
  2. Cache ikinci çağrıyı emiyor (bütçe 1 artıyor, 2 değil)
  3. hakem-llm İngilizce yapılandırılmış JSON üretebiliyor

Kullanım:
    export APP_KEY_JAILBREAK=...      # dağıtım ortamının .env dosyasından
    uv run python scripts/01_smoke_gateway.py
"""
from __future__ import annotations

import json
from pathlib import Path

from aax import config
from aax.gateway import build_default_client
from aax.judge import JudgeParseError, extract_json

STAGE = "smoke"

PROBE = (
    "Classify each of the following statements as either \"factual\" or "
    "\"opinion\".\n\n"
    "[ITEM 0] Water boils at 100 degrees Celsius at sea level.\n"
    "[ITEM 1] Blue is the most beautiful colour.\n"
    "[ITEM 2] The Earth orbits the Sun.\n\n"
    "Respond with ONLY a JSON array of 3 strings, in order. No other text."
)


def read_budget(budget_path: Path) -> int:
    """Bütçe dosyasındaki tüm aşama sayaçlarının toplamı.

    Dosya yoksa 0 döner (hiç istek atılmamış demektir). Yol parametre olarak
    alınır ki testler gerçek `config.BUDGET_PATH`'e değil, `tmp_path` altında
    sahte bir dosyaya baksın — ağa da diskteki gerçek bütçeye de dokunmadan.
    """
    if not budget_path.exists():
        return 0
    return sum(json.loads(budget_path.read_text(encoding="utf-8")).values())


def check_cache_hit(raw: str, raw_again: str) -> bool:
    """İkinci çağrı cache'ten dönüp ilkiyle birebir aynı yanıtı mı verdi?"""
    return raw == raw_again


def check_budget_delta(before: int, after: int) -> tuple[bool, int]:
    """Bütçe tam olarak 1 mi arttı?

    İkinci çağrı cache'ten karşılanmalı, yani yalnızca ilk çağrı bütçe
    harcamalı. `spent` 0 ise ilk çağrı da bir önceki koşudan kalma cache'e
    denk gelmiştir (bütçe hiç artmadı); 2 ise cache hiç çalışmamıştır. İkisi
    de bu smoke testinin amacına aykırıdır, bu yüzden yalnızca `1` TAMAM
    sayılır.
    """
    spent = after - before
    return spent == 1, spent


def check_json_shape(raw: str) -> tuple[str, object]:
    """Ham model yanıtını JSON-şekil doğrulama kararına indirger.

    Üç sonuçtan biri döner:
      - `("ok", parsed)`   — `parsed` 3 elemanlı bir liste
      - `("warn", parsed)` — JSON ayrıştı ama şekil beklenenden farklı
      - `("fail", exc)`    — `extract_json` hiçbir aday metinden JSON
        çıkaramadı (`exc` bir `JudgeParseError`)
    """
    try:
        parsed = extract_json(raw)
    except JudgeParseError as exc:
        return "fail", exc
    if isinstance(parsed, list) and len(parsed) == 3:
        return "ok", parsed
    return "warn", parsed


def main() -> int:
    client = build_default_client()
    before = read_budget(config.BUDGET_PATH)

    print("1) İlk çağrı gönderiliyor...")
    raw = client.chat([{"role": "user", "content": PROBE}], stage=STAGE, temperature=0.0)
    print(f"   Ham yanıt:\n   {raw[:400]}\n")

    print("2) Aynı çağrı tekrar (cache'ten dönmeli)...")
    raw_again = client.chat(
        [{"role": "user", "content": PROBE}], stage=STAGE, temperature=0.0
    )

    after = read_budget(config.BUDGET_PATH)
    ok = True

    if check_cache_hit(raw, raw_again):
        print("   TAMAM: cache aynı yanıtı döndürdü")
    else:
        print("   BAŞARISIZ: cache aynı yanıtı döndürmedi")
        ok = False

    budget_ok, spent = check_budget_delta(before, after)
    if budget_ok:
        print("   TAMAM: bütçe tam olarak 1 arttı")
    else:
        print(f"   BAŞARISIZ: bütçe {spent} arttı, 1 beklenirdi (cache çalışmıyor)")
        ok = False

    print("3) JSON ayrıştırma...")
    verdict, payload = check_json_shape(raw)
    if verdict == "ok":
        print(f"   TAMAM: {payload}")
    elif verdict == "warn":
        print(f"   UYARI: JSON çıktı ama şekil beklenenden farklı: {payload!r}")
        print("   → Aşama 0.5 hakem kapısında prompt düzeltmesi gerekebilir.")
    else:
        print(f"   BAŞARISIZ: {payload}")
        print("   → hakem-llm İngilizce JSON üretemiyor. Hakem hattı gözden geçirilmeli.")
        ok = False

    print(f"\nToplam gönderilen istek: {client.sends_made}")
    print(f"Log: {config.CALL_LOG_PATH}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
