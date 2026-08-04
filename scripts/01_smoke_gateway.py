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
import sys
from pathlib import Path

from aax import config
from aax.gateway import (
    BudgetCorrupted,
    BudgetExceeded,
    CircuitOpen,
    GatewayError,
    build_default_client,
)
from aax.judge import JudgeParseError, extract_json

STAGE = "smoke"

# Çıkış kodları — bir kabuk pipeline'ının ayırt edebilmesi için:
#   0  her şey TAMAM
#   1  bağlantı kuruldu ama bir kontrol BAŞARISIZ (ör. JSON üretilemiyor)
#   2  koşu hiç yapılamadı (anahtar yok, bütçe doldu, devre açık, taşıma hatası)
EXIT_OK = 0
EXIT_KONTROL_BASARISIZ = 1
EXIT_KOSULAMADI = 2

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


def diagnose_budget_delta(
    spent: int, first_call_sends: int, second_call_sends: int
) -> tuple[str, str]:
    """Bütçe artışını DOĞRU teşhise indirger. `(karar, mesaj)` döner.

    Karar `"ok"`, `"warn"` veya `"fail"`.

    Neden `spent` tek başına yetmiyor: ilk çağrı geçici bir 5xx yüzünden bir
    kez yeniden denendiyse `spent` 2 olur — ama cache PEKÂLÂ çalışıyordur.
    Cache hiç çalışmasaydı da `spent` 2 olurdu. İki senaryo aynı sayıyı verir.
    Ayıran ölçüm `client.sends_made`: ikinci çağrı hiç gönderim yapmadıysa
    (`second_call_sends == 0`) cache çalışıyor demektir, `spent` ne olursa
    olsun. Eski kod bu ayrımı yapmadan "cache çalışmıyor" diyordu — projenin
    production'a ilk temasında yanlış teşhis.
    """
    if second_call_sends > 0:
        return "fail", (
            f"bütçe {spent} arttı ve İKİNCİ çağrı {second_call_sends} gerçek istek "
            "attı — cache çalışmıyor."
        )
    if spent == 0:
        return "fail", (
            "bütçe hiç artmadı — ilk çağrı da önceki bir koşunun cache'inden geldi. "
            "Bu smoke testi gerçek bir istek doğrulayamadı; "
            "data/gateway_cache/ temizlenip tekrar denenmeli."
        )
    if spent == 1:
        return "ok", "bütçe tam olarak 1 arttı"
    return "warn", (
        f"bütçe {spent} arttı: cache ÇALIŞIYOR (ikinci çağrı hiç istek atmadı) "
        f"ama ilk çağrı {first_call_sends - 1} kez yeniden denendi — sunucuda "
        "geçici hata olmuş olabilir."
    )


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


def run_probe(client) -> int:
    """İki çağrılık asıl doğrulama. Gateway istisnalarını `main()`'e bırakır."""
    before = read_budget(config.BUDGET_PATH)

    print("1) İlk çağrı gönderiliyor...")
    raw = client.chat([{"role": "user", "content": PROBE}], stage=STAGE, temperature=0.0)
    # İki çağrının gönderimlerini AYRI ölç: retry ile cache arızasını ancak
    # bu ayırır (bkz. `diagnose_budget_delta`).
    first_call_sends = client.sends_made
    print(f"   Ham yanıt:\n   {raw[:400]}\n")

    print("2) Aynı çağrı tekrar (cache'ten dönmeli)...")
    raw_again = client.chat(
        [{"role": "user", "content": PROBE}], stage=STAGE, temperature=0.0
    )
    second_call_sends = client.sends_made - first_call_sends

    after = read_budget(config.BUDGET_PATH)
    ok = True

    if check_cache_hit(raw, raw_again):
        print("   TAMAM: cache aynı yanıtı döndürdü")
    else:
        print("   BAŞARISIZ: cache aynı yanıtı döndürmedi")
        ok = False

    verdict, message = diagnose_budget_delta(
        after - before, first_call_sends, second_call_sends
    )
    if verdict == "ok":
        print(f"   TAMAM: {message}")
    elif verdict == "warn":
        print(f"   UYARI: {message}")
    else:
        print(f"   BAŞARISIZ: {message}")
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
    return EXIT_OK if ok else EXIT_KONTROL_BASARISIZ


def main() -> int:
    """Tanı sarmalayıcısı.

    Bu script projenin production `hakem-llm` sunucusuna İLK temasıdır ve onu
    çoğunlukla anahtarı yeni export etmiş bir operatör koşar. Ham bir traceback
    (eksik `APP_KEY_JAILBREAK`, dolmuş bütçe, açık devre kesici, bozuk bütçe
    dosyası) tam da o anda en faydasız çıktıdır — hepsi anlaşılır bir Türkçe
    tanı ve sıfırdan farklı bir çıkış koduna çevrilir.
    """
    try:
        client = build_default_client()
    except RuntimeError as exc:
        print(f"BAŞARISIZ: gateway istemcisi kurulamadı.\n  {exc}", file=sys.stderr)
        return EXIT_KOSULAMADI

    try:
        return run_probe(client)
    except BudgetCorrupted as exc:
        print(
            f"BAŞARISIZ: bütçe dosyası okunamıyor — hiç istek atılmadı.\n  {exc}\n"
            f"  Dosya: {config.BUDGET_PATH}\n"
            "  Sayaç sıfırlanmış sayılmaz: dosyayı elle onar ya da bilinçli olarak sil.",
            file=sys.stderr,
        )
    except BudgetExceeded as exc:
        print(
            f"BAŞARISIZ: çağrı bütçesi dolu — hiç istek atılmadı.\n  {exc}\n"
            f"  Sayaç: {config.BUDGET_PATH}. Tavan yükseltilmez; "
            "önceki koşuların harcamasını gözden geçir.",
            file=sys.stderr,
        )
    except CircuitOpen as exc:
        print(
            f"BAŞARISIZ: devre kesici açık — koşu durduruldu.\n  {exc}\n"
            "  Ortak production sunucusunu zorlamıyoruz. Sunucunun durumunu "
            "kontrol et ve süreci yeniden başlat.",
            file=sys.stderr,
        )
    except GatewayError as exc:
        print(
            f"BAŞARISIZ: gateway çağrısı başarısız oldu.\n  {exc}\n"
            "  401/403 ise APP_KEY_JAILBREAK yanlış; 5xx ise sunucu şu an sorunlu.\n"
            f"  Ayrıntılı log: {config.CALL_LOG_PATH}",
            file=sys.stderr,
        )
    return EXIT_KOSULAMADI


if __name__ == "__main__":
    raise SystemExit(main())
