"""Proje geneli sabitler ve yollar."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
CACHE_DIR = DATA_DIR / "gateway_cache"
BUDGET_PATH = DATA_DIR / "gateway_budget.json"
CALL_LOG_PATH = DATA_DIR / "gateway_calls.jsonl"

# LLM Gateway'in /Jailbreak/ uygulaması: hakem promptlarına müdahale edilmiyor.
GATEWAY_BASE_URL = "https://gateway.invalid/app"
GATEWAY_MODEL = "hakem-llm"

TARGET_MODEL = "Qwen/Qwen3-1.7B"

# Spec Bölüm 6'daki bütçe dağılımı. Her aşama kendi anahtarını kullanır.
#
# BİRİM: bu sayaçlar **HTTP gönderimi** sayar, mantıksal çağrı değil. Bir
# mantıksal çağrı retry'larla 1, 2 veya 3 gönderim harcayabilir (MAX_RETRIES).
# Spec'in Bölüm 6 tablosu mantıksal çağrıları sayar; oradaki 1.082'lik toplam
# ile buradaki 1.320'lik toplamın farkı bilinçli **retry payıdır**.
#
# Neden pay şart: pay yokken `stage5_drift` 320 gönderim / ~320 mantıksal
# çağrıydı — tek bir geçici 5xx bile aşamayı sonuna varmadan kesiyordu.
# Kural: pay ≈ tabanın %20'si, küçük aşamalarda en az 10 gönderim, 5'in
# katına yuvarlanır.
#
# | Aşama              | Mantıksal | Pay | Bütçe |
# |--------------------|----------:|----:|------:|
# | smoke              |         2 |   8 |    10 |
# | stage0_roles       |       120 |  25 |   145 |
# | stage05_judge_gate |         5 |  10 |    15 |
# | stage2_probe_labels|       250 |  50 |   300 |
# | stage4_steering    |       175 |  35 |   210 |
# | stage5_drift       |       320 |  65 |   385 |
# | stage6_capping     |       150 |  30 |   180 |
# | stage7_turkish     |        60 |  15 |    75 |
# | TOPLAM             |     1.082 | 238 | 1.320 |
#
# Toplam GLOBAL_BUDGET'ın (1500) altında kalmak ZORUNDA — tavan kullanıcının
# onayladığı sayıdır ve yükseltilmez. Bir aşama sığmıyorsa batch küçültülür.
STAGE_BUDGETS: dict[str, int] = {
    "smoke": 10,
    "stage0_roles": 145,
    "stage05_judge_gate": 15,
    "stage2_probe_labels": 300,
    "stage4_steering": 210,
    "stage5_drift": 385,
    "stage6_capping": 180,
    "stage7_turkish": 75,
}

# Aşama tablosunun dayandığı mantıksal çağrı sayıları (spec Bölüm 6).
# Yalnızca belgelendirme ve test içindir; hiçbir koruma buna bakmaz.
STAGE_LOGICAL_CALLS: dict[str, int] = {
    "smoke": 2,
    "stage0_roles": 120,
    "stage05_judge_gate": 5,
    "stage2_probe_labels": 250,
    "stage4_steering": 175,
    "stage5_drift": 320,
    "stage6_capping": 150,
    "stage7_turkish": 60,
}

# Sert tavan. Kullanıcının onayladığı sayı — hiçbir gerekçeyle yükseltilmez.
GLOBAL_BUDGET = 1500

RATE_LIMIT_RPS = 1.0
MAX_CONCURRENCY = 2
MAX_RETRIES = 3
CIRCUIT_THRESHOLD = 3


def api_key() -> str:
    """Gateway anahtarını ortamdan oku.

    Anahtar dağıtım-ortamı'deki deploy .env dosyasındadır; yerel llm-gateway/.env
    kopyasında yoktur. Repoya asla yazılmaz.
    """
    key = os.environ.get("APP_KEY_JAILBREAK")
    if not key:
        raise RuntimeError(
            "APP_KEY_JAILBREAK ortam değişkeni tanımlı değil. "
            "Dağıtım ortamınızın .env dosyasından alıp kabuğunuzda export edin."
        )
    return key
