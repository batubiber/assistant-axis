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
STAGE_BUDGETS: dict[str, int] = {
    "smoke": 10,
    "stage0_roles": 130,
    "stage05_judge_gate": 10,
    "stage2_probe_labels": 250,
    "stage4_steering": 175,
    "stage5_drift": 320,
    "stage6_capping": 150,
    "stage7_turkish": 60,
}

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
