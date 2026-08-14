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

# `AAX_TARGET_MODEL` ile ortamdan geçersiz kılınabilir — ikinci bir hedef
# modelle (ör. Qwen/Qwen3-0.6B) koşmak için kaynak değişikliği gerekmez.
# Okuma import ANINDA olur (süreç başlarken); her script kendi süreci içinde
# `uv run python scripts/xx.py` ile koştuğu için bu, gerçek koşularda yeterli.
TARGET_MODEL = os.environ.get("AAX_TARGET_MODEL", "Qwen/Qwen3-1.7B")


def model_slug(model_id: str | None = None) -> str:
    """Model id'sini dosya yolu için güvenli, kısa bir isme çevir.

    "Qwen/Qwen3-1.7B" -> "qwen3-1.7b". `model_id` verilmezse çağrı ANINDAKİ
    `config.TARGET_MODEL` kullanılır — import anındaki değil. Fonksiyon
    gövdesi bare `TARGET_MODEL` adına başvurur, bu da Python'da modül
    globals'ından çağrı anında okunur; bir test `config.TARGET_MODEL`'i
    monkeypatch'lerse bu fonksiyonun (ve ona dayanan `model_data_dir` /
    `model_results_dir`'in) döndürdüğü değer de bunu izler.
    """
    if model_id is None:
        model_id = TARGET_MODEL
    name = model_id.strip()
    if not name:
        raise ValueError("model_id boş olamaz")
    # "Qwen/Qwen3-1.7B" -> "Qwen3-1.7B" -> "qwen3-1.7b". Yalnızca son path
    # bileşeni alınır (org/ad ayrımı) ve küçük harfe çevrilir.
    name = name.rsplit("/", 1)[-1]
    return name.lower()


def model_data_dir(model_id: str | None = None) -> Path:
    """Modele özel veri kökü: `data/models/<slug>/`.

    Model BAĞIMSIZ artifact'lar (roles.json, questions.json, gateway
    bütçesi/cache'i) burada DEĞİL, doğrudan `DATA_DIR` altında kalır — bkz.
    spec Bölüm 4.2.
    """
    return DATA_DIR / "models" / model_slug(model_id)


def model_results_dir(model_id: str | None = None) -> Path:
    """Modele özel sonuç kökü: `results/models/<slug>/`."""
    return RESULTS_DIR / "models" / model_slug(model_id)


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
# | stage4_steering    |       350 |  10 |   360 |
# | stage4_controls    |       225 |  15 |   240 |
# | stage5_drift       |       120 |  25 |   145 |
# | stage6_capping     |       150 |  30 |   180 |
# | stage7_turkish     |        60 |  15 |    75 |
# | TOPLAM             |     1.282 | 188 | 1.470 |
#
# `stage4_steering` satırı 2026-08-10 düzeltmesiyle (bkz.
# `.superpowers/sdd/p3-task-5-fix1-brief.md`, madde F7) güncellendi: eski
# 175/35/210 tek katmanlık bir sweep'in sayılarıydı. Gerçek plan iki katman
# (L14, L19) × 7 güç = 14 grup, grup başına 250 öğe (50 rol × 5 introspektif
# soru), `batch_size=10` ile grup başına 25 çağrı → 14 × 25 = 350 mantıksal
# çağrı, +10 böl-ve-kurtar payı = 360 bütçe (bkz. `STAGE_BUDGETS` altındaki
# E4 yorumu — o zaten 360'tı; burada güncellenen yalnızca bu YORUM
# TABLOSU ve aşağıdaki `STAGE_LOGICAL_CALLS`, ikisi de E4'ten SONRA bayat
# kalmıştı).
#
# Toplam GLOBAL_BUDGET'ın (1500) altında kalmak ZORUNDA — tavan kullanıcının
# onayladığı sayıdır ve yükseltilmez. Bir aşama sığmıyorsa batch küçültülür.
STAGE_BUDGETS: dict[str, int] = {
    "smoke": 10,
    "stage0_roles": 145,
    "stage05_judge_gate": 15,
    "stage2_probe_labels": 300,
    # 2026-08-10 düzeltmesi (bkz. `.superpowers/sdd/p3-task-5-supplement.md`,
    # madde E4): eski 210 TEK katmanlık bir sweep içindi. Gerçek plan iki
    # katman × 7 güç = 14 grup, grup başına 250 öğe (50 rol × 5 soru),
    # batch_size=10 ile grup başına 25 çağrı -> 14 × 25 = 350 çağrı, +10
    # böl-ve-kurtar payı = 360. GLOBAL_BUDGET (1500) DEĞİŞMEDİ — bu artış
    # yalnızca stage4_steering'in KENDİ alt sayacını büyütür.
    "stage4_steering": 360,
    "stage4_controls": 240,  # 2026-08-14: Kontrol deneyi (stage4_steering yönü
                             # belirtilmedi testi), 225 çağrılık, 15 böl-ve-kurtar payı.
    "stage5_drift": 145,  # 2026-08-14: Bütçe stage4_controls'a kaydırıldı. Aşama 5
                          # koşulmadan ÖNCE yeniden bütçelenmelidir (bkz. rapor).
    "stage6_capping": 180,
    "stage7_turkish": 75,
}

# Çoklu model desteği (2026-08-10 bütçe düzeltmesi): bu tavanların HANGİ
# sayaca uygulandığı aşamaya göre değişir.
#
# * Bu kümedeki aşamalar MODEL-BAĞIMLIDIR — çağrı hacmi hedef modele (rol
#   ifade eden yanıtların TAMAMI hedef modelden gelir: hakem kapısı pilot
#   rollout'ları, probe etiketleme 16k rollout'u, steering/drift/capping/
#   Türkçe transfer hepsi hedef modelin ÜRETTİĞİ metni değerlendirir).
#   `gateway.py` bu aşamalarda sayaç anahtarını `f"{stage}:{model_slug}"`
#   yapar — tavan HER MODEL İÇİN AYRI uygulanır. İkinci bir hedef modelin
#   koşusu, birinci modelin zaten harcadığı payı görmez.
# * `smoke` ve `stage0_roles` bu kümede DEĞİLDİR — bare anahtarla kalırlar.
#   Rol kataloğu (`roles.json`) ve smoke testi gateway'den (`hakem-llm`)
#   üretilir, hedef modelden değil; bir kez üretilip HER model tarafından
#   paylaşılır (bkz. spec Bölüm 4.2). Aynı tavanı ikinci model için tekrar
#   açmak, zaten var olan ortak bir artefaktı gereksiz yere yeniden
#   üretmeye izin verirdi.
#
# GLOBAL_BUDGET'ın anlamı DEĞİŞMEDİ: sayaç dosyasındaki TÜM anahtarların
# (bare veya model-scoped) toplamı, hâlâ 1500'ü aşamaz. Model-bağımlı bir
# aşamanın kendi tavanı bu global toplamı GENİŞLETMEZ — bkz.
# `tests/test_gateway.py::test_global_budget_still_binds_across_model_scoped_stage_keys`.
MODEL_DEPENDENT_STAGES: frozenset[str] = frozenset(
    {
        "stage05_judge_gate",
        "stage2_probe_labels",
        "stage4_steering",
        "stage4_controls",
        "stage5_drift",
        "stage6_capping",
        "stage7_turkish",
    }
)

# Aşama tablosunun dayandığı mantıksal çağrı sayıları (spec Bölüm 6).
# Yalnızca belgelendirme ve test içindir; hiçbir koruma buna bakmaz.
#
# `stage4_steering` 2026-08-10 düzeltmesiyle (bkz.
# `.superpowers/sdd/p3-task-5-fix1-brief.md`, madde F7) 175'ten 350'ye
# güncellendi — yukarıdaki yorum tablosuyla AYNI gerekçe: eski sayı tek
# katmanlık bir sweep'in kalıntısıydı, `STAGE_BUDGETS["stage4_steering"]`
# (360) E4'te zaten doğru sayıya yükseltilmişti ama bu sözlük SENKRONSUZ
# kalmıştı.
STAGE_LOGICAL_CALLS: dict[str, int] = {
    "smoke": 2,
    "stage0_roles": 120,
    "stage05_judge_gate": 5,
    "stage2_probe_labels": 250,
    "stage4_steering": 350,
    "stage4_controls": 225,
    "stage5_drift": 120,
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
