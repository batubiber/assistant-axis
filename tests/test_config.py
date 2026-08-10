import importlib
import os

import pytest

from aax import config


def test_stage_budgets_sum_below_global_cap():
    assert sum(config.STAGE_BUDGETS.values()) <= config.GLOBAL_BUDGET


def test_global_cap_is_the_approved_number():
    """1500 kullanıcının onayladığı tavan — kod içinde kaymamalı."""
    assert config.GLOBAL_BUDGET == 1500


def test_every_stage_has_retry_headroom():
    """Hiçbir aşama mantıksal çağrı sayısına eşit bütçeyle koşmamalı.

    Bütçe HTTP gönderimi sayar, mantıksal çağrı değil. `stage5_drift` 320
    çağrılık bir aşama için 320 gönderim bütçesiyle koşuyordu: tek bir geçici
    5xx aşamayı sonuna varmadan kesiyordu.
    """
    assert set(config.STAGE_LOGICAL_CALLS) == set(config.STAGE_BUDGETS)
    for stage, logical in config.STAGE_LOGICAL_CALLS.items():
        budget = config.STAGE_BUDGETS[stage]
        pay = budget - logical
        assert pay >= 8, f"'{stage}' retry payı yok: {budget} bütçe, {logical} çağrı"
        assert pay >= min(10, logical) or pay >= 0.2 * logical, (
            f"'{stage}' payı kurala uymuyor: {pay}"
        )


def test_stage_budget_table_matches_spec_bolum_6():
    """config.py ile spec Bölüm 6 tablosu aynı sayıları söylemeli.

    Sayılar sürüklendiğinde (stage0 130 vs 120, stage05 10 vs 5, spec'te hiç
    olmayan `smoke: 10`) hangi belgenin doğru olduğu belirsizleşiyordu.
    """
    assert config.STAGE_LOGICAL_CALLS == {
        "smoke": 2,
        "stage0_roles": 120,
        "stage05_judge_gate": 5,
        "stage2_probe_labels": 250,
        "stage4_steering": 175,
        "stage5_drift": 320,
        "stage6_capping": 150,
        "stage7_turkish": 60,
    }
    assert config.STAGE_BUDGETS == {
        "smoke": 10,
        "stage0_roles": 145,
        "stage05_judge_gate": 15,
        "stage2_probe_labels": 300,
        "stage4_steering": 210,
        "stage5_drift": 385,
        "stage6_capping": 180,
        "stage7_turkish": 75,
    }
    assert sum(config.STAGE_LOGICAL_CALLS.values()) == 1082
    assert sum(config.STAGE_BUDGETS.values()) == 1320


def test_api_key_raises_when_env_missing(monkeypatch):
    monkeypatch.delenv("APP_KEY_JAILBREAK", raising=False)
    with pytest.raises(RuntimeError, match="APP_KEY_JAILBREAK"):
        config.api_key()


def test_api_key_returns_env_value(monkeypatch):
    monkeypatch.setenv("APP_KEY_JAILBREAK", "secret-value")
    assert config.api_key() == "secret-value"


def test_gateway_url_targets_jailbreak_app():
    assert config.GATEWAY_BASE_URL.endswith("/Jailbreak")
    assert config.GATEWAY_BASE_URL.startswith("https://")


# --- çoklu model desteği: slug türetme, yol izolasyonu, env geçersiz kılma --


def test_model_slug_derives_short_lowercase_name_from_model_id():
    assert config.model_slug("Qwen/Qwen3-1.7B") == "qwen3-1.7b"
    assert config.model_slug("Qwen/Qwen3-0.6B") == "qwen3-0.6b"


def test_model_slug_uses_only_the_final_path_component():
    """Org/ad ayrımındaki '/' dosya yolunda kullanılamaz — yalnızca son
    bileşen alınır, org adı (`Qwen`) sonuca karışmaz."""
    assert config.model_slug("some-org/Some-Namespace/Weird-Model") == "weird-model"


def test_model_slug_rejects_blank_model_id():
    with pytest.raises(ValueError):
        config.model_slug("   ")


def test_model_slug_defaults_to_the_currently_active_target_model(monkeypatch):
    """Argümansız çağrı, ÇAĞRI ANINDAKİ `config.TARGET_MODEL`'i kullanır —
    import anındaki değeri DEĞİL. Bu, `config.TARGET_MODEL`'i monkeypatch'leyen
    testlerin (ör. script-düzeyi path testleri) `model_slug()`'un da bunu
    izlemesine dayanabilmesi için gerekli."""
    monkeypatch.setattr(config, "TARGET_MODEL", "Qwen/Qwen3-0.6B")
    assert config.model_slug() == "qwen3-0.6b"
    monkeypatch.setattr(config, "TARGET_MODEL", "Qwen/Qwen3-1.7B")
    assert config.model_slug() == "qwen3-1.7b"


def test_model_dependent_paths_differ_between_two_models(monkeypatch):
    """`data/models/<slug>/` ve `results/models/<slug>/` her model için AYRI
    dizin verir — bu, ikinci bir hedef modelin ilkinin sonuçlarını sessizce
    ezmemesinin temel garantisi."""
    dir_a = config.model_data_dir("Qwen/Qwen3-1.7B")
    dir_b = config.model_data_dir("Qwen/Qwen3-0.6B")
    assert dir_a != dir_b
    assert dir_a == config.DATA_DIR / "models" / "qwen3-1.7b"
    assert dir_b == config.DATA_DIR / "models" / "qwen3-0.6b"

    results_a = config.model_results_dir("Qwen/Qwen3-1.7B")
    results_b = config.model_results_dir("Qwen/Qwen3-0.6B")
    assert results_a != results_b
    assert results_a == config.RESULTS_DIR / "models" / "qwen3-1.7b"
    assert results_b == config.RESULTS_DIR / "models" / "qwen3-0.6b"

    # Argümansız kullanım (script'lerin gerçekte yaptığı) da aynı izolasyonu
    # sağlamalı: aktif model değişince yol da değişir.
    monkeypatch.setattr(config, "TARGET_MODEL", "Qwen/Qwen3-1.7B")
    active_a = config.model_data_dir()
    monkeypatch.setattr(config, "TARGET_MODEL", "Qwen/Qwen3-0.6B")
    active_b = config.model_data_dir()
    assert active_a != active_b
    assert active_a == dir_a
    assert active_b == dir_b


def test_model_independent_paths_are_identical_regardless_of_target_model(monkeypatch):
    """Rol kataloğu, ortak sorular, gateway bütçesi/cache'i TEK bir proje
    genelinde paylaşılır — hangi model aktif olursa olsun aynı yolda kalmalı.
    Bunlar `config.TARGET_MODEL`'e hiç bakmayan sabit sabitlerdir; bu test
    ikinci bir modele geçmenin bunları YANLIŞLIKLA model-özel bir dizine
    taşımadığını kanıtlar.
    """
    roles_a = config.DATA_DIR / "roles.json"
    questions_a = config.DATA_DIR / "questions.json"
    budget_a = config.BUDGET_PATH
    cache_a = config.CACHE_DIR
    calls_a = config.CALL_LOG_PATH

    monkeypatch.setattr(config, "TARGET_MODEL", "Qwen/Qwen3-1.7B")
    assert config.DATA_DIR / "roles.json" == roles_a
    assert config.DATA_DIR / "questions.json" == questions_a
    assert config.BUDGET_PATH == budget_a
    assert config.CACHE_DIR == cache_a
    assert config.CALL_LOG_PATH == calls_a

    monkeypatch.setattr(config, "TARGET_MODEL", "Qwen/Qwen3-0.6B")
    assert config.DATA_DIR / "roles.json" == roles_a
    assert config.DATA_DIR / "questions.json" == questions_a
    assert config.BUDGET_PATH == budget_a
    assert config.CACHE_DIR == cache_a
    assert config.CALL_LOG_PATH == calls_a

    # `.lock` dosyası da (gateway.py) BUDGET_PATH'ten türer — o da izole olmaz.
    assert budget_a.with_name(budget_a.name + ".lock") == (
        config.BUDGET_PATH.with_name(config.BUDGET_PATH.name + ".lock")
    )


def test_target_model_env_override_changes_active_model_and_therefore_paths(monkeypatch):
    """`AAX_TARGET_MODEL` ortam değişkeni, kaynak değişikliği olmadan ikinci
    bir hedef modele geçebilmeli.

    `TARGET_MODEL` modül import ANINDA ortamdan okunur, bu yüzden burada
    `importlib.reload` şart. `aax.config` `sys.modules`'te TEK bir nesnedir
    (bu dosyadaki `config` adı da dahil TÜM `from aax import config`
    kullanıcıları aynı nesneyi paylaşır) — reload testler arası SIZINTI
    riski taşır. `finally` bloğu ortam değişkenini geri alıp modülü TEKRAR
    reload ederek testin sonunda orijinal (env'siz) hâle döner; bu olmadan
    bu testten sonra koşan HER test yanlış bir `TARGET_MODEL` görürdü.
    """
    original_target_model = config.TARGET_MODEL
    original_env = os.environ.get("AAX_TARGET_MODEL")

    monkeypatch.setenv("AAX_TARGET_MODEL", "Qwen/Qwen3-0.6B")
    try:
        importlib.reload(config)
        assert config.TARGET_MODEL == "Qwen/Qwen3-0.6B"
        assert config.model_slug() == "qwen3-0.6b"
        assert config.model_data_dir() == config.DATA_DIR / "models" / "qwen3-0.6b"
        assert config.model_results_dir() == config.RESULTS_DIR / "models" / "qwen3-0.6b"
    finally:
        monkeypatch.delenv("AAX_TARGET_MODEL", raising=False)
        if original_env is not None:
            monkeypatch.setenv("AAX_TARGET_MODEL", original_env)
        importlib.reload(config)

    assert config.TARGET_MODEL == original_target_model
