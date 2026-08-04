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
