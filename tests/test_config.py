import pytest

from aax import config


def test_stage_budgets_sum_below_global_cap():
    assert sum(config.STAGE_BUDGETS.values()) <= config.GLOBAL_BUDGET


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
