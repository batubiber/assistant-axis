import json

import pytest

from aax.gateway import (
    BudgetExceeded,
    CircuitOpen,
    GatewayClient,
    GatewayConfig,
    GatewayError,
)


class FakeClock:
    """Testlerin gerçek zamanda beklememesi için enjekte edilen saat."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def ok_body(text: str = "merhaba"):
    return {
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3},
    }


def make_client(tmp_path, transport, *, global_budget=10, stage_budget=10, rps=1.0):
    clock = FakeClock()
    cfg = GatewayConfig(
        base_url="https://example.invalid/Jailbreak",
        model="hakem-llm",
        api_key="test-key",
        requests_per_second=rps,
        max_concurrency=2,
        global_budget=global_budget,
        stage_budgets={"test": stage_budget},
    )
    client = GatewayClient(
        cfg,
        cache_dir=tmp_path / "cache",
        budget_path=tmp_path / "budget.json",
        log_path=tmp_path / "calls.jsonl",
        transport=transport,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    return client, clock


MSG = [{"role": "user", "content": "selam"}]


def test_returns_message_content(tmp_path):
    calls = []

    def transport(payload):
        calls.append(payload)
        return 200, ok_body("cevap")

    client, _ = make_client(tmp_path, transport)
    assert client.chat(MSG, stage="test") == "cevap"
    assert len(calls) == 1
    assert calls[0]["model"] == "hakem-llm"


def test_global_budget_exceeded_raises_without_sending(tmp_path):
    calls = []

    def transport(payload):
        calls.append(payload)
        return 200, ok_body()

    client, _ = make_client(tmp_path, transport, global_budget=2, stage_budget=99)
    client.chat([{"role": "user", "content": "a"}], stage="test")
    client.chat([{"role": "user", "content": "b"}], stage="test")
    with pytest.raises(BudgetExceeded):
        client.chat([{"role": "user", "content": "c"}], stage="test")
    assert len(calls) == 2, "bütçe dolduktan sonra hiç istek gitmemeli"


def test_stage_budget_is_enforced_separately(tmp_path):
    calls = []

    def transport(payload):
        calls.append(payload)
        return 200, ok_body()

    client, _ = make_client(tmp_path, transport, global_budget=99, stage_budget=1)
    client.chat([{"role": "user", "content": "a"}], stage="test")
    with pytest.raises(BudgetExceeded):
        client.chat([{"role": "user", "content": "b"}], stage="test")
    assert len(calls) == 1


def test_budget_persists_across_client_instances(tmp_path):
    def transport(payload):
        return 200, ok_body()

    client_a, _ = make_client(tmp_path, transport, global_budget=2, stage_budget=99)
    client_a.chat([{"role": "user", "content": "a"}], stage="test")

    client_b, _ = make_client(tmp_path, transport, global_budget=2, stage_budget=99)
    client_b.chat([{"role": "user", "content": "b"}], stage="test")
    with pytest.raises(BudgetExceeded):
        client_b.chat([{"role": "user", "content": "c"}], stage="test")


def test_cache_hit_avoids_second_request(tmp_path):
    calls = []

    def transport(payload):
        calls.append(payload)
        return 200, ok_body("cache-edilmis")

    client, _ = make_client(tmp_path, transport)
    first = client.chat(MSG, stage="test")
    second = client.chat(MSG, stage="test")
    assert first == second == "cache-edilmis"
    assert len(calls) == 1, "aynı payload iki kez gönderilmemeli"
    assert client.sends_made == 1


def test_cache_survives_new_client(tmp_path):
    calls = []

    def transport(payload):
        calls.append(payload)
        return 200, ok_body("kalici")

    client_a, _ = make_client(tmp_path, transport)
    client_a.chat(MSG, stage="test")

    client_b, _ = make_client(tmp_path, transport)
    assert client_b.chat(MSG, stage="test") == "kalici"
    assert len(calls) == 1


def test_would_call_reports_cache_state_without_sending(tmp_path):
    calls = []

    def transport(payload):
        calls.append(payload)
        return 200, ok_body()

    client, _ = make_client(tmp_path, transport)
    assert client.would_call(MSG) is True
    assert len(calls) == 0, "would_call istek atmamalı"

    client.chat(MSG, stage="test")
    assert client.would_call(MSG) is False


def test_circuit_opens_after_three_consecutive_failures(tmp_path):
    calls = []

    def transport(payload):
        calls.append(payload)
        return 500, {"error": "bozuk"}

    client, _ = make_client(tmp_path, transport, global_budget=99, stage_budget=99)

    for i in range(3):
        with pytest.raises(GatewayError):
            client.chat([{"role": "user", "content": f"m{i}"}], stage="test")

    sends_before = len(calls)
    with pytest.raises(CircuitOpen):
        client.chat([{"role": "user", "content": "sonraki"}], stage="test")
    assert len(calls) == sends_before, "devre açıkken hiç istek gitmemeli"


def test_success_resets_failure_counter(tmp_path):
    state = {"fail": True}

    def transport(payload):
        if state["fail"]:
            return 500, {"error": "bozuk"}
        return 200, ok_body()

    client, _ = make_client(tmp_path, transport, global_budget=99, stage_budget=99)
    with pytest.raises(GatewayError):
        client.chat([{"role": "user", "content": "a"}], stage="test")

    state["fail"] = False
    client.chat([{"role": "user", "content": "b"}], stage="test")

    state["fail"] = True
    with pytest.raises(GatewayError):
        client.chat([{"role": "user", "content": "c"}], stage="test")
    with pytest.raises(GatewayError):
        client.chat([{"role": "user", "content": "d"}], stage="test")
    # Sayaç sıfırlandığı için devre henüz açılmamalı: bu 3. ardışık hata
    with pytest.raises(GatewayError):
        client.chat([{"role": "user", "content": "e"}], stage="test")
    with pytest.raises(CircuitOpen):
        client.chat([{"role": "user", "content": "f"}], stage="test")


def test_retries_on_429_then_succeeds(tmp_path):
    attempts = {"n": 0}

    def transport(payload):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return 429, {"error": "cok hizli"}
        return 200, ok_body("nihayet")

    client, clock = make_client(tmp_path, transport, global_budget=99, stage_budget=99)
    assert client.chat(MSG, stage="test") == "nihayet"
    assert attempts["n"] == 2
    assert client.sends_made == 2, "retry de bütçeden sayılmalı"
    assert clock.slept, "retry öncesi backoff uygulanmalı"


def test_rate_limiter_spaces_requests(tmp_path):
    def transport(payload):
        return 200, ok_body()

    client, clock = make_client(tmp_path, transport, global_budget=99, stage_budget=99, rps=1.0)
    client.chat([{"role": "user", "content": "a"}], stage="test")
    t_after_first = clock.now
    client.chat([{"role": "user", "content": "b"}], stage="test")
    assert clock.now - t_after_first >= 1.0, "iki istek arasında en az 1 sn olmalı"


def test_call_log_written_as_jsonl(tmp_path):
    def transport(payload):
        return 200, ok_body()

    client, _ = make_client(tmp_path, transport)
    client.chat(MSG, stage="test")

    lines = (tmp_path / "calls.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["stage"] == "test"
    assert entry["status"] == 200
    assert entry["cached"] is False
    assert "api_key" not in json.dumps(entry), "anahtar log'a sızmamalı"


def test_api_key_never_appears_in_cache_files(tmp_path):
    def transport(payload):
        return 200, ok_body()

    client, _ = make_client(tmp_path, transport)
    client.chat(MSG, stage="test")

    for path in (tmp_path / "cache").rglob("*"):
        if path.is_file():
            assert "test-key" not in path.read_text()
