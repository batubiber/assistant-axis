import json
import threading

import pytest

from aax.gateway import (
    BudgetCorrupted,
    BudgetExceeded,
    CircuitOpen,
    GatewayClient,
    GatewayConfig,
    GatewayError,
)


class FakeClock:
    """Testlerin gerçek zamanda beklememesi için enjekte edilen saat.

    Çok iş parçacıklı testlerde de kullanıldığı için mutasyonlar kilitlidir.
    """

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []
        self._lock = threading.Lock()

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        with self._lock:
            self.slept.append(seconds)
            self.now += seconds


class SahteAgHatasi(OSError):
    """Gerçek transport'un fırlatabileceği ağ hatasını taklit eder."""


class KapanabilirTransport:
    """`close()` sunan sahte transport — kaynak bırakmayı doğrulamak için."""

    def __init__(self) -> None:
        self.closed = 0

    def __call__(self, payload):
        return 200, ok_body()

    def close(self) -> None:
        self.closed += 1


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


def budget_counts(tmp_path) -> dict:
    return json.loads((tmp_path / "budget.json").read_text(encoding="utf-8"))


def run_in_threads(targets):
    """Verilen çağrılabilirleri paralel çalıştır, hepsinin bitmesini bekle."""
    threads = [threading.Thread(target=target) for target in targets]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert not any(thread.is_alive() for thread in threads), "iş parçacıkları takıldı"


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

    assert budget_counts(tmp_path) == {"test": 2}
    assert list(tmp_path.glob(".budget-*")) == [], "atomik yazımdan artık kalmamalı"


def test_retry_storm_never_exceeds_global_budget(tmp_path):
    """Kritik: kalıcı sayaç tavanı retry fırtınasında bile aşmamalı.

    Eski kod bütçeyi döngü ÖNCESİ bir kez kontrol edip her denemede harcıyordu;
    global_budget=2 ile 3 gönderim yapıp diske {'test': 3} yazıyordu.
    """
    calls = []

    def transport(payload):
        calls.append(payload)
        return 500, {"error": "bozuk"}

    client, _ = make_client(tmp_path, transport, global_budget=2, stage_budget=99)

    with pytest.raises(BudgetExceeded):
        client.chat(MSG, stage="test")

    assert len(calls) == 2, "tavan 2 iken 2'den fazla gönderim olmamalı"
    assert client.sends_made == 2
    counts = budget_counts(tmp_path)
    assert sum(counts.values()) == 2, f"diskteki sayaç tavanı aştı: {counts}"


def test_stage_budget_also_holds_mid_retry(tmp_path):
    calls = []

    def transport(payload):
        calls.append(payload)
        return 503, {"error": "mesgul"}

    client, _ = make_client(tmp_path, transport, global_budget=99, stage_budget=2)
    with pytest.raises(BudgetExceeded):
        client.chat(MSG, stage="test")
    assert len(calls) == 2
    assert budget_counts(tmp_path)["test"] == 2


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
    assert budget_counts(tmp_path)["test"] == 2, "tavanı zorlayan sayaç disktekidir"
    assert clock.slept, "retry öncesi backoff uygulanmalı"


def test_client_error_status_is_not_retried(tmp_path):
    """401/400/404 retry ile düzelmez: tek gönderim, ama hata olarak sayılır."""
    calls = []

    def transport(payload):
        calls.append(payload)
        return 401, {"error": "yetkisiz"}

    client, clock = make_client(tmp_path, transport, global_budget=99, stage_budget=99)

    with pytest.raises(GatewayError):
        client.chat([{"role": "user", "content": "a"}], stage="test")
    assert len(calls) == 1, "başarısız olacağı kesin istek yeniden denenmemeli"
    assert client.sends_made == 1
    assert budget_counts(tmp_path)["test"] == 1
    assert clock.slept == [], "backoff beklemesi olmamalı"

    with pytest.raises(GatewayError):
        client.chat([{"role": "user", "content": "b"}], stage="test")
    with pytest.raises(GatewayError):
        client.chat([{"role": "user", "content": "c"}], stage="test")
    with pytest.raises(CircuitOpen):
        client.chat([{"role": "user", "content": "d"}], stage="test")
    assert len(calls) == 3


def test_transport_exception_is_retried_and_logged(tmp_path):
    """Taşıma istisnası (ağ hatası) log'lanmalı ve yeniden denenmeli."""
    attempts = {"n": 0}

    def transport(payload):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise SahteAgHatasi("sunucuya ulasilamadi")
        return 200, ok_body("nihayet")

    client, clock = make_client(tmp_path, transport, global_budget=99, stage_budget=99)
    assert client.chat(MSG, stage="test") == "nihayet"
    assert attempts["n"] == 3
    assert client.sends_made == 3, "istisna atan gönderim de bütçeden sayılmalı"
    assert budget_counts(tmp_path)["test"] == 3
    assert clock.slept, "istisna sonrası backoff uygulanmalı"

    lines = (tmp_path / "calls.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3, "her deneme için bir log satırı olmalı"
    first = json.loads(lines[0])
    assert first["status"] is None
    assert first["error"] == "SahteAgHatasi"


def test_transport_exception_opens_circuit(tmp_path):
    calls = []

    def transport(payload):
        calls.append(payload)
        raise SahteAgHatasi("sunucuya ulasilamadi")

    client, _ = make_client(tmp_path, transport, global_budget=99, stage_budget=99)
    for i in range(3):
        with pytest.raises(GatewayError):
            client.chat([{"role": "user", "content": f"m{i}"}], stage="test")

    sends_before = len(calls)
    with pytest.raises(CircuitOpen):
        client.chat([{"role": "user", "content": "sonraki"}], stage="test")
    assert len(calls) == sends_before, "devre açıkken hiç istek gitmemeli"


def test_malformed_200_body_counts_as_failure(tmp_path):
    """HTTP 200 ama gövde bozuk: sonsuza dek çağrılmamalı, devre açılmalı."""
    calls = []

    def transport(payload):
        calls.append(payload)
        return 200, {"tuhaf": "govde"}

    client, _ = make_client(tmp_path, transport, global_budget=99, stage_budget=99)
    for i in range(3):
        with pytest.raises(GatewayError):
            client.chat([{"role": "user", "content": f"m{i}"}], stage="test")
    assert len(calls) == 3, "şekil hatası yeniden denenmemeli"

    with pytest.raises(CircuitOpen):
        client.chat([{"role": "user", "content": "sonraki"}], stage="test")
    assert len(calls) == 3


def test_unknown_stage_is_rejected_before_sending(tmp_path):
    calls = []

    def transport(payload):
        calls.append(payload)
        return 200, ok_body()

    client, _ = make_client(tmp_path, transport)
    with pytest.raises(ValueError):
        client.chat(MSG, stage="tesst")
    assert calls == [], "tanımsız aşama tek bir istek bile atmamalı"


def test_truncated_budget_file_is_fatal(tmp_path):
    calls = []

    def transport(payload):
        calls.append(payload)
        return 200, ok_body()

    client, _ = make_client(tmp_path, transport, global_budget=2, stage_budget=99)
    # Yazım ortasında çökmüş gibi kırpılmış dosya
    (tmp_path / "budget.json").write_text('{"test": 14', encoding="utf-8")

    with pytest.raises(BudgetCorrupted):
        client.chat(MSG, stage="test")
    assert calls == [], "bozuk bütçe dosyası sayacı sıfırlamış sayılmamalı"


def test_non_dict_budget_file_is_fatal(tmp_path):
    calls = []

    def transport(payload):
        calls.append(payload)
        return 200, ok_body()

    client, _ = make_client(tmp_path, transport, global_budget=2, stage_budget=99)
    (tmp_path / "budget.json").write_text("null", encoding="utf-8")

    with pytest.raises(BudgetCorrupted):
        client.chat(MSG, stage="test")
    assert calls == []


def test_concurrent_callers_never_exceed_global_budget(tmp_path):
    guard = threading.Lock()
    calls = []

    def transport(payload):
        with guard:
            calls.append(payload)
        return 200, ok_body()

    client, _ = make_client(tmp_path, transport, global_budget=5, stage_budget=99)

    raised: list[BaseException] = []

    def worker(index):
        def run():
            try:
                client.chat([{"role": "user", "content": f"m{index}"}], stage="test")
            except BaseException as exc:  # noqa: BLE001 — testte sınıfı doğruluyoruz
                with guard:
                    raised.append(exc)

        return run

    run_in_threads([worker(i) for i in range(12)])

    assert len(calls) == 5, f"eşzamanlı çağrılar tavanı deldi: {len(calls)}"
    assert budget_counts(tmp_path) == {"test": 5}
    assert len(raised) == 7
    assert all(isinstance(exc, BudgetExceeded) for exc in raised), raised


def test_concurrent_sends_never_exceed_max_concurrency(tmp_path):
    state = {"in_flight": 0, "max_in_flight": 0}
    guard = threading.Lock()
    # İkili buluşma noktası: aynı anda 2 gönderim olamıyorsa burada kırılır.
    pair = threading.Barrier(2, timeout=10.0)

    def transport(payload):
        with guard:
            state["in_flight"] += 1
            state["max_in_flight"] = max(state["max_in_flight"], state["in_flight"])
        try:
            pair.wait()
        finally:
            with guard:
                state["in_flight"] -= 1
        return 200, ok_body()

    client, _ = make_client(tmp_path, transport, global_budget=99, stage_budget=99)

    raised: list[BaseException] = []

    def worker(index):
        def run():
            try:
                client.chat([{"role": "user", "content": f"m{index}"}], stage="test")
            except BaseException as exc:  # noqa: BLE001
                with guard:
                    raised.append(exc)

        return run

    run_in_threads([worker(i) for i in range(4)])

    assert raised == [], f"eşzamanlı çağrılar hata verdi: {raised}"
    assert state["max_in_flight"] == 2, (
        f"aynı anda uçuşta olan istek sayısı 2 olmalı, görülen: {state['max_in_flight']}"
    )


def test_shared_budget_file_survives_concurrent_clients(tmp_path):
    """İki ayrı istemci aynı bütçe dosyasını paylaşır: kayıp güncelleme olmamalı.

    `self._lock` istemci başınadır; süreçler arası korumayı `fcntl.flock`
    sağlar. flock kilitleri açık dosya tanımına bağlıdır, bu yüzden aynı süreç
    içindeki iki ayrı `open()` de birbirini bekler.
    """
    guard = threading.Lock()
    calls = []

    def transport(payload):
        with guard:
            calls.append(payload)
        return 200, ok_body()

    client_a, _ = make_client(tmp_path, transport, global_budget=6, stage_budget=99)
    client_b, _ = make_client(tmp_path, transport, global_budget=6, stage_budget=99)

    raised: list[BaseException] = []

    def worker(index):
        client = client_a if index % 2 == 0 else client_b

        def run():
            try:
                client.chat([{"role": "user", "content": f"m{index}"}], stage="test")
            except BaseException as exc:  # noqa: BLE001
                with guard:
                    raised.append(exc)

        return run

    run_in_threads([worker(i) for i in range(14)])

    assert len(calls) == 6, f"paylaşılan tavan delindi: {len(calls)}"
    assert budget_counts(tmp_path) == {"test": 6}, "kayıp güncelleme var"
    assert len(raised) == 8
    assert all(isinstance(exc, BudgetExceeded) for exc in raised), raised


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


def test_close_releases_transport_and_context_manager_works(tmp_path):
    transport = KapanabilirTransport()
    client, _ = make_client(tmp_path, transport)

    with client as entered:
        assert entered is client
        assert client.chat(MSG, stage="test") == "merhaba"
    assert transport.closed == 1, "context manager çıkışında transport kapanmalı"

    client.close()
    assert transport.closed == 2, "close() tekrar çağrılabilir olmalı"


def test_close_is_noop_for_plain_callable_transport(tmp_path):
    def transport(payload):
        return 200, ok_body()

    client, _ = make_client(tmp_path, transport)
    client.close()  # close()'u olmayan transport ile de patlamamalı
