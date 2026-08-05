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


def make_client(
    tmp_path,
    transport,
    *,
    global_budget=10,
    stage_budget=10,
    rps=1.0,
    base_url="https://example.invalid/Jailbreak",
    clock=None,
):
    """Sahte transport'lu istemci.

    `clock` verilirse iki istemci AYNI sahte saati paylaşır — paylaşılan hız
    sınırlayıcının gerçekten tek bütçeye uyduğunu ölçebilmek için şart.
    `base_url` verilirse istemci ayrı bir paylaşılan durum kovasına düşer.
    """
    clock = clock if clock is not None else FakeClock()
    cfg = GatewayConfig(
        base_url=base_url,
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


def test_remaining_budget_reports_untouched_caps(tmp_path):
    def transport(payload):
        return 200, ok_body()

    client, _ = make_client(tmp_path, transport, global_budget=20, stage_budget=5)
    assert client.remaining_budget("test") == (5, 20)


def test_remaining_budget_shrinks_as_budget_is_spent(tmp_path):
    def transport(payload):
        return 200, ok_body()

    client, _ = make_client(tmp_path, transport, global_budget=20, stage_budget=5)
    client.chat([{"role": "user", "content": "a"}], stage="test")
    client.chat([{"role": "user", "content": "b"}], stage="test")
    assert client.remaining_budget("test") == (3, 18)


def test_remaining_budget_counts_other_stages_against_global_only(tmp_path):
    """Başka aşamaların harcaması globali düşürür, aşama kalanını değil."""

    def transport(payload):
        return 200, ok_body()

    client, _ = make_client(tmp_path, transport, global_budget=20, stage_budget=5)
    (tmp_path / "budget.json").write_text('{"baska": 12}', encoding="utf-8")
    assert client.remaining_budget("test") == (5, 8)


def test_remaining_budget_never_goes_negative(tmp_path):
    def transport(payload):
        return 200, ok_body()

    client, _ = make_client(tmp_path, transport, global_budget=20, stage_budget=5)
    (tmp_path / "budget.json").write_text('{"test": 99}', encoding="utf-8")
    assert client.remaining_budget("test") == (0, 0)


def test_remaining_budget_rejects_unknown_stage(tmp_path):
    def transport(payload):
        return 200, ok_body()

    client, _ = make_client(tmp_path, transport)
    with pytest.raises(ValueError):
        client.remaining_budget("tesst")


def test_remaining_budget_sends_nothing(tmp_path):
    calls = []

    def transport(payload):
        calls.append(payload)
        return 200, ok_body()

    client, _ = make_client(tmp_path, transport)
    client.remaining_budget("test")
    assert calls == []
    assert client.sends_made == 0
    assert not (tmp_path / "budget.json").exists(), "salt okunur olmalı"


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

    # `rps` bilerek çok yüksek: hız sınırlayıcı hiç uyumaz, `clock.slept`
    # SAF backoff dizisi olur. Varsayılan 1 istek/sn ile bu ayrım imkânsızdı —
    # backoff tamamen kaldırıldığında hız sınırlayıcı aynı 1.0'ı yazdığı için
    # `assert clock.slept == [1.0]` bile yeşil kalıyordu (mutasyonla doğrulandı).
    # Hız sınırlayıcının kendi testleri ayrı: test_rate_limiter_* .
    client, clock = make_client(
        tmp_path, transport, global_budget=99, stage_budget=99, rps=1000.0
    )
    assert client.chat(MSG, stage="test") == "nihayet"
    assert attempts["n"] == 2
    assert client.sends_made == 2, "retry de bütçeden sayılmalı"
    assert budget_counts(tmp_path)["test"] == 2, "tavanı zorlayan sayaç disktekidir"
    # Tek retry → tek backoff → 2.0**0 = 1.0.
    assert clock.slept == [1.0], f"backoff dizisi beklenenden farklı: {clock.slept}"


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

    # `rps` yüksek → `clock.slept` saf backoff dizisi (bkz. 429 testindeki not).
    client, clock = make_client(
        tmp_path, transport, global_budget=99, stage_budget=99, rps=1000.0
    )
    assert client.chat(MSG, stage="test") == "nihayet"
    assert attempts["n"] == 3
    assert client.sends_made == 3, "istisna atan gönderim de bütçeden sayılmalı"
    assert budget_counts(tmp_path)["test"] == 3
    # İki retry → iki backoff → 2.0**0, 2.0**1.
    assert clock.slept == [1.0, 2.0], f"backoff dizisi beklenenden farklı: {clock.slept}"

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


@pytest.mark.parametrize(
    "govde",
    [
        '{"test": true}',  # bool, Python'da int'in alt sınıfı
        '{"test": false}',
        '{"test": -1000}',  # negatif sayaç toplamı küçültür
        '{"test": 1, "smoke": true, "diger": -1000}',  # 1500 tavanını genişletirdi
        '{"test": 1.5}',
        '{"test": "3"}',
    ],
)
def test_out_of_domain_budget_values_are_fatal(tmp_path, govde):
    """Sayaç yalnızca negatif olmayan gerçek int olabilir.

    `{"a": true, "b": -1000}` eski kontrolü geçip -999 topluyordu, yani
    global tavanı sessizce genişletiyordu. `_read_budget` operatöre bu dosyayı
    elle onarmasını söylüyor — bu değerler erişilebilir.
    """
    calls = []

    def transport(payload):
        calls.append(payload)
        return 200, ok_body()

    client, _ = make_client(tmp_path, transport, global_budget=2, stage_budget=99)
    (tmp_path / "budget.json").write_text(govde, encoding="utf-8")

    with pytest.raises(BudgetCorrupted):
        client.chat(MSG, stage="test")
    assert calls == [], "alan dışı sayaç tek bir istek bile attırmamalı"


def test_zero_budget_value_is_accepted(tmp_path):
    """0 geçerli bir sayaçtır — negatif olmayan int kuralı 0'ı dışlamamalı."""

    def transport(payload):
        return 200, ok_body()

    client, _ = make_client(tmp_path, transport, global_budget=99, stage_budget=99)
    (tmp_path / "budget.json").write_text('{"test": 0}', encoding="utf-8")

    assert client.chat(MSG, stage="test") == "merhaba"
    assert budget_counts(tmp_path)["test"] == 1


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


# --- paylaşılan süreç içi durum (hız sınırlayıcı + semafor + devre kesici) ---


def test_rate_limiter_is_shared_across_clients(tmp_path):
    """İki ayrı istemci TEK bir 1 istek/sn bütçesine uymalı.

    Regresyon: durum `__init__`'te tutulurken (istemci başına) iki istemci
    aynı süreçte 1.00 sn içinde 4 istek gönderiyordu — `build_default_client()`
    her çağrıda taze bir istemci ürettiği için bu Plan 2'nin doğal şekliydi.
    """
    sent = []

    def transport(payload):
        sent.append(payload)
        return 200, ok_body()

    clock = FakeClock()
    client_a, _ = make_client(
        tmp_path, transport, global_budget=99, stage_budget=99, clock=clock
    )
    client_b, _ = make_client(
        tmp_path, transport, global_budget=99, stage_budget=99, clock=clock
    )

    for index, client in enumerate((client_a, client_b, client_a, client_b)):
        client.chat([{"role": "user", "content": f"m{index}"}], stage="test")

    assert len(sent) == 4
    # 4 gönderim, 1 istek/sn → aralarında tam 3 tane 1 sn'lik bekleme olmalı.
    assert clock.slept == [1.0, 1.0, 1.0], (
        f"paylaşılan hız sınırlayıcı devrede değil: {clock.slept}"
    )
    assert clock.now >= 3.0


def test_concurrency_semaphore_is_shared_across_clients(tmp_path):
    """Semafor da paylaşılır: iki istemci TOPLAMDA 2 eşzamanlı gönderim yapar.

    Buluşma noktası bilerek 3 kişilik: paylaşılan semafor (2) altında hiçbir
    zaman dolamaz ve zaman aşımıyla kırılır. İstemci başına semaforla (2+2=4)
    üç iş parçacığı buluşur ve `max_in_flight` 3'e çıkar — regresyon budur.
    """
    state = {"in_flight": 0, "max_in_flight": 0}
    guard = threading.Lock()
    ucler = threading.Barrier(3, timeout=0.5)

    def transport(payload):
        with guard:
            state["in_flight"] += 1
            state["max_in_flight"] = max(state["max_in_flight"], state["in_flight"])
        try:
            ucler.wait()
        except threading.BrokenBarrierError:
            pass  # beklenen: paylaşılan semafor 3. gönderimi hiç başlatmıyor
        finally:
            with guard:
                state["in_flight"] -= 1
        return 200, ok_body()

    client_a, _ = make_client(tmp_path, transport, global_budget=99, stage_budget=99)
    client_b, _ = make_client(tmp_path, transport, global_budget=99, stage_budget=99)

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

    run_in_threads([worker(i) for i in range(4)])

    assert raised == [], f"eşzamanlı çağrılar hata verdi: {raised}"
    assert state["max_in_flight"] == 2, (
        f"iki istemci semaforu paylaşmalı, görülen eşzamanlılık: {state['max_in_flight']}"
    )


def test_separate_endpoints_get_separate_semaphores(tmp_path):
    """Kayıt defteri anahtarı gerçekten `base_url`: ayrı endpoint, ayrı semafor."""
    state = {"in_flight": 0, "max_in_flight": 0}
    guard = threading.Lock()
    dortler = threading.Barrier(4, timeout=10.0)

    def transport(payload):
        with guard:
            state["in_flight"] += 1
            state["max_in_flight"] = max(state["max_in_flight"], state["in_flight"])
        try:
            dortler.wait()
        finally:
            with guard:
                state["in_flight"] -= 1
        return 200, ok_body()

    client_a, _ = make_client(tmp_path, transport, global_budget=99, stage_budget=99)
    client_b, _ = make_client(
        tmp_path,
        transport,
        global_budget=99,
        stage_budget=99,
        base_url="https://baska.invalid/Jailbreak",
    )

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

    run_in_threads([worker(i) for i in range(4)])

    assert raised == [], f"eşzamanlı çağrılar hata verdi: {raised}"
    assert state["max_in_flight"] == 4, (
        "iki farklı endpoint 2+2 eşzamanlılık vermeli, görülen: "
        f"{state['max_in_flight']}"
    )


def test_circuit_opened_by_one_client_is_seen_by_another(tmp_path):
    """Bir istemcinin açtığı devre kesici diğerini de durdurur."""
    calls = []

    def transport(payload):
        calls.append(payload)
        return 500, {"error": "bozuk"}

    client_a, _ = make_client(tmp_path, transport, global_budget=99, stage_budget=99)
    client_b, _ = make_client(tmp_path, transport, global_budget=99, stage_budget=99)

    for i in range(3):
        with pytest.raises(GatewayError):
            client_a.chat([{"role": "user", "content": f"m{i}"}], stage="test")

    sends_before = len(calls)
    with pytest.raises(CircuitOpen):
        client_b.chat([{"role": "user", "content": "b-den"}], stage="test")
    assert len(calls) == sends_before, (
        "devre A tarafından açıldı; B tek bir istek bile atmamalı"
    )


def test_failure_counter_is_shared_across_clients(tmp_path):
    """Ardışık hata sayacı istemciler arasında birikir."""
    calls = []

    def transport(payload):
        calls.append(payload)
        return 500, {"error": "bozuk"}

    client_a, _ = make_client(tmp_path, transport, global_budget=99, stage_budget=99)
    client_b, _ = make_client(tmp_path, transport, global_budget=99, stage_budget=99)

    # Hatalar ikiye bölünüyor: A'da 2, B'de 1 → eşik (3) yine dolmalı.
    with pytest.raises(GatewayError):
        client_a.chat([{"role": "user", "content": "a"}], stage="test")
    with pytest.raises(GatewayError):
        client_b.chat([{"role": "user", "content": "b"}], stage="test")
    with pytest.raises(GatewayError):
        client_a.chat([{"role": "user", "content": "c"}], stage="test")

    with pytest.raises(CircuitOpen):
        client_b.chat([{"role": "user", "content": "d"}], stage="test")


def test_separate_endpoints_do_not_share_state(tmp_path):
    """Kayıt defteri `base_url` ile anahtarlanır: farklı endpoint, farklı devre."""

    def transport(payload):
        return 500, {"error": "bozuk"}

    def ok_transport(payload):
        return 200, ok_body("baska-endpoint")

    client_a, _ = make_client(tmp_path, transport, global_budget=99, stage_budget=99)
    client_b, _ = make_client(
        tmp_path,
        ok_transport,
        global_budget=99,
        stage_budget=99,
        base_url="https://baska.invalid/Jailbreak",
    )

    for i in range(3):
        with pytest.raises(GatewayError):
            client_a.chat([{"role": "user", "content": f"m{i}"}], stage="test")

    with pytest.raises(CircuitOpen):
        client_a.chat([{"role": "user", "content": "yine-a"}], stage="test")
    # B başka bir endpoint: A'nın devresi onu bağlamaz.
    assert client_b.chat([{"role": "user", "content": "b"}], stage="test") == (
        "baska-endpoint"
    )


def test_conflicting_max_concurrency_is_rejected(tmp_path):
    """Canlı semafor güvenle küçültülemez — uyuşmazlık sessizce yutulmaz."""

    def transport(payload):
        return 200, ok_body()

    make_client(tmp_path, transport)

    cfg = GatewayConfig(
        base_url="https://example.invalid/Jailbreak",
        model="hakem-llm",
        api_key="test-key",
        max_concurrency=5,
        stage_budgets={"test": 10},
    )
    with pytest.raises(ValueError, match="eşzamanlılık"):
        GatewayClient(
            cfg,
            cache_dir=tmp_path / "cache",
            budget_path=tmp_path / "budget.json",
            log_path=tmp_path / "calls.jsonl",
            transport=transport,
        )


def test_strictest_rate_limit_wins_across_clients(tmp_path):
    """Daha gevşek bir ikinci istemci ortak hız bütçesini gevşetemez."""
    sent = []

    def transport(payload):
        sent.append(payload)
        return 200, ok_body()

    clock = FakeClock()
    # Önce gevşek (10/sn), sonra katı (1/sn) — katı olan kazanmalı.
    client_gevsek, _ = make_client(
        tmp_path, transport, global_budget=99, stage_budget=99, rps=10.0, clock=clock
    )
    client_kati, _ = make_client(
        tmp_path, transport, global_budget=99, stage_budget=99, rps=1.0, clock=clock
    )

    client_gevsek.chat([{"role": "user", "content": "a"}], stage="test")
    client_gevsek.chat([{"role": "user", "content": "b"}], stage="test")

    assert clock.slept == [1.0], (
        f"gevşek istemci de katı aralığa uymalı: {clock.slept}"
    )
    assert client_kati is not client_gevsek


def _fake_default_env(tmp_path, monkeypatch):
    """`build_default_client()`'ı gerçek yollara ve gerçek HTTP'ye dokunmadan kur."""
    from aax import config as cfg
    from aax import gateway

    monkeypatch.setenv("APP_KEY_JAILBREAK", "test-key")
    monkeypatch.setattr(cfg, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(cfg, "BUDGET_PATH", tmp_path / "budget.json")
    monkeypatch.setattr(cfg, "CALL_LOG_PATH", tmp_path / "calls.jsonl")
    monkeypatch.setattr(
        gateway, "_httpx_transport", lambda *a, **k: (lambda payload: (200, ok_body()))
    )


def test_build_default_client_is_memoized(tmp_path, monkeypatch):
    """Tekrarlanan çağrı AYNI istemciyi döndürmeli — her seferinde taze değil.

    Regresyon: her çağrıda yeni istemci üretmek, koruma durumu istemci başına
    tutulduğunda hız sınırını ve devre kesiciyi sessizce sıfırlıyordu.
    """
    from aax.gateway import build_default_client, reset_shared_state

    _fake_default_env(tmp_path, monkeypatch)

    first = build_default_client()
    second = build_default_client()
    assert first is second

    # Farklı aşama bütçeleri ayrı bir istemci demektir (bilinçli).
    ozel = build_default_client({"test": 3})
    assert ozel is not first

    reset_shared_state()
    assert build_default_client() is not first, "sıfırlama memo'yu da temizlemeli"


def test_build_default_client_shares_state_with_manual_client(tmp_path, monkeypatch):
    """`build_default_client()` ile elle kurulan istemci aynı devreyi paylaşır."""
    from aax import config as cfg
    from aax.gateway import build_default_client

    _fake_default_env(tmp_path, monkeypatch)

    varsayilan = build_default_client()
    elle, _ = make_client(
        tmp_path,
        lambda payload: (200, ok_body()),
        global_budget=99,
        stage_budget=99,
        base_url=cfg.GATEWAY_BASE_URL,
    )
    assert varsayilan._state is elle._state


def test_api_key_is_not_in_config_repr():
    """`repr(config)` anahtarı basmamalı — Plan 2-4 notebook dostu."""
    cfg = GatewayConfig(
        base_url="https://example.invalid/Jailbreak",
        model="hakem-llm",
        api_key="cok-gizli-anahtar",
        stage_budgets={"test": 1},
    )
    assert "cok-gizli-anahtar" not in repr(cfg)
    assert "example.invalid" in repr(cfg), "diğer alanlar hâlâ görünmeli"


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


def test_cache_hit_log_row_has_same_schema_as_send_row(tmp_path):
    """Cache satırı ile gönderim satırı aynı anahtar kümesini taşımalı.

    Bu JSONL projenin planlar arası TEK denetim izi. Cache satırı `attempt`,
    `prompt_tokens` ve `completion_tokens`'ı hiç yazmıyordu; satır satır
    değişen bir şema onu okuyan her aracı `.get()` savunmasına zorlardı.
    """

    def transport(payload):
        return 200, ok_body()

    client, _ = make_client(tmp_path, transport)
    client.chat(MSG, stage="test")  # gönderim satırı
    client.chat(MSG, stage="test")  # cache satırı

    lines = (tmp_path / "calls.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    gonderim, cache = json.loads(lines[0]), json.loads(lines[1])

    assert gonderim["cached"] is False and cache["cached"] is True
    assert set(cache) == set(gonderim), (
        f"şema uyuşmuyor — yalnızca gönderimde: {set(gonderim) - set(cache)}, "
        f"yalnızca cache'te: {set(cache) - set(gonderim)}"
    )
    assert set(gonderim) == {
        "ts",
        "stage",
        "status",
        "cached",
        "attempt",
        "latency",
        "prompt_tokens",
        "completion_tokens",
    }
    # Cache satırında bu alanlar anlamsız ama açıkça None olarak var olmalı.
    assert cache["attempt"] is None
    assert cache["prompt_tokens"] is None
    assert cache["completion_tokens"] is None
    assert gonderim["prompt_tokens"] == 5


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
