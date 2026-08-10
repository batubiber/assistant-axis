"""hakem-llm gateway istemcisi.

Bu modül dışarı giden TEK HTTP noktasıdır. Hız sınırlama, disk cache'i,
kalıcı bütçe ve devre kesici burada kapalıdır; çağıran taraf hiçbirini bilmez.

Bütçe her HTTP gönderimini sayar (retry'lar dahil) — retry'ların tavanı
gizlice aşamaması için bilinçli olarak muhafazakâr. Bütçe kontrolü ve harcaması
HER denemede aynı kilit bloğunda yapılır; tavan aşılırsa retry döngüsünün
ortasında bile `BudgetExceeded` yükselir ve `GatewayError` içine yutulmaz.

Kapsam farkı — bilerek böyle:

* **Bütçe disktedir ve süreçler arasıdır.** `fcntl.flock` ile korunan
  oku-değiştir-yaz sayesinde aynı `budget_path`'i paylaşan farklı istemciler
  (ve farklı süreçler) birbirinin güncellemesini ezmez. Tavan globaldir.
* **Hız sınırlayıcı, eşzamanlılık semaforu ve devre kesici SÜREÇ İÇİNDE
  paylaşılır — istemci başına değil.** Durum, `base_url` ile anahtarlanan
  modül düzeyinde bir kayıt defterinde tutulur (`_ENDPOINT_STATE`). Aynı
  endpoint'e bakan kaç `GatewayClient` üretirsen üret hepsi TEK bir 1 istek/sn
  bütçesine, TEK bir semafora ve TEK bir devre kesiciye uyar. Bu bilinçli:
  Plan 2'nin doğal şekli aşama başına bir istemci ya da eşzamanlı yakalama +
  hakemlik; istemci başına durum o kurulumda sunucuya giden hızı sessizce
  ikiye katlardı.
* **Süreçler arası paylaşım yoktur.** İki ayrı süreç çalıştırırsan sunucuya
  giden hız 1 değil 2 istek/sn olur ve devre kesici paylaşılmaz. Süreçler
  arasında sert garanti veren tek şey disktedeki bütçedir; koşuları tek
  süreçte tut.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

Transport = Callable[[dict], tuple[int, dict]]


class GatewayError(RuntimeError):
    """Retry'lar tükendikten sonra çağrı başarısız."""


class BudgetExceeded(RuntimeError):
    """Aşama veya global çağrı tavanı doldu."""


class BudgetCorrupted(RuntimeError):
    """Bütçe dosyası okunamıyor — sayaç sıfırlanmış sayılamaz.

    Bütçe koruması kapalı yönde (fail-closed) hata verir: bozuk dosyayı boş
    sözlük kabul etmek 1500'lük tavanı sessizce kaldırırdı.
    """


class CircuitOpen(RuntimeError):
    """Ardışık hatalar nedeniyle koşu durduruldu."""


@dataclass
class GatewayConfig:
    base_url: str
    model: str
    # `repr=False`: Plan 2-4 notebook dostu. `print(config)` ya da bir hücre
    # çıktısı anahtarı `.ipynb`'ye gömerdi; anahtar hiçbir dosyaya yazılmaz.
    api_key: str = field(repr=False)
    requests_per_second: float = 1.0
    max_concurrency: int = 2
    global_budget: int = 1500
    stage_budgets: dict[str, int] = field(default_factory=dict)
    # Çoklu model desteği (2026-08-10 bütçe düzeltmesi): `model_dependent_stages`
    # içindeki aşamalarda diskteki sayaç anahtarı `f"{stage}:{model_slug}"` olur
    # — tavan HER MODEL İÇİN AYRI uygulanır. Boş küme (varsayılan) eski
    # davranışı birebir korur: hiçbir aşama bölünmez, hepsi bare anahtarla
    # sayılır. `model_slug` yalnızca bu kümedeki aşamalarda okunur; boşsa ve
    # bir model-bağımlı aşama kullanılırsa `_ledger_key` `ValueError` verir —
    # sessizce bare anahtara düşmek iki modelin sayaçlarını birleştirirdi.
    model_dependent_stages: frozenset[str] = field(default_factory=frozenset)
    model_slug: str = ""
    max_retries: int = 3
    circuit_threshold: int = 3
    timeout_seconds: float = 120.0


@dataclass
class _EndpointState:
    """Bir endpoint için süreç genelinde paylaşılan koruma durumu.

    Bu alanlar bilerek `GatewayClient` örneğinde DEĞİL burada: aynı sunucuya
    bakan iki istemci tek bir hız bütçesine ve tek bir devre kesiciye uymalı.
    `lock` bu alanların tamamını korur; `semaphore` aynı anda uçuşta olan
    gönderim sayısını sınırlar.
    """

    lock: threading.Lock
    semaphore: threading.Semaphore
    max_concurrency: int
    min_interval: float
    last_send_at: float | None = None
    consecutive_failures: int = 0
    circuit_open: bool = False


_REGISTRY_LOCK = threading.Lock()
_ENDPOINT_STATE: dict[str, _EndpointState] = {}
_DEFAULT_CLIENTS: dict[tuple, "GatewayClient"] = {}


def _endpoint_state(
    base_url: str, max_concurrency: int, min_interval: float
) -> _EndpointState:
    """`base_url` için paylaşılan durumu getir, yoksa oluştur.

    İki uyuşmazlık kuralı — ikisi de kapalı yönde:

    * `max_concurrency` farklıysa `ValueError`. Canlı bir semaforu güvenle
      küçültemeyiz; farkı sessizce yutmak daha katı olan ayarı kaybettirirdi.
    * `min_interval` farklıysa **en katısı** (en büyüğü) kazanır. Daha gevşek
      bir ikinci istemci ortak hız bütçesini gevşetemez.
    """
    with _REGISTRY_LOCK:
        state = _ENDPOINT_STATE.get(base_url)
        if state is None:
            state = _EndpointState(
                lock=threading.Lock(),
                semaphore=threading.Semaphore(max_concurrency),
                max_concurrency=max_concurrency,
                min_interval=min_interval,
            )
            _ENDPOINT_STATE[base_url] = state
            return state
        if state.max_concurrency != max_concurrency:
            raise ValueError(
                f"'{base_url}' için eşzamanlılık sınırı zaten "
                f"{state.max_concurrency} olarak kuruldu; {max_concurrency} istendi. "
                "Aynı endpoint'e bakan istemciler tek bir semaforu paylaşır — "
                "aynı max_concurrency ile kur."
            )
        if min_interval > state.min_interval:
            state.min_interval = min_interval
        return state


def reset_shared_state() -> None:
    """Modül düzeyindeki paylaşılan durumu sıfırla — YALNIZCA testler için.

    Üretim kodunda çağrılmaz: devre kesiciyi sıfırlamak, tam da onu açtıran
    sunucuyu yeniden dövmek demektir. Testler arası sızıntıyı önlemek için
    `tests/conftest.py` bunu her testten önce ve sonra çağırır.
    """
    with _REGISTRY_LOCK:
        _ENDPOINT_STATE.clear()
        _DEFAULT_CLIENTS.clear()


def _is_valid_count(value: object) -> bool:
    """Bütçe sayacı olarak kabul edilebilir mi?

    `isinstance(value, int)` tek başına yetmez, iki ayrı sızıntısı var:

    * `bool` Python'da `int`'in alt sınıfıdır — `{"a": true}` sayaç 1 olurdu.
      Bu, `judge.py:96`'da zaten kapatılmış hata sınıfının kardeşidir.
    * Negatif değer TOPLAMI küçültür: `{"a": true, "b": -1000}` doğrulamayı
      geçip -999 toplar, yani 1500'lük global tavanı fiilen genişletirdi.

    `_read_budget` docstring'i operatöre bu dosyayı elle onarmasını söylüyor,
    yani ikisi de erişilebilir senaryolar.
    """
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


class _HttpxTransport:
    """Gerçek HTTP taşıması. `close()` ile bağlantı havuzu kapatılır."""

    def __init__(self, base_url: str, api_key: str, timeout: float) -> None:
        import httpx

        self._url = f"{base_url.rstrip('/')}/v1/chat/completions"
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._client = httpx.Client(timeout=timeout)

    def __call__(self, payload: dict) -> tuple[int, dict]:
        response = self._client.post(self._url, json=payload, headers=self._headers)
        try:
            body = response.json()
        except ValueError:
            body = {"error": response.text[:500]}
        return response.status_code, body

    def close(self) -> None:
        self._client.close()


def _httpx_transport(base_url: str, api_key: str, timeout: float) -> Transport:
    return _HttpxTransport(base_url, api_key, timeout)


class GatewayClient:
    def __init__(
        self,
        config: GatewayConfig,
        *,
        cache_dir: Path,
        budget_path: Path,
        log_path: Path,
        transport: Transport | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.budget_path = Path(budget_path)
        self.log_path = Path(log_path)
        self._monotonic = monotonic
        self._sleep = sleep
        self._transport = transport or _httpx_transport(
            config.base_url, config.api_key, config.timeout_seconds
        )

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.budget_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Hız sınırlayıcı / semafor / devre kesici durumu istemcide DEĞİL,
        # `base_url` ile anahtarlanan modül düzeyindeki kayıt defterinde:
        # aynı sunucuya bakan kaç istemci olursa olsun tek bütçeye uyar.
        self._state = _endpoint_state(
            config.base_url,
            config.max_concurrency,
            1.0 / config.requests_per_second,
        )
        # `sends_made` bilinçli olarak istemci başınadır: "bu istemci kaç
        # istek attı?" bir tanı sorusudur, koruma değil. Koruma disktedeki
        # bütçe sayacı ve paylaşılan devre kesicidir.
        self.sends_made = 0

    # --- yaşam döngüsü ---------------------------------------------------

    def close(self) -> None:
        """Taşıma katmanının kaynaklarını bırak. Birden fazla çağrı güvenli."""
        closer = getattr(self._transport, "close", None)
        if callable(closer):
            closer()

    def __enter__(self) -> "GatewayClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # --- payload ve cache anahtarı -------------------------------------

    def _payload(self, messages: list[dict], temperature: float, max_tokens: int) -> dict:
        return {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

    @staticmethod
    def _cache_key(payload: dict) -> str:
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _cache_read(self, key: str) -> str | None:
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))["content"]
        except (ValueError, KeyError):
            return None

    def _cache_write(self, key: str, content: str) -> None:
        # Yalnızca içerik yazılır — payload ve anahtar cache'e girmez.
        self._cache_path(key).write_text(
            json.dumps({"content": content}, ensure_ascii=False), encoding="utf-8"
        )

    # --- bütçe ----------------------------------------------------------

    @contextmanager
    def _budget_file_lock(self) -> Iterator[None]:
        """Bütçe dosyası için süreçler arası karşılıklı dışlama.

        Kilit `budget.json` üzerinde değil, yanındaki `.lock` dosyası üzerinde
        tutulur: bütçe dosyası her yazımda `os.replace` ile değiştiği için
        (inode değişir) onun üzerindeki flock süreçler arası anlamını
        kaybederdi. Ayrı ve hiç yer değiştirmeyen bir kilit dosyası şart.
        """
        lock_path = self.budget_path.with_name(self.budget_path.name + ".lock")
        handle = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
        finally:
            os.close(handle)

    def _read_budget(self) -> dict[str, int]:
        """Bütçe sayaçlarını oku. Bozuk dosya ÖLÜMCÜLDÜR, sıfır değildir."""
        if not self.budget_path.exists():
            return {}
        raw = self.budget_path.read_text(encoding="utf-8")
        try:
            counts = json.loads(raw)
        except ValueError as exc:
            raise BudgetCorrupted(
                f"Bütçe dosyası ayrıştırılamadı: {self.budget_path}. "
                "Sayaç sıfırlanmış sayılmaz — dosyayı elle onar veya bilinçli olarak sil."
            ) from exc
        if not isinstance(counts, dict) or not all(
            isinstance(key, str) and _is_valid_count(value) for key, value in counts.items()
        ):
            raise BudgetCorrupted(
                f"Bütçe dosyası beklenen şekilde değil "
                f"(str -> negatif olmayan int sözlük): {self.budget_path}."
            )
        return counts

    def _write_budget(self, counts: dict[str, int]) -> None:
        """Atomik yazım: geçici dosya + `os.replace`.

        Yazım ortasında çökme `budget.json`'ı kırpamaz; ya eski ya yeni içerik
        görünür. Kırpılmış dosya tavanı sessizce sıfırlardı.
        """
        directory = self.budget_path.parent
        fd, tmp_name = tempfile.mkstemp(dir=directory, prefix=".budget-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(counts, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, self.budget_path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise

    def _ledger_key(self, stage: str) -> str:
        """`stage` için diskteki sayaç ANAHTARINI hesapla.

        Bu, tavan sorgusunda kullanılan `stage` dizesinden BİLEREK ayrı bir
        adım: `_assert_budget_available` / `remaining_budget` tavanı HER ZAMAN
        çıplak `stage` argümanıyla `self.config.stage_budgets`'ta arar — bu
        sözlük yalnızca kanonik bare aşama adlarıyla anahtarlanır (bkz.
        `config.STAGE_BUDGETS`). Yani `stage="stage2_probe_labels:sahte-model"`
        gibi UYDURULMUŞ bir dize `stage_budgets.get(...)` içinde asla
        bulunmaz ve tavan araması `ValueError` ile reddeder — sayaç anahtarını
        çağıranın serbestçe seçebileceği bir dizeye çevirmek, model-bağımlı
        aşamaları keyfî alt-anahtarlara bölüp tavanı delme kapısı açardı.

        Model-BAĞIMSIZ aşamalarda (`model_dependent_stages` dışında kalanlar)
        anahtar `stage`'in kendisidir — smoke testi ve rol kataloğu gibi bir
        kez üretilip HER model tarafından paylaşılan artefaktlar için tavan
        tektir ve modeller arasında paylaşılır.
        """
        if stage not in self.config.model_dependent_stages:
            return stage
        if not self.config.model_slug:
            raise ValueError(
                f"'{stage}' model-bağımlı bir aşama ama `GatewayConfig.model_slug` "
                "boş — model-bağımlı aşamalar için model_slug ZORUNLUDUR "
                "(bkz. `build_default_client`)."
            )
        return f"{stage}:{self.config.model_slug}"

    def _assert_budget_available(self, stage: str, counts: dict[str, int]) -> None:
        stage_cap = self.config.stage_budgets.get(stage)
        if stage_cap is None:
            # Kapalı yönde hata: tanımsız aşama adı (tipik olarak yazım hatası)
            # alt bütçesiz kalıp global 1500'ün tamamını yiyebilirdi.
            raise ValueError(
                f"Bilinmeyen aşama adı: {stage!r}. "
                f"Tanımlı aşamalar: {sorted(self.config.stage_budgets)}"
            )
        # Global tavan HER ZAMAN diskteki TÜM anahtarların (bare + model-scoped)
        # toplamıdır — model-bağımlı bir aşamanın kendi alt sayacı bu toplamı
        # asla genişletmez.
        total = sum(counts.values())
        if total >= self.config.global_budget:
            raise BudgetExceeded(
                f"Global bütçe doldu: {total}/{self.config.global_budget}"
            )
        ledger_key = self._ledger_key(stage)
        if counts.get(ledger_key, 0) >= stage_cap:
            raise BudgetExceeded(
                f"'{ledger_key}' aşama bütçesi doldu: {counts.get(ledger_key, 0)}/{stage_cap}"
            )

    def _check_budget(self, stage: str) -> None:
        """Salt okunur ön kontrol — ilk gönderimden önce hızlı hata için."""
        with self._budget_file_lock():
            self._assert_budget_available(stage, self._read_budget())

    def _check_and_spend_budget(self, stage: str) -> None:
        """Kontrol ve harcamayı AYNI kilit altında yap.

        İkisini ayırmak tavanı delerdi: retry döngüsünde her deneme kontrolsüz
        harcarsa tek iş parçacığı bile tavanı `max_retries - 1` kadar aşar.
        """
        with self._budget_file_lock():
            counts = self._read_budget()
            self._assert_budget_available(stage, counts)
            ledger_key = self._ledger_key(stage)
            counts[ledger_key] = counts.get(ledger_key, 0) + 1
            self._write_budget(counts)

    # --- hız sınırlama --------------------------------------------------

    def _wait_for_slot(self) -> None:
        """Paylaşılan hız penceresini bekle. Çağıran `self._state.lock`'u tutar."""
        min_interval = self._state.min_interval
        if self._state.last_send_at is not None:
            elapsed = self._monotonic() - self._state.last_send_at
            if elapsed < min_interval:
                self._sleep(min_interval - elapsed)
        self._state.last_send_at = self._monotonic()

    # --- devre kesici ----------------------------------------------------

    def _record_failure(self) -> None:
        with self._state.lock:
            self._state.consecutive_failures += 1
            if self._state.consecutive_failures >= self.config.circuit_threshold:
                self._state.circuit_open = True

    @staticmethod
    def _is_retriable(status: int | None) -> bool:
        """Yalnızca 429 ve 5xx yeniden denenir.

        401/400/404 gibi durumlar retry ile düzelmez; üç kez denemek hem bütçe
        yakar hem ortak sunucuya boşuna yük bindirir.
        """
        if status is None:  # taşıma istisnası — geçici kabul edilir
            return True
        return status == 429 or 500 <= status < 600

    # --- log ------------------------------------------------------------

    def _log(self, entry: dict) -> None:
        """JSONL denetim izine tek satır yaz.

        Şema tek tiptir: her satırda aynı anahtarlar bulunur. Cache isabetinde
        `attempt` / `prompt_tokens` / `completion_tokens` anlamsızdır ama
        anahtarlar açıkça `None` olarak yazılır — bu JSONL projenin planlar
        arası TEK denetim izi ve satır satır değişen bir şema, onu okuyan her
        aracı `.get()` savunmasına zorlardı.
        """
        row = {
            "ts": time.time(),
            "stage": None,
            "status": None,
            "cached": None,
            "attempt": None,
            "latency": None,
            "prompt_tokens": None,
            "completion_tokens": None,
        }
        row.update(entry)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    # --- kamuya açık API -------------------------------------------------

    def would_call(
        self, messages: list[dict], *, temperature: float = 0.0, max_tokens: int = 1024
    ) -> bool:
        """Bu çağrı bütçe harcar mıydı? Hiçbir istek atmaz."""
        payload = self._payload(messages, temperature, max_tokens)
        return self._cache_read(self._cache_key(payload)) is None

    def remaining_budget(self, stage: str) -> tuple[int, int]:
        """`(aşama için kalan, global kalan)` — istek atmaz, bütçe harcamaz.

        `--dry-run` ön kontrolünün ihtiyacı olan sayı budur. Planlanan çağrıyı
        `config.STAGE_BUDGETS[stage]` ile kıyaslamak yanıltıcıdır: aşama
        bütçesinin çoğu önceki bir koşuda harcanmış olabilir ve tavana ne
        kadar KALDIĞI diskteki sayaca bağlıdır.

        Bilinmeyen aşama adı `chat()` ile aynı şekilde `ValueError`'dır:
        `--dry-run`, yazım hatası yüzünden temiz bir 0 dönmemeli.

        Model-bağımlı bir aşamada (`config.model_dependent_stages`) "aşama
        için kalan" bu istemcinin `config.model_slug`'ına ÖZELDİR — başka bir
        modelin aynı aşamadaki harcaması bu sayıyı etkilemez, ama global
        kalanı (ikinci dönüş değeri) her zaman etkiler.
        """
        stage_cap = self.config.stage_budgets.get(stage)
        if stage_cap is None:
            raise ValueError(
                f"Bilinmeyen aşama adı: {stage!r}. "
                f"Tanımlı aşamalar: {sorted(self.config.stage_budgets)}"
            )
        ledger_key = self._ledger_key(stage)
        with self._budget_file_lock():
            counts = self._read_budget()
        stage_remaining = max(0, stage_cap - counts.get(ledger_key, 0))
        global_remaining = max(0, self.config.global_budget - sum(counts.values()))
        return stage_remaining, global_remaining

    def chat(
        self,
        messages: list[dict],
        *,
        stage: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        payload = self._payload(messages, temperature, max_tokens)
        key = self._cache_key(payload)

        cached = self._cache_read(key)
        if cached is not None:
            self._log({"stage": stage, "status": 200, "cached": True, "latency": 0.0})
            return cached

        with self._state.lock:
            if self._state.circuit_open:
                raise CircuitOpen(
                    f"Devre kesici açık ({self.config.circuit_threshold} ardışık hata). "
                    "Koşu durduruldu — sunucuyu zorlamıyoruz."
                )
            # Ön kontrol: tavan zaten doluysa tek bir gönderim bile yapmadan çık.
            self._check_budget(stage)

        last_status: int | None = None
        last_body: dict = {}
        last_error: BaseException | None = None

        for attempt in range(self.config.max_retries):
            with self._state.semaphore:
                with self._state.lock:
                    self._wait_for_slot()
                    # Her denemede yeniden kontrol + harcama, tek kilit altında.
                    # Buradan çıkan BudgetExceeded bilinçli olarak yakalanmaz:
                    # retry ortasında bile GatewayError'a dönüşmeden yükselir.
                    self._check_and_spend_budget(stage)
                    self.sends_made += 1
                started = self._monotonic()
                try:
                    status, body = self._transport(payload)
                except Exception as exc:  # ağ/zaman aşımı/havuz hataları
                    transport_error: BaseException | None = exc
                    status, body = None, {}
                else:
                    transport_error = None
                latency = self._monotonic() - started

            usage = body.get("usage", {}) if isinstance(body, dict) else {}
            entry = {
                "stage": stage,
                "status": status,
                "cached": False,
                "attempt": attempt + 1,
                "latency": latency,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
            }
            if transport_error is not None:
                entry["error"] = type(transport_error).__name__
            self._log(entry)

            if transport_error is None and status == 200:
                try:
                    content = body["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError) as exc:
                    # Bozuk gövdeli 200 de bir hatadır: saymazsak sunucu
                    # sonsuza dek çağrılıp bütçeyi yakar, devre hiç açılmaz.
                    self._record_failure()
                    raise GatewayError(f"Beklenmeyen yanıt şekli: {body}") from exc
                with self._state.lock:
                    self._state.consecutive_failures = 0
                self._cache_write(key, content)
                return content

            last_status, last_body, last_error = status, body, transport_error

            if not self._is_retriable(status):
                break
            if attempt + 1 < self.config.max_retries:
                self._sleep(2.0 ** attempt)

        self._record_failure()
        if last_error is not None:
            raise GatewayError(
                f"Taşıma katmanı hatası ({type(last_error).__name__}): {last_error}"
            ) from last_error
        raise GatewayError(f"Çağrı başarısız (HTTP {last_status}): {last_body}")


def build_default_client(stage_budgets: dict[str, int] | None = None) -> GatewayClient:
    """Gerçek endpoint'e bağlı istemci. Anahtarı ortamdan okur.

    **Memoize edilir.** Aynı aşama bütçeleriyle tekrar çağırmak AYNI istemciyi
    döndürür; her çağrıda yeni bir istemci (ve yeni bir httpx bağlantı havuzu)
    üretmek gereksizdi. Koruma açısından zaten fark etmez — hız sınırlayıcı,
    semafor ve devre kesici `base_url` üzerinden süreç genelinde paylaşılıyor —
    ama bağlantı havuzunu ve `sends_made` tanısını tek yerde tutar.
    """
    from aax import config as cfg

    resolved = cfg.STAGE_BUDGETS if stage_budgets is None else stage_budgets
    # `model_slug()` ÇAĞRI ANINDAKİ `cfg.TARGET_MODEL`'i okur (bkz. o
    # fonksiyonun docstring'i) — memo anahtarına dahil edilmesi şart: aksi
    # halde aynı süreç içinde `AAX_TARGET_MODEL` değişse bile (ör. testte
    # monkeypatch) önbelleğe alınmış istemci ESKİ modelin `model_slug`'ını
    # taşımaya devam ederdi.
    active_model_slug = cfg.model_slug()
    memo_key = (
        cfg.GATEWAY_BASE_URL,
        cfg.GATEWAY_MODEL,
        tuple(sorted(resolved.items())),
        active_model_slug,
    )
    cached_client = _DEFAULT_CLIENTS.get(memo_key)
    if cached_client is not None:
        return cached_client

    gateway_config = GatewayConfig(
        base_url=cfg.GATEWAY_BASE_URL,
        model=cfg.GATEWAY_MODEL,
        api_key=cfg.api_key(),
        requests_per_second=cfg.RATE_LIMIT_RPS,
        max_concurrency=cfg.MAX_CONCURRENCY,
        global_budget=cfg.GLOBAL_BUDGET,
        # Bilinçli olarak `is None`: açıkça verilen boş sözlük "hiçbir aşamaya
        # izin yok" demektir, "varsayılana dön" değil.
        stage_budgets=resolved,
        model_dependent_stages=cfg.MODEL_DEPENDENT_STAGES,
        model_slug=active_model_slug,
        max_retries=cfg.MAX_RETRIES,
        circuit_threshold=cfg.CIRCUIT_THRESHOLD,
    )
    client = GatewayClient(
        gateway_config,
        cache_dir=cfg.CACHE_DIR,
        budget_path=cfg.BUDGET_PATH,
        log_path=cfg.CALL_LOG_PATH,
    )
    _DEFAULT_CLIENTS[memo_key] = client
    return client
