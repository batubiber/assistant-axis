"""hakem-llm gateway istemcisi.

Bu modül dışarı giden TEK HTTP noktasıdır. Hız sınırlama, disk cache'i,
kalıcı bütçe ve devre kesici burada kapalıdır; çağıran taraf hiçbirini bilmez.

Bütçe her HTTP gönderimini sayar (retry'lar dahil) — retry'ların tavanı
gizlice aşamaması için bilinçli olarak muhafazakâr.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

Transport = Callable[[dict], tuple[int, dict]]


class GatewayError(RuntimeError):
    """Retry'lar tükendikten sonra çağrı başarısız."""


class BudgetExceeded(RuntimeError):
    """Aşama veya global çağrı tavanı doldu."""


class CircuitOpen(RuntimeError):
    """Ardışık hatalar nedeniyle koşu durduruldu."""


@dataclass
class GatewayConfig:
    base_url: str
    model: str
    api_key: str
    requests_per_second: float = 1.0
    max_concurrency: int = 2
    global_budget: int = 1500
    stage_budgets: dict[str, int] = field(default_factory=dict)
    max_retries: int = 3
    circuit_threshold: int = 3
    timeout_seconds: float = 120.0


def _httpx_transport(base_url: str, api_key: str, timeout: float) -> Transport:
    import httpx

    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    client = httpx.Client(timeout=timeout)

    def send(payload: dict) -> tuple[int, dict]:
        response = client.post(url, json=payload, headers=headers)
        try:
            body = response.json()
        except ValueError:
            body = {"error": response.text[:500]}
        return response.status_code, body

    return send


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

        self._lock = threading.Lock()
        self._semaphore = threading.Semaphore(config.max_concurrency)
        self._last_send_at: float | None = None
        self._consecutive_failures = 0
        self._circuit_open = False
        self.sends_made = 0

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

    def _read_budget(self) -> dict[str, int]:
        if not self.budget_path.exists():
            return {}
        try:
            return json.loads(self.budget_path.read_text(encoding="utf-8"))
        except ValueError:
            return {}

    def _check_budget(self, stage: str) -> None:
        counts = self._read_budget()
        total = sum(counts.values())
        if total >= self.config.global_budget:
            raise BudgetExceeded(
                f"Global bütçe doldu: {total}/{self.config.global_budget}"
            )
        stage_cap = self.config.stage_budgets.get(stage)
        if stage_cap is not None and counts.get(stage, 0) >= stage_cap:
            raise BudgetExceeded(
                f"'{stage}' aşama bütçesi doldu: {counts.get(stage, 0)}/{stage_cap}"
            )

    def _spend_budget(self, stage: str) -> None:
        counts = self._read_budget()
        counts[stage] = counts.get(stage, 0) + 1
        self.budget_path.write_text(json.dumps(counts, indent=2), encoding="utf-8")

    # --- hız sınırlama --------------------------------------------------

    def _wait_for_slot(self) -> None:
        min_interval = 1.0 / self.config.requests_per_second
        if self._last_send_at is not None:
            elapsed = self._monotonic() - self._last_send_at
            if elapsed < min_interval:
                self._sleep(min_interval - elapsed)
        self._last_send_at = self._monotonic()

    # --- log ------------------------------------------------------------

    def _log(self, entry: dict) -> None:
        entry = {"ts": time.time(), **entry}
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # --- kamuya açık API -------------------------------------------------

    def would_call(
        self, messages: list[dict], *, temperature: float = 0.0, max_tokens: int = 1024
    ) -> bool:
        """Bu çağrı bütçe harcar mıydı? Hiçbir istek atmaz."""
        payload = self._payload(messages, temperature, max_tokens)
        return self._cache_read(self._cache_key(payload)) is None

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

        with self._lock:
            if self._circuit_open:
                raise CircuitOpen(
                    f"Devre kesici açık ({self.config.circuit_threshold} ardışık hata). "
                    "Koşu durduruldu — sunucuyu zorlamıyoruz."
                )
            self._check_budget(stage)

        last_status = None
        last_body: dict = {}

        for attempt in range(self.config.max_retries):
            with self._semaphore:
                with self._lock:
                    self._wait_for_slot()
                    self._spend_budget(stage)
                    self.sends_made += 1
                started = self._monotonic()
                status, body = self._transport(payload)
                latency = self._monotonic() - started

            usage = body.get("usage", {}) if isinstance(body, dict) else {}
            self._log(
                {
                    "stage": stage,
                    "status": status,
                    "cached": False,
                    "attempt": attempt + 1,
                    "latency": latency,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                }
            )

            if status == 200:
                try:
                    content = body["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError) as exc:
                    raise GatewayError(f"Beklenmeyen yanıt şekli: {body}") from exc
                with self._lock:
                    self._consecutive_failures = 0
                self._cache_write(key, content)
                return content

            last_status, last_body = status, body
            if attempt + 1 < self.config.max_retries:
                self._sleep(2.0 ** attempt)

        with self._lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.config.circuit_threshold:
                self._circuit_open = True

        raise GatewayError(f"Çağrı başarısız (HTTP {last_status}): {last_body}")


def build_default_client(stage_budgets: dict[str, int] | None = None) -> GatewayClient:
    """Gerçek endpoint'e bağlı istemci. Anahtarı ortamdan okur."""
    from aax import config as cfg

    gateway_config = GatewayConfig(
        base_url=cfg.GATEWAY_BASE_URL,
        model=cfg.GATEWAY_MODEL,
        api_key=cfg.api_key(),
        requests_per_second=cfg.RATE_LIMIT_RPS,
        max_concurrency=cfg.MAX_CONCURRENCY,
        global_budget=cfg.GLOBAL_BUDGET,
        stage_budgets=stage_budgets or cfg.STAGE_BUDGETS,
        max_retries=cfg.MAX_RETRIES,
        circuit_threshold=cfg.CIRCUIT_THRESHOLD,
    )
    return GatewayClient(
        gateway_config,
        cache_dir=cfg.CACHE_DIR,
        budget_path=cfg.BUDGET_PATH,
        log_path=cfg.CALL_LOG_PATH,
    )
