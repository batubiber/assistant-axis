"""Smoke test script'inin karar mantığı testleri.

Ağa çıkmaz: `read_budget()`, `check_cache_hit()`, `check_budget_delta()`,
`check_json_shape()` fonksiyonlarını doğrudan çağırır; `main()`'i de sahte
(fake transport'lu, gerçek `GatewayClient`) bir istemciyle ağsız uçtan uca
dener — `build_default_client` monkeypatch'lenir, gerçek endpoint'e hiç
dokunulmaz. Script dosya adı bir rakamla başladığı için
(`01_smoke_gateway.py`) normal `import` ile içe aktarılamaz; `importlib` ile
dosya yolundan yüklenir (bkz. `tests/test_generate_role_data.py`).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from aax import config
from aax.gateway import (
    BudgetCorrupted,
    BudgetExceeded,
    CircuitOpen,
    GatewayClient,
    GatewayConfig,
    GatewayError,
)
from aax.judge import JudgeParseError

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "01_smoke_gateway.py"
)


def _load_script():
    spec = importlib.util.spec_from_file_location("smoke_gateway", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


sg = _load_script()


# --- read_budget --------------------------------------------------------


def test_read_budget_returns_zero_when_file_missing(tmp_path):
    assert sg.read_budget(tmp_path / "yok.json") == 0


def test_read_budget_sums_all_stage_counts(tmp_path):
    path = tmp_path / "budget.json"
    path.write_text(json.dumps({"smoke": 1, "stage0_roles": 5}), encoding="utf-8")
    assert sg.read_budget(path) == 6


def test_read_budget_zero_when_all_counts_zero(tmp_path):
    path = tmp_path / "budget.json"
    path.write_text(json.dumps({"smoke": 0}), encoding="utf-8")
    assert sg.read_budget(path) == 0


# --- check_cache_hit -------------------------------------------------------


def test_check_cache_hit_true_when_responses_match():
    assert sg.check_cache_hit("ayni yanit", "ayni yanit") is True


def test_check_cache_hit_false_when_responses_differ():
    assert sg.check_cache_hit("birinci", "ikinci") is False


# --- diagnose_budget_delta ---------------------------------------------------


def test_diagnose_budget_delta_ok_when_exactly_one():
    verdict, message = sg.diagnose_budget_delta(
        spent=1, first_call_sends=1, second_call_sends=0
    )
    assert verdict == "ok"
    assert "tam olarak 1" in message


def test_diagnose_budget_delta_fails_when_nothing_was_spent():
    # İlk çağrı da bir önceki koşudan kalma cache'e denk gelmişse bütçe hiç
    # artmaz — bu smoke testinin amacına aykırı, TAMAM sayılmamalı.
    verdict, message = sg.diagnose_budget_delta(
        spent=0, first_call_sends=0, second_call_sends=0
    )
    assert verdict == "fail"
    assert "cache'inden geldi" in message


def test_diagnose_budget_delta_fails_when_second_call_actually_sent():
    """Gerçek cache arızası: ikinci çağrı da istek attı."""
    verdict, message = sg.diagnose_budget_delta(
        spent=2, first_call_sends=1, second_call_sends=1
    )
    assert verdict == "fail"
    assert "cache çalışmıyor" in message


def test_diagnose_budget_delta_does_not_blame_cache_for_retries():
    """Bulgu: ilk çağrı retry ettiyse `spent` 2'dir ama cache ÇALIŞIYOR.

    Eski kod her iki durumda da "cache çalışmıyor" diyordu — projenin
    production'a ilk temasında yanlış teşhis. Ayıran ölçüm ikinci çağrının
    gönderim sayısı, `spent` değil: iki senaryo da `spent == 2` verir.
    """
    verdict, message = sg.diagnose_budget_delta(
        spent=2, first_call_sends=2, second_call_sends=0
    )
    assert verdict == "warn", "retry cache arızası olarak raporlanmamalı"
    assert "cache ÇALIŞIYOR" in message
    assert "1 kez yeniden denendi" in message


def test_diagnose_budget_delta_reports_multiple_retries():
    verdict, message = sg.diagnose_budget_delta(
        spent=3, first_call_sends=3, second_call_sends=0
    )
    assert verdict == "warn"
    assert "2 kez yeniden denendi" in message


def test_diagnose_budget_delta_same_spent_two_different_verdicts():
    """Aynı `spent` değeri, ayırt edici ölçüme göre farklı karar vermeli."""
    retry_verdict, _ = sg.diagnose_budget_delta(2, first_call_sends=2, second_call_sends=0)
    cache_verdict, _ = sg.diagnose_budget_delta(2, first_call_sends=1, second_call_sends=1)
    assert retry_verdict == "warn"
    assert cache_verdict == "fail"


# --- check_json_shape ---------------------------------------------------------


def test_check_json_shape_ok_for_list_of_three():
    verdict, payload = sg.check_json_shape('["factual", "opinion", "factual"]')
    assert verdict == "ok"
    assert payload == ["factual", "opinion", "factual"]


def test_check_json_shape_warn_for_wrong_length_list():
    verdict, payload = sg.check_json_shape('["factual", "opinion"]')
    assert verdict == "warn"
    assert payload == ["factual", "opinion"]


def test_check_json_shape_warn_for_non_list_json():
    verdict, payload = sg.check_json_shape('{"a": 1}')
    assert verdict == "warn"
    assert payload == {"a": 1}


def test_check_json_shape_fail_for_unparseable_text():
    verdict, payload = sg.check_json_shape("bu hicbir sekilde json degil, fence de yok")
    assert verdict == "fail"
    assert isinstance(payload, JudgeParseError)


# --- main() uçtan uca, ağsız (sahte transport'lu gerçek GatewayClient) -------


def _make_client(tmp_path, response_text: str):
    calls: list[dict] = []

    def transport(payload):
        calls.append(payload)
        return 200, {
            "choices": [{"message": {"content": response_text}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    cfg = GatewayConfig(
        base_url="https://example.invalid/Jailbreak",
        model="hakem-llm",
        api_key="test-key",
        stage_budgets={"smoke": 10},
    )
    client = GatewayClient(
        cfg,
        cache_dir=tmp_path / "cache",
        budget_path=tmp_path / "budget.json",
        log_path=tmp_path / "calls.jsonl",
        transport=transport,
    )
    return client, calls


def test_main_returns_zero_and_all_ok_for_well_formed_json(tmp_path, monkeypatch, capsys):
    client, calls = _make_client(tmp_path, '["factual", "opinion", "factual"]')
    monkeypatch.setattr(sg, "build_default_client", lambda: client)
    monkeypatch.setattr(config, "BUDGET_PATH", tmp_path / "budget.json")
    monkeypatch.setattr(config, "CALL_LOG_PATH", tmp_path / "calls.jsonl")

    exit_code = sg.main()

    assert exit_code == 0
    assert len(calls) == 1, "ikinci çağrı cache'ten dönmeli, gerçek istek atmamalı"
    out = capsys.readouterr().out
    assert out.count("TAMAM") == 3
    assert "BAŞARISIZ" not in out
    assert "Toplam gönderilen istek: 1" in out


def test_main_returns_nonzero_when_json_unparseable(tmp_path, monkeypatch, capsys):
    client, calls = _make_client(tmp_path, "bu hicbir sekilde json degil")
    monkeypatch.setattr(sg, "build_default_client", lambda: client)
    monkeypatch.setattr(config, "BUDGET_PATH", tmp_path / "budget.json")
    monkeypatch.setattr(config, "CALL_LOG_PATH", tmp_path / "calls.jsonl")

    exit_code = sg.main()

    assert exit_code != 0
    out = capsys.readouterr().out
    assert "BAŞARISIZ" in out
    assert "hakem-llm İngilizce JSON üretemiyor" in out


def test_main_exit_zero_when_json_shape_only_warns(tmp_path, monkeypatch, capsys):
    # 2 elemanlı liste: JSON ayrıştı ama şekil beklenenden farklı — brief'te
    # bu dal `ok`'u False yapmıyor, yalnızca UYARI basıyor.
    client, calls = _make_client(tmp_path, '["factual", "opinion"]')
    monkeypatch.setattr(sg, "build_default_client", lambda: client)
    monkeypatch.setattr(config, "BUDGET_PATH", tmp_path / "budget.json")
    monkeypatch.setattr(config, "CALL_LOG_PATH", tmp_path / "calls.jsonl")

    exit_code = sg.main()

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "UYARI" in out
    assert "BAŞARISIZ" not in out


def test_main_retry_on_first_call_is_not_reported_as_cache_failure(
    tmp_path, monkeypatch, capsys
):
    """Uçtan uca: ilk çağrı 500 alıp retry ederse teşhis "cache çalışmıyor"
    OLMAMALI — bütçe 2 artar ama cache pekâlâ çalışıyordur."""
    durum = {"n": 0}

    def transport(payload):
        durum["n"] += 1
        if durum["n"] == 1:
            return 500, {"error": "gecici"}
        return 200, {
            "choices": [{"message": {"content": '["factual", "opinion", "factual"]'}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    cfg = GatewayConfig(
        base_url="https://example.invalid/Jailbreak",
        model="hakem-llm",
        api_key="test-key",
        stage_budgets={"smoke": 10},
    )
    client = GatewayClient(
        cfg,
        cache_dir=tmp_path / "cache",
        budget_path=tmp_path / "budget.json",
        log_path=tmp_path / "calls.jsonl",
        transport=transport,
        monotonic=lambda: 0.0,
        sleep=lambda _s: None,
    )
    monkeypatch.setattr(sg, "build_default_client", lambda: client)
    monkeypatch.setattr(config, "BUDGET_PATH", tmp_path / "budget.json")
    monkeypatch.setattr(config, "CALL_LOG_PATH", tmp_path / "calls.jsonl")

    exit_code = sg.main()

    out = capsys.readouterr().out
    assert client.sends_made == 2, "kurulum: ilk çağrı bir kez yeniden denendi"
    assert "cache çalışmıyor" not in out, "retry cache arızası olarak raporlanmamalı"
    assert "cache ÇALIŞIYOR" in out
    assert exit_code == 0, "geçici retry smoke testini düşürmemeli"


# --- main() tanı sarmalayıcısı: ham traceback yok, anlaşılır Türkçe var ------


def test_main_reports_missing_api_key_without_traceback(monkeypatch, capsys):
    """Operatörün en olası ilk hatası: anahtar export edilmemiş."""

    def patlayan():
        raise RuntimeError("APP_KEY_JAILBREAK ortam değişkeni tanımlı değil.")

    monkeypatch.setattr(sg, "build_default_client", patlayan)

    exit_code = sg.main()

    assert exit_code == sg.EXIT_KOSULAMADI
    err = capsys.readouterr().err
    assert "BAŞARISIZ" in err
    assert "APP_KEY_JAILBREAK" in err


@pytest.mark.parametrize(
    "istisna, beklenen",
    [
        (BudgetCorrupted("bütçe dosyası ayrıştırılamadı"), "bütçe dosyası okunamıyor"),
        (BudgetExceeded("Global bütçe doldu: 1500/1500"), "çağrı bütçesi dolu"),
        (CircuitOpen("Devre kesici açık"), "devre kesici açık"),
        (GatewayError("Çağrı başarısız (HTTP 401)"), "gateway çağrısı başarısız"),
    ],
)
def test_main_turns_gateway_exceptions_into_diagnostics(
    monkeypatch, capsys, istisna, beklenen
):
    """Ham traceback yerine anlaşılır tanı + sıfırdan farklı çıkış."""

    class PatlayanIstemci:
        sends_made = 0

        def chat(self, messages, *, stage, temperature=0.0, max_tokens=1024):
            raise istisna

    monkeypatch.setattr(sg, "build_default_client", PatlayanIstemci)

    exit_code = sg.main()

    assert exit_code == sg.EXIT_KOSULAMADI
    err = capsys.readouterr().err
    assert "BAŞARISIZ" in err
    assert beklenen in err
    assert "Traceback" not in err


def test_main_exit_codes_are_distinct():
    """Kabuk pipeline'ı "koşulamadı"yı "kontrol başarısız"tan ayırabilmeli."""
    assert sg.EXIT_OK == 0
    assert sg.EXIT_KONTROL_BASARISIZ == 1
    assert sg.EXIT_KOSULAMADI == 2


def test_main_probe_prompt_matches_brief_verbatim():
    assert sg.PROBE == (
        "Classify each of the following statements as either \"factual\" or "
        "\"opinion\".\n\n"
        "[ITEM 0] Water boils at 100 degrees Celsius at sea level.\n"
        "[ITEM 1] Blue is the most beautiful colour.\n"
        "[ITEM 2] The Earth orbits the Sun.\n\n"
        "Respond with ONLY a JSON array of 3 strings, in order. No other text."
    )


def test_stage_is_smoke():
    assert sg.STAGE == "smoke"
