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

from aax import config
from aax.gateway import GatewayClient, GatewayConfig
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


# --- check_budget_delta ------------------------------------------------------


def test_check_budget_delta_ok_when_exactly_one():
    ok, spent = sg.check_budget_delta(before=4, after=5)
    assert ok is True
    assert spent == 1


def test_check_budget_delta_not_ok_when_zero():
    # İlk çağrı da bir önceki koşudan kalma cache'e denk gelmişse bütçe hiç
    # artmaz — bu smoke testinin amacına aykırı, TAMAM sayılmamalı.
    ok, spent = sg.check_budget_delta(before=4, after=4)
    assert ok is False
    assert spent == 0


def test_check_budget_delta_not_ok_when_two():
    # Cache hiç çalışmadı: ikinci çağrı da gerçek istek attı.
    ok, spent = sg.check_budget_delta(before=4, after=6)
    assert ok is False
    assert spent == 2


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
