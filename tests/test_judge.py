import json
from pathlib import Path
from unittest import mock

import pytest

from aax.judge import (
    ROLE_SCORE_RUBRIC,
    JudgeParseError,
    _drop_trailing_commas,
    extract_json,
    score_role_expression,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_extract_bare_json_array():
    assert extract_json("[1, 2, 3]") == [1, 2, 3]


def test_extract_fenced_json():
    text = "```json\n[0, 3]\n```"
    assert extract_json(text) == [0, 3]


def test_extract_fenced_without_language():
    text = "```\n{\"a\": 1}\n```"
    assert extract_json(text) == {"a": 1}


def test_extract_json_surrounded_by_prose():
    text = "İşte sonuçlar:\n```json\n[2, 2, 1]\n```\nUmarım yardımcı olur."
    assert extract_json(text) == [2, 2, 1]


def test_extract_json_with_leading_prose_no_fence():
    text = "Sonuç: [3, 0]"
    assert extract_json(text) == [3, 0]


def test_extract_raises_on_garbage():
    with pytest.raises(JudgeParseError):
        extract_json("burada hiç json yok")


# --- Sondaki virgül onarımı -------------------------------------------


def test_drop_trailing_commas_before_bracket():
    assert _drop_trailing_commas("[1, 2,]") == "[1, 2]"


def test_drop_trailing_commas_before_brace():
    assert _drop_trailing_commas('{"a": 1,}') == '{"a": 1}'


def test_drop_trailing_commas_multiple_in_one_document():
    text = '{"a": [1, 2,], "b": {"c": 3,},}'
    assert _drop_trailing_commas(text) == '{"a": [1, 2], "b": {"c": 3}}'


def test_drop_trailing_commas_preserves_string_containing_comma_bracket():
    # Bir string değeri harfiyen ",]" veya ",}" içerebilir; scanner bunu
    # bozmamalı.
    text = '["a value with ,] inside", "another with ,} too",]'
    expected = '["a value with ,] inside", "another with ,} too"]'
    assert _drop_trailing_commas(text) == expected


def test_drop_trailing_commas_handles_escaped_quote_before_trailing_comma():
    # Kaçışlı bir tırnak, string'in bittiğini yanlış işaretlememeli — aksi
    # halde tarayıcı senkronunu kaybeder.
    text = r'["she said \"hi,]\" to me",]'
    expected = r'["she said \"hi,]\" to me"]'
    assert _drop_trailing_commas(text) == expected


def test_drop_trailing_commas_no_trailing_commas_unchanged():
    text = '{"a": [1, 2], "b": "no trailing comma here"}'
    assert _drop_trailing_commas(text) == text


def test_extract_json_repairs_trailing_comma_array():
    assert extract_json("[1, 2, 3,]") == [1, 2, 3]


def test_extract_json_repairs_trailing_comma_object():
    assert extract_json('{"a": 1, "b": 2,}') == {"a": 1, "b": 2}


def test_extract_json_repair_does_not_touch_string_with_comma_bracket():
    text = '{"note": "list is a,] weird case", "n": 1,}'
    assert extract_json(text) == {"note": "list is a,] weird case", "n": 1}


def test_extract_json_repair_cannot_save_genuinely_malformed_json():
    with pytest.raises(JudgeParseError):
        extract_json("{not json at all, ]")


def test_extract_json_valid_input_never_reaches_repair():
    # Sıkı bir şekilde ayrıştırılabilen girdi, onarım yoluna hiç girmemeli.
    with mock.patch("aax.judge._drop_trailing_commas") as repair:
        result = extract_json("[1, 2, 3]")
    assert result == [1, 2, 3]
    repair.assert_not_called()


def test_extract_json_real_prophet_fixture_parses_after_repair():
    raw = (FIXTURES_DIR / "prophet_trailing_comma.txt").read_text(encoding="utf-8")
    parsed = extract_json(raw)
    assert isinstance(parsed, dict)
    assert len(parsed["instructions"]) == 3
    assert len(parsed["questions"]) == 40


def test_extract_json_real_prophet_fixture_needs_repair():
    # Belgeler bu fixture'ın sıkı ayrıştırmayı başarısız kıldığını iddia
    # ediyor; bunu doğrula, aksi halde yukarıdaki test onarım yolunu hiç
    # egzersiz etmeyebilir.
    raw = (FIXTURES_DIR / "prophet_trailing_comma.txt").read_text(encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        json.loads(raw.strip())


class StubClient:
    """chat() çağrılarını kaydeden ve sırayla sabit yanıt döndüren sahte istemci."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages, *, stage, temperature=0.0, max_tokens=1024):
        self.calls.append({"messages": messages, "stage": stage})
        return self.responses.pop(0)


def make_items(n):
    return [(f"soru {i}", f"yanit {i}") for i in range(n)]


def test_score_role_expression_returns_one_score_per_item():
    client = StubClient(["[3, 2, 0]"])
    scores = score_role_expression(
        client, role="pirate", description="a swashbuckling sailor",
        items=make_items(3), stage="test",
    )
    assert scores == [3, 2, 0]
    assert len(client.calls) == 1


def test_score_role_expression_batches_by_ten():
    client = StubClient(["[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]", "[2, 2]"])
    scores = score_role_expression(
        client, role="hermit", description="a solitary recluse",
        items=make_items(12), stage="test", batch_size=10,
    )
    assert scores == [1] * 10 + [2, 2]
    assert len(client.calls) == 2, "12 öğe 10'luk batch'lerde 2 çağrı etmeli"


def test_score_role_expression_raises_on_length_mismatch():
    client = StubClient(["[1, 2]"])
    with pytest.raises(JudgeParseError, match="uzunluk"):
        score_role_expression(
            client, role="ghost", description="a restless spirit",
            items=make_items(3), stage="test",
        )


def test_score_role_expression_raises_on_out_of_range_score():
    client = StubClient(["[1, 7]"])
    with pytest.raises(JudgeParseError, match="aralığı"):
        score_role_expression(
            client, role="ghost", description="a restless spirit",
            items=make_items(2), stage="test",
        )


def test_score_role_expression_raises_on_boolean_score():
    # bool is a subclass of int in Python; [true, false] must not be silently
    # accepted as scores 1, 0 — that would be exactly the guessed/coerced
    # score the module's invariant forbids.
    client = StubClient(["[true, false]"])
    with pytest.raises(JudgeParseError, match="aralığı"):
        score_role_expression(
            client, role="ghost", description="a restless spirit",
            items=make_items(2), stage="test",
        )


def test_score_role_expression_raises_on_float_score():
    client = StubClient(["[1.5]"])
    with pytest.raises(JudgeParseError, match="aralığı"):
        score_role_expression(
            client, role="ghost", description="a restless spirit",
            items=make_items(1), stage="test",
        )


def test_prompt_contains_role_and_rubric():
    client = StubClient(["[3]"])
    score_role_expression(
        client, role="leviathan", description="a vast sea creature",
        items=make_items(1), stage="test",
    )
    prompt = client.calls[0]["messages"][-1]["content"]
    assert "leviathan" in prompt
    assert "vast sea creature" in prompt
    assert "soru 0" in prompt and "yanit 0" in prompt
    assert ROLE_SCORE_RUBRIC in prompt
