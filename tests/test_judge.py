import pytest

from aax.judge import (
    ROLE_SCORE_RUBRIC,
    JudgeParseError,
    extract_json,
    score_role_expression,
)


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
