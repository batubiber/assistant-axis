import pytest

from aax.judge import JudgeParseError
from aax.persona_judge import (
    NON_ASSISTANT_PERSONA,
    PERSONA_CATEGORIES,
    classify_personas,
)


class StubClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages, *, stage, temperature=0.0, max_tokens=1024):
        self.calls.append({"messages": messages, "stage": stage})
        return self.responses.pop(0)


def items(n):
    return [(f"soru {i}", f"yanit {i}") for i in range(n)]


def test_categories_match_the_paper_rubric():
    assert PERSONA_CATEGORIES == (
        "assistant", "human_role", "nonhuman_role", "weird_role",
        "ambiguous", "other", "nonsensical",
    )


def test_non_assistant_set_is_the_three_role_categories():
    assert NON_ASSISTANT_PERSONA == {"human_role", "nonhuman_role", "weird_role"}
    assert NON_ASSISTANT_PERSONA < set(PERSONA_CATEGORIES)


def test_classify_returns_one_label_per_item_in_order():
    client = StubClient(['["assistant", "human_role", "weird_role"]'])
    out = classify_personas(client, items(3), stage="stage4_steering")
    assert out == ["assistant", "human_role", "weird_role"]


def test_classify_batches_by_ten():
    ten = '["assistant","assistant","assistant","assistant","assistant",' \
          '"assistant","assistant","assistant","assistant","assistant"]'
    client = StubClient([ten, '["other", "nonsensical"]'])
    out = classify_personas(client, items(12), stage="stage4_steering", batch_size=10)
    assert out == ["assistant"] * 10 + ["other", "nonsensical"]
    assert len(client.calls) == 2


def test_classify_rejects_length_mismatch():
    client = StubClient(['["assistant", "human_role"]'])
    with pytest.raises(JudgeParseError, match="uzunluk"):
        classify_personas(client, items(3), stage="stage4_steering")


def test_classify_rejects_unknown_category():
    client = StubClient(['["assistant", "pirate"]'])
    with pytest.raises(JudgeParseError, match="kategori"):
        classify_personas(client, items(2), stage="stage4_steering")


def test_classify_rejects_non_string_label():
    client = StubClient(['["assistant", 3]'])
    with pytest.raises(JudgeParseError, match="kategori"):
        classify_personas(client, items(2), stage="stage4_steering")


def test_classify_accepts_fenced_json():
    client = StubClient(['```json\n["assistant", "other"]\n```'])
    assert classify_personas(client, items(2), stage="stage4_steering") == [
        "assistant", "other",
    ]


def test_prompt_lists_every_category_and_the_items():
    client = StubClient(['["assistant"]'])
    classify_personas(client, items(1), stage="stage4_steering")
    prompt = client.calls[0]["messages"][-1]["content"]
    for c in PERSONA_CATEGORIES:
        assert c in prompt
    assert "soru 0" in prompt and "yanit 0" in prompt
