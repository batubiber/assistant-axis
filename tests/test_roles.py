import json

import pytest

from aax.roles import (
    ROLE_NAMES,
    build_generation_prompt,
    parse_generation_response,
)
from aax.judge import JudgeParseError


def test_catalog_has_exactly_120_roles():
    assert len(ROLE_NAMES) == 120


def test_role_names_are_unique():
    assert len(set(ROLE_NAMES)) == len(ROLE_NAMES)


def test_role_names_are_lowercase_single_words():
    for name in ROLE_NAMES:
        assert name == name.lower(), name
        assert name.isalpha(), name


def test_catalog_includes_paper_anchor_roles():
    # Makalenin Tablo 1/2 ve Şekil 2'sinde açıkça geçen roller.
    for anchor in ("generalist", "leviathan", "egregore", "bohemian", "consultant"):
        assert anchor in ROLE_NAMES


def test_generation_prompt_mentions_role_and_counts():
    prompt = build_generation_prompt("pirate")
    assert "pirate" in prompt
    assert "3" in prompt and "40" in prompt


def test_parse_generation_response_extracts_fields():
    raw = (
        '```json\n{"description": "a swashbuckling sailor", '
        '"instructions": ["a", "b", "c"], '
        '"questions": ' + str([f"q{i}" for i in range(40)]).replace("'", '"') + "}\n```"
    )
    parsed = parse_generation_response("pirate", raw)
    assert parsed["role"] == "pirate"
    assert parsed["description"] == "a swashbuckling sailor"
    assert len(parsed["instructions"]) == 3
    assert len(parsed["questions"]) == 40


def test_parse_generation_response_rejects_wrong_instruction_count():
    raw = (
        '{"description": "x", "instructions": ["a"], "questions": '
        + str([f"q{i}" for i in range(40)]).replace("'", '"')
        + "}"
    )
    with pytest.raises(JudgeParseError, match="instructions"):
        parse_generation_response("pirate", raw)


def test_parse_generation_response_rejects_wrong_question_count():
    raw = '{"description": "x", "instructions": ["a", "b", "c"], "questions": ["q1"]}'
    with pytest.raises(JudgeParseError, match="questions"):
        parse_generation_response("pirate", raw)


def test_parse_generation_response_rejects_non_dict_top_level():
    raw = "[1, 2, 3]"
    with pytest.raises(JudgeParseError):
        parse_generation_response("pirate", raw)


def test_parse_generation_response_rejects_missing_description():
    raw = json.dumps(
        {
            "instructions": ["a", "b", "c"],
            "questions": [f"q{i}" for i in range(40)],
        }
    )
    with pytest.raises(JudgeParseError, match="description"):
        parse_generation_response("pirate", raw)


def test_parse_generation_response_rejects_empty_description():
    raw = json.dumps(
        {
            "description": "   ",
            "instructions": ["a", "b", "c"],
            "questions": [f"q{i}" for i in range(40)],
        }
    )
    with pytest.raises(JudgeParseError, match="description"):
        parse_generation_response("pirate", raw)


def test_parse_generation_response_rejects_non_string_question_item():
    questions = [f"q{i}" for i in range(40)]
    questions[5] = None
    raw = json.dumps(
        {"description": "x", "instructions": ["a", "b", "c"], "questions": questions}
    )
    with pytest.raises(JudgeParseError, match="questions"):
        parse_generation_response("pirate", raw)


def test_parse_generation_response_rejects_non_string_instruction_item():
    raw = json.dumps(
        {
            "description": "x",
            "instructions": ["a", 42, "c"],
            "questions": [f"q{i}" for i in range(40)],
        }
    )
    with pytest.raises(JudgeParseError, match="instructions"):
        parse_generation_response("pirate", raw)
