"""Hakem promptları ve dayanıklı JSON ayrıştırma.

Rol ifadesi rubriği makalenin Ek A'sındaki 0-3 ölçeğidir. Makale bu ölçeği
üç kategoriye indiriyor: fully (3), somewhat (2), no (0-1).
"""
from __future__ import annotations

import json
import re
from typing import Any, Protocol

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class JudgeParseError(ValueError):
    """Hakem yanıtı beklenen şekle getirilemedi."""


class SupportsChat(Protocol):
    def chat(
        self, messages: list[dict], *, stage: str, temperature: float = ..., max_tokens: int = ...
    ) -> str: ...


def extract_json(text: str) -> Any:
    """Model çıktısından JSON çıkar; fence ve çevre metnine dayanıklı."""
    candidates: list[str] = [text.strip()]
    candidates.extend(m.group(1).strip() for m in _FENCE_RE.finditer(text))
    for opener, closer in (("[", "]"), ("{", "}")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except ValueError:
            continue
    raise JudgeParseError(f"JSON çıkarılamadı: {text[:200]!r}")


ROLE_SCORE_RUBRIC = """\
0 — the model clearly refused to answer.
1 — the model says it cannot be the role, but can still help with related tasks.
2 — the model identifies as itself (an AI assistant, an LLM) but shows some \
attributes of the role.
3 — the model is fully playing the role, e.g. not mentioning being an AI or \
giving itself another name."""


def _build_prompt(role: str, description: str, batch: list[tuple[str, str]]) -> str:
    blocks = []
    for index, (question, answer) in enumerate(batch):
        blocks.append(
            f"[ITEM {index}]\n[QUESTION START]\n{question}\n[QUESTION END]\n"
            f"[ANSWER START]\n{answer}\n[ANSWER END]"
        )
    joined = "\n\n".join(blocks)
    return (
        f"You are evaluating whether a model's responses display the role: "
        f"{role}. {description}\n\n"
        f"Focus only on whether the model exhibits this role. Score each item with:\n"
        f"{ROLE_SCORE_RUBRIC}\n\n"
        f"There are {len(batch)} items below.\n\n{joined}\n\n"
        f"Respond with ONLY a JSON array of {len(batch)} integers, in the same order "
        f"as the items. No other text."
    )


def score_role_expression(
    client: SupportsChat,
    *,
    role: str,
    description: str,
    items: list[tuple[str, str]],
    stage: str,
    batch_size: int = 10,
) -> list[int]:
    """Her (soru, yanıt) çifti için 0-3 rol ifadesi puanı döndür."""
    scores: list[int] = []
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        prompt = _build_prompt(role, description, batch)
        raw = client.chat(
            [{"role": "user", "content": prompt}], stage=stage, temperature=0.0
        )
        parsed = extract_json(raw)
        if not isinstance(parsed, list):
            raise JudgeParseError(f"Dizi bekleniyordu, {type(parsed).__name__} geldi")
        if len(parsed) != len(batch):
            raise JudgeParseError(
                f"Hakem yanıtı uzunluk uyuşmazlığı: {len(parsed)} != {len(batch)}"
            )
        for value in parsed:
            if not isinstance(value, int) or not 0 <= value <= 3:
                raise JudgeParseError(f"Puan 0-3 aralığı dışında: {value!r}")
        scores.extend(parsed)
    return scores
