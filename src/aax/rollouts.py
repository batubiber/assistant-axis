"""Rollout kayıtlarının şeması ve diske yazımı.

Yazım atomik: 16.000 kayıtlık bir dosyanın yarısı diskte kalırsa aşağı akış
sessizce eksik veriyle çalışır.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from aax.prompts import RolloutSpec


def rollout_record(spec: RolloutSpec, answer: str) -> dict:
    if not answer or not answer.strip():
        raise ValueError("boş yanıt kaydedilemez")
    return {
        "kind": spec.kind,
        "role": spec.role,
        "system_prompt": spec.system_prompt,
        "question": spec.question,
        "sample_index": spec.sample_index,
        "answer": answer,
    }


def write_rollouts(path: str | Path, records: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def read_rollouts(path: str | Path) -> list[dict]:
    records = []
    for number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except ValueError as exc:
            raise ValueError(f"{path}: satır {number} bozuk: {exc}") from exc
    return records
