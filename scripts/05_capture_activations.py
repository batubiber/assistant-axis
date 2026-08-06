#!/usr/bin/env python3
"""Aşama 1 — üretilmiş rollout'ların aktivasyonlarını yakala.

vLLM süreçten çıkmış olmalı: iki motor aynı anda VRAM'e sığmaz.

Kullanım:
    uv run --extra ml python scripts/05_capture_activations.py
    uv run --extra ml python scripts/05_capture_activations.py --batch-size 4  # OOM olursa
"""
from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np

from aax import config
from aax.activations import mean_response_activations
from aax.model import free_vram_mib, load_hf_model
from aax.prompts import RolloutSpec, to_chat_messages
from aax.rollouts import read_rollouts

ACTS_PATH = config.DATA_DIR / "activations.npy"
INDEX_PATH = config.DATA_DIR / "activations_index.json"


def compute_run_id(records: list[dict]) -> str:
    """Yakalanan rollout kayıtlarından türetilen koşu kimliği.

    `scripts/00_generate_role_data.py::compute_run_id` ile aynı desen (bkz.
    orada): saatten değil İÇERİKTEN türetilir. Öncesinde
    `activations_index.json` hiç `run_id` yazmıyordu — `07_extract_axis.py`
    `index.get("run_id")` okuyup `criterion_a.json`'a `null` basıyordu ve
    verdict artefaktının kaynak koşuya (bu `rollouts.jsonl` kümesine) geri
    bağlantısı hiç kurulmuyordu.

    Blob, satır sırasıyla `kind`/`role`/`system_prompt`/`question` alanlarını
    birleştirir: aynı rollout kümesi (aynı sıra, aynı içerik) her zaman aynı
    kimliği üretir; roller, sistem promptları ya da sorular değişirse kimlik
    de değişir.
    """
    blob = "\n".join(
        f"{r['kind']}\t{r['role']}\t{r['system_prompt']}\t{r['question']}" for r in records
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="aktivasyon yakalamada satır başına batch boyutu (OOM olursa düşür)",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    records = read_rollouts(config.DATA_DIR / "rollouts.jsonl")
    print(f"{len(records)} rollout okundu")

    bundle = load_hf_model()
    print(f"model: {bundle.n_layers} katman, {bundle.d_model} genişlik, boş VRAM {free_vram_mib()} MiB")
    tok = bundle.tokenizer

    items = []
    for record in records:
        spec = RolloutSpec(
            kind=record["kind"],
            role=record["role"],
            system_prompt=record["system_prompt"],
            question=record["question"],
            sample_index=record["sample_index"],
        )
        prompt_text = tok.apply_chat_template(
            to_chat_messages(spec),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
        answer_ids = tok(record["answer"], add_special_tokens=False)["input_ids"]
        items.append((prompt_ids, answer_ids))

    acts = mean_response_activations(bundle, items, batch_size=args.batch_size)

    np.save(ACTS_PATH, acts)
    INDEX_PATH.write_text(
        json.dumps(
            {
                "n_rows": int(acts.shape[0]),
                "n_layers": int(acts.shape[1]),
                "d_model": int(acts.shape[2]),
                "model": config.TARGET_MODEL,
                "run_id": compute_run_id(records),
                "middle_layer": bundle.middle_layer,
                "rows": [
                    {"kind": r["kind"], "role": r["role"], "system_prompt": r["system_prompt"]}
                    for r in records
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Yazıldı: {ACTS_PATH} {acts.shape} float32 (~{acts.nbytes/1e9:.2f} GB)")
    print(f"Yazıldı: {INDEX_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
