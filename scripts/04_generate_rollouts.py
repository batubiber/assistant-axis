#!/usr/bin/env python3
"""Aşama 1 — 16.000 rollout'un vLLM ile üretimi.

vLLM burada kullanılıyor çünkü steering yok, sadece metin lazım. Steering'li
ve capping'li her koşu HF transformers kullanır (spec Bölüm 4.1) — makale
vLLM steering'inin tutarlı %2-3 daha kötü ölçtüğünü raporluyor.

VRAM: RTX 4060'ın 8188 MiB'ının ~1215'i masaüstünde. vLLM
gpu_memory_utilization'ı TOPLAM belleğe göre hesaplar, bu yüzden 0.9 gibi bir
değer OOM verir. Varsayılanı 0.70 tuttuk.

Kullanım:
    uv run --extra gen python scripts/04_generate_rollouts.py
    uv run --extra gen python scripts/04_generate_rollouts.py --limit 100  # duman testi
"""
from __future__ import annotations

import argparse
import json

from aax import config
from aax.prompts import build_default_specs, build_role_specs, load_role_catalog, to_chat_messages
from aax.rollouts import rollout_record, write_rollouts

OUT_PATH = config.DATA_DIR / "rollouts.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="ilk N spec (duman testi)")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--samples-per-default-prompt", type=int, default=10)
    args = parser.parse_args()

    catalog = load_role_catalog(config.DATA_DIR / "roles.json")
    questions = json.loads(
        (config.DATA_DIR / "questions.json").read_text(encoding="utf-8")
    )["shared_questions"]

    specs = build_role_specs(catalog, questions) + build_default_specs(
        questions, samples_per_prompt=args.samples_per_default_prompt
    )
    if args.limit is not None:
        specs = specs[: args.limit]
    print(f"{len(specs)} rollout üretilecek")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(config.TARGET_MODEL)
    prompts = [
        tokenizer.apply_chat_template(
            to_chat_messages(spec),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for spec in specs
    ]

    llm = LLM(
        model=config.TARGET_MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=2048,
    )
    sampling = SamplingParams(
        max_tokens=args.max_new_tokens, temperature=1.0, top_p=0.95
    )
    outputs = llm.generate(prompts, sampling)

    records = []
    empty = 0
    for spec, output in zip(specs, outputs):
        answer = output.outputs[0].text.strip()
        if not answer:
            empty += 1
            continue
        records.append(rollout_record(spec, answer))

    write_rollouts(OUT_PATH, records)
    print(f"Yazıldı: {OUT_PATH} ({len(records)} kayıt, {empty} boş yanıt atlandı)")
    if empty > len(specs) * 0.02:
        print(f"UYARI: boş yanıt oranı %{100*empty/len(specs):.1f} — max_model_len'i kontrol et")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
