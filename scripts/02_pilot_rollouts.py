#!/usr/bin/env python3
"""Aşama 0.5 için küçük pilot üretim.

Amaç hakem kapısına yem üretmek: birkaç rolden, elle etiketlenebilecek
sayıda yanıt. Tam üretim (Aşama 1) bu kapı geçilmeden koşulmaz.

Bu script HF transformers kullanır (vLLM değil) — 40 yanıt için vLLM'in
başlatma maliyeti anlamsız.

Kullanım:
    uv run --extra ml python scripts/02_pilot_rollouts.py --roles 9 --questions 5
"""
from __future__ import annotations

import argparse
import json

from aax import config
from aax.model import load_hf_model
from aax.prompts import build_role_specs, load_role_catalog, to_chat_messages
from aax.rollouts import rollout_record

OUT_PATH = config.model_data_dir() / "pilot_rollouts.jsonl"

# `scripts/03_judge_gate.py::MIN_LABELLED` ile AYNI sayı — kapının
# açılabilmesi için gereken asgari elle etiketlenmiş satır. Burada bir UYARI
# eşiği olarak kullanılıyor (Önemli 6): eski varsayılan 8×5=40, kapının
# tabanıyla BİREBİR aynıydı, sıfır boşluk payı yoktu. Tek bir boş yanıt
# (`rollout_record`'ın atladığı) kapıyı `--min-labelled 39` olmadan
# açılamaz hale getiriyordu — üstelik operatör zaten HER ŞEYİ elle
# etiketledikten SONRA. Varsayılan `--roles` 9'a çıkarıldı (9×5=45, 5 satır
# pay); bu sabit de üretilen kayıt sayısı yine de tabanın altında kalırsa
# (fazla boş yanıt) operatörü insan etiketlemesine BAŞLAMADAN önce uyarır.
GATE_MIN_LABELLED = 40


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roles", type=int, default=9)
    parser.add_argument("--questions", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    args = parser.parse_args()

    catalog = load_role_catalog(config.DATA_DIR / "roles.json")[: args.roles]
    questions = json.loads(
        (config.DATA_DIR / "questions.json").read_text(encoding="utf-8")
    )["shared_questions"][: args.questions]

    # Rol başına tek sistem promptu yeter: kapı hakemi ölçüyor, rolü değil.
    trimmed = [{**r, "instructions": r["instructions"][:1]} for r in catalog]
    specs = build_role_specs(trimmed, questions)
    print(f"{len(specs)} pilot rollout üretilecek ({args.roles} rol × {args.questions} soru)")

    bundle = load_hf_model()
    tok, model = bundle.tokenizer, bundle.model

    import torch

    records = []
    empty = 0
    for index, spec in enumerate(specs, start=1):
        text = tok.apply_chat_template(
            to_chat_messages(spec),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=1.0,
                top_p=0.95,
                pad_token_id=tok.eos_token_id,
            )
        answer = tok.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
        # `rollout_record` üzerinden: burada kayıt elle kuruluyordu ve
        # o yolun BOŞ YANIT KORUMASI yoktu (`rollout_record` boş/whitespace
        # yanıtta `ValueError` fırlatır). Boş bir pilot yanıtı hakeme kadar
        # gider, bloklayıcı kapının ~40 slotundan birini yer ve insan
        # etiketleyiciye boş bir satır olarak çıkardı — kapının ölçtüğü uyum
        # oranı da o satıra göre kayardı.
        if not answer.strip():
            empty += 1
            print(f"\r{index}/{len(specs)} — boş yanıt atlandı ({empty})", end="")
            continue
        records.append(rollout_record(spec, answer))
        print(f"\r{index}/{len(specs)}", end="")

    print()
    if empty:
        print(f"UYARI: {empty} boş yanıt atlandı — kapıya {len(records)} örnek gidiyor")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Yazıldı: {OUT_PATH} ({len(records)} kayıt)")
    # Önemli 6: kapının tabanı (scripts/03_judge_gate.py::MIN_LABELLED) bu
    # dosyanın satır sayısına bağlıdır. Boş yanıtlar bu tabanın altına
    # düşürebilir; operatör bunu insan etiketlemesine BAŞLAMADAN önce
    # görmeli, 40 satırı elle doldurduktan SONRA değil.
    if len(records) < GATE_MIN_LABELLED:
        print(
            f"UYARI: yalnızca {len(records)} kayıt üretildi — hakem kapısı "
            f"(scripts/03_judge_gate.py) varsayılan olarak en az {GATE_MIN_LABELLED} ister.\n"
            f"  --score bu tabanın altında çıkış 2 ile durur (--min-labelled ile "
            "düşürülebilir, ama bu kapının ölçüm gücünü azaltır).\n"
            "  --roles ve/veya --questions'ı artırıp tekrar üretin."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
