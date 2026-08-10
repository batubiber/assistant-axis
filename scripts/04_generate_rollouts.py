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

import os

# vLLM'in varsayılan sampler'ı FlashInfer'dır ve ilk kullanımda bir CUDA
# kernel'ini `nvcc` ile JIT-derler. Bu makinenin sistem `gcc`'si 15.2.0;
# CUDA 12.8'in `nvcc`'si host derleyici olarak en fazla gcc 14'ü kabul
# ediyor, `g++-13` kurulu değil ve şifresiz sudo yok — yani araç zincirini
# yama yapmak bir seçenek değil. Sonuç: JIT derlemesi motor başlatmasında
# (`LLM(...)` çağrısında) çöker, tek bir rollout üretilmeden önce.
# `VLLM_USE_FLASHINFER_SAMPLER=0` FlashInfer'i devre dışı bırakıp vLLM'i
# derleme gerektirmeyen yerli PyTorch top-p/top-k sampler'ına düşürür.
# `setdefault` kullanıyoruz: araç zinciri düzeltilmiş bir operatör bu
# değişkeni kendi ortamında export ederek geçersiz kılabilir. Araç zinciri
# düzeltildiğinde (uygun g++ kurulup nvcc uyumlu hale geldiğinde) bu satır
# kaldırılmalı. Vllm import edilmeden ÖNCE çalışması şart — aksi halde
# FlashInfer zaten varsayılan olarak seçilmiş olur.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import argparse
import json
import sys

from aax import config
from aax.prompts import build_default_specs, build_role_specs, load_role_catalog, to_chat_messages
from aax.rollouts import rollout_record, write_rollouts, write_rollouts_meta

OUT_PATH = config.model_data_dir() / "rollouts.jsonl"
# `rollouts.jsonl`'ın künyesi: `limit`/`n`/`run_id`. Aşama 0'ın
# `roles.json` zarfıyla aynı işi görür — bir PİLOT artefaktının kanonik
# sanılmasını yapısal olarak imkânsız kılar (bkz. `aax.rollouts`).
META_PATH = config.model_data_dir() / "rollouts_meta.json"


def stride_sample(specs: list, n: int) -> list:
    """Gruptan `n` öğeyi EŞİT ARALIKLARLA seç — baştan `n` tane değil.

    `role_specs` rol-ana sıralıdır (rol × sistem promptu × soru): tam
    ölçekte her rol 120 ardışık satır kaplar. Bu yüzden `role_specs[:90]`
    doksan satırın TAMAMINI ilk rolden alır — duman testi tek bir sistem
    promptu ailesini sınar ve rol çeşitliliğini hiç görmez. Adım örneklemesi
    aynı `n` sayısını grubun tamamına yayar.

    Determinizm: `i * len(specs) // n` saf tamsayı aritmetiğidir, rastgelelik
    ya da tohum yok — aynı girdi her koşuda aynı seçimi verir. Grubun kendi
    iç sırası korunur (indeksler artan).
    """
    if n <= 0:
        return []
    if n >= len(specs):
        return list(specs)
    return [specs[i * len(specs) // n] for i in range(n)]


def select_specs(
    role_specs: list, default_specs: list, limit: int | None
) -> tuple[list, int, int]:
    """`--limit` uygulanırken role/default oranını koru.

    Basit `(role_specs + default_specs)[:limit]` kırpması yanlış: role
    spec'leri listede default'lardan önce geldiği için küçük bir limit
    (duman testinin `--limit 100`'ü dahil) yalnızca "role" türünü kapsar ve
    `system_prompt=None` olan, yapısal olarak farklı default-Assistant
    durumunu (`to_chat_messages` sistem mesajını tamamen atlar) hiç sınamaz.
    Bunun yerine her iki gruptan tam kümenin oranını koruyacak şekilde
    orantılı örnekleriz.

    Grup İÇİNDE de dilimlemek yerine adım örneklemesi yapılır
    (`stride_sample`): oran düzeltmesi `kind` çeşitliliğini geri getirmişti
    ama ROL çeşitliliğini değil — `role_specs` rol-ana sıralı olduğu için
    `--limit 100`'ün 90 rol satırının hepsi rol 0'dan geliyordu.

    Döner: (seçilen spec'ler, seçilen role sayısı, seçilen default sayısı).
    """
    if limit is None:
        return role_specs + default_specs, len(role_specs), len(default_specs)
    total = len(role_specs) + len(default_specs)
    role_fraction = len(role_specs) / total if total else 0.0
    n_role = min(len(role_specs), round(limit * role_fraction))
    n_default = min(len(default_specs), limit - n_role)
    return (
        stride_sample(role_specs, n_role) + stride_sample(default_specs, n_default),
        n_role,
        n_default,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="toplam N spec (rol/varsayılan oranı korunarak, duman testi)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=160,
        help="rollout başına üretilecek azami yeni token sayısı",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.70,
        help="vLLM'e ayrılacak VRAM oranı (TOPLAM bellek üzerinden, kullanılabilir değil)",
    )
    parser.add_argument(
        "--samples-per-default-prompt",
        type=int,
        default=10,
        help="her nötr (default) sistem promptu × soru kombinasyonu için tekrar sayısı",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    catalog = load_role_catalog(config.DATA_DIR / "roles.json")
    questions = json.loads(
        (config.DATA_DIR / "questions.json").read_text(encoding="utf-8")
    )["shared_questions"]

    role_specs = build_role_specs(catalog, questions)
    default_specs = build_default_specs(
        questions, samples_per_prompt=args.samples_per_default_prompt
    )
    specs, n_role, n_default = select_specs(role_specs, default_specs, args.limit)
    print(f"{len(specs)} rollout üretilecek ({n_role} role, {n_default} default)")

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
    meta = write_rollouts_meta(META_PATH, records, args.limit)
    print(f"Yazıldı: {OUT_PATH} ({len(records)} kayıt, {empty} boş yanıt atlandı)")
    print(f"Yazıldı: {META_PATH} (limit={meta['limit']}, run_id={meta['run_id']})")
    if meta["limit"] is not None:
        # Aşama 0'ın pilot uyarısıyla aynı: dosya kanonik yolda duruyor ama
        # kanonik DEĞİL. Aşağı akış (`05_capture_activations.py`) künyeye
        # bakıp reddedecek; operatörün bunu burada da görmesi gerekir.
        print(
            f"PİLOT KOŞU (--limit={meta['limit']}): {OUT_PATH.name} kanonik rollout "
            "kümesi DEĞİLDİR.\n"
            "  scripts/05_capture_activations.py bunu reddeder (--allow-pilot ile "
            "bilinçli olarak geçilebilir).\n"
            "  Tam koşu için --limit vermeden tekrar çalıştırın."
        )
    if empty > len(specs) * 0.02:
        print(f"UYARI: boş yanıt oranı %{100*empty/len(specs):.1f} — max_model_len'i kontrol et")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
