#!/usr/bin/env python3
"""Aşama 4 — steering sweep'i. Gateway'e DOKUNMAZ, sadece yerel üretim.

İki katmanda koşar: orta katman (makalenin seçimi) ve varsayılanın uç
desile girdiği katman (bizim A kriteri bulgumuz). Steering gücü her
katmanın KENDİ ortalama residual normunun oranıdır — L14=137, L19=436,
mutlak ölçek karşılaştırmayı anlamsız kılardı.

Kullanım:
    uv run --extra ml python scripts/08_steering_sweep.py --layers 14 19
    uv run --extra ml python scripts/08_steering_sweep.py --layers 14 --limit-roles 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import timedelta
from pathlib import Path
from time import monotonic

import numpy as np

from aax import config
from aax.model import load_hf_model
from aax.steering import generate_steered, mean_residual_norm
from aax.susceptibility import (
    INTROSPECTIVE_QUESTIONS,
    STRENGTHS,
    select_assistant_end_roles,
)


def planned_generation_count(
    *, n_layers: int, n_strengths: int, n_roles: int, n_questions: int
) -> int:
    dims = {"katman": n_layers, "güç": n_strengths, "rol": n_roles, "soru": n_questions}
    for ad, v in dims.items():
        if v <= 0:
            raise ValueError(f"{ad} boyutu sıfır veya negatif: {v}")
    return n_layers * n_strengths * n_roles * n_questions


def sweep_record(*, layer: int, strength: float, role: str, question: str, answer: str) -> dict:
    if not answer or not answer.strip():
        raise ValueError("boş yanıt kaydedilemez")
    return {
        "layer": layer,
        "strength": strength,
        "role": role,
        "question": question,
        "answer": answer,
    }


def write_sweep(path: str | Path, records) -> None:
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


def read_sweep(path: str | Path) -> list[dict]:
    out = []
    for number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except ValueError as exc:
            raise ValueError(f"{path}: satır {number} bozuk: {exc}") from exc
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, nargs="+", required=True,
                        help="steering yapılacak katmanlar, ör. --layers 14 19")
    parser.add_argument("--n-roles", type=int, default=50,
                        help="Assistant ucuna en yakın kaç rol (varsayılan 50)")
    parser.add_argument("--limit-roles", type=int, default=None,
                        help="duman testi: yalnızca ilk N rol")
    parser.add_argument("--max-new-tokens", type=int, default=120,
                        help="yanıt başına üretilecek azami token")
    args = parser.parse_args()

    D = config.model_data_dir()
    R = config.model_results_dir() / "axis"
    try:
        axis = np.load(R / "assistant_axis.npy")
        vectors = np.load(R / "role_vectors.npy")
        names = json.loads((R / "role_names.json").read_text(encoding="utf-8"))
        index = json.loads((D / "activations_index.json").read_text(encoding="utf-8"))
        acts = np.load(D / "activations.npy", mmap_mode="r")
    except (FileNotFoundError, ValueError) as exc:
        print(f"BAŞARISIZ: Aşama 3 artifact'leri okunamadı.\n  {exc}\n"
              "  Önce scripts/07_extract_axis.py çalıştırılmalı.", file=sys.stderr)
        return 2

    for layer in args.layers:
        if not 0 <= layer < axis.shape[0]:
            print(f"BAŞARISIZ: katman {layer} aralık dışı (0-{axis.shape[0]-1}).",
                  file=sys.stderr)
            return 2

    # Rol seçimi BİLEREK tek katmana (args.layers[0]) sabitlenir ve TÜM
    # sweep boyunca değişmeden kullanılır — katman başına yeniden seçilmez.
    # Aksi hâlde L14 ve L19 farklı rol kümeleriyle karşılaştırılmış olur ve
    # iki katmanı aynı sweep'te koşmanın amacı (aynı roller üstünde etki
    # kıyası) kaybolur. Seçilen roller aşağıda meta artifact'ine yazıldığı
    # için bu seçim geriye dönük denetlenebilir kalır.
    roles = select_assistant_end_roles(vectors, names, axis, args.layers[0], args.n_roles)
    if args.limit_roles is not None:
        roles = roles[: args.limit_roles]
    # "rol::kategori" biçimindeki adlarda yalnızca rol kısmı sistem promptu araması için kullanılır.
    role_keys = [r.split("::")[0] for r in roles]

    catalog = {
        rec["role"]: rec
        for rec in json.loads(
            (config.DATA_DIR / "roles.json").read_text(encoding="utf-8")
        )["roles"]
    }
    missing = sorted({r for r in role_keys if r not in catalog})
    if missing:
        print(f"BAŞARISIZ: şu roller katalogda yok: {missing[:5]}", file=sys.stderr)
        return 2

    default_rows = [i for i, r in enumerate(index["rows"]) if r["kind"] == "default"]
    layer_norms = {
        L: mean_residual_norm(np.asarray(acts[default_rows[:1000]]), L)
        for L in args.layers
    }

    total = planned_generation_count(
        n_layers=len(args.layers), n_strengths=len(STRENGTHS),
        n_roles=len(role_keys), n_questions=len(INTROSPECTIVE_QUESTIONS),
    )
    print(f"{total} üretim planlandı "
          f"({len(args.layers)} katman × {len(STRENGTHS)} güç × "
          f"{len(role_keys)} rol × {len(INTROSPECTIVE_QUESTIONS)} soru)")
    for L, n in layer_norms.items():
        print(f"  L{L} ortalama residual normu: {n:.1f}")

    bundle = load_hf_model()
    records: list[dict] = []
    started = monotonic()
    done = 0
    try:
        for layer in args.layers:
            direction = axis[layer]
            for strength in STRENGTHS:
                for role in role_keys:
                    # Her katalog rolü üç sistem promptu varyantı taşır;
                    # sweep boyunca sabit ilkini kullanmak koşuyu
                    # deterministik tutar (varyant başına 3× daha fazla
                    # üretim yerine).
                    system_prompt = catalog[role]["instructions"][0]
                    for question in INTROSPECTIVE_QUESTIONS:
                        answer = generate_steered(
                            bundle,
                            [{"role": "system", "content": system_prompt},
                             {"role": "user", "content": question}],
                            layer=layer, direction=direction, strength=strength,
                            layer_norm=layer_norms[layer],
                            max_new_tokens=args.max_new_tokens,
                        )
                        done += 1
                        if answer.strip():
                            records.append(sweep_record(
                                layer=layer, strength=strength, role=role,
                                question=question, answer=answer))
                        if done % 100 == 0:
                            el = monotonic() - started
                            eta = el / done * (total - done)
                            print(f"\r  {done}/{total} — geçen {timedelta(seconds=int(el))}, "
                                  f"kalan ~{timedelta(seconds=int(eta))}", end="", flush=True)
    except KeyboardInterrupt:
        print("\nKESİLDİ — o ana kadar üretilenler yazılıyor.", file=sys.stderr)

    print()
    out = D / "steering_sweep.jsonl"
    write_sweep(out, records)
    (D / "steering_sweep_meta.json").write_text(json.dumps({
        "layers": args.layers,
        "strengths": list(STRENGTHS),
        "n_roles": len(role_keys),
        "roles": role_keys,
        "questions": list(INTROSPECTIVE_QUESTIONS),
        "layer_norms": {str(k): v for k, v in layer_norms.items()},
        "axis_run_id": index.get("run_id"),
        "planned": total,
        "produced": len(records),
        "complete": len(records) == total,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Yazıldı: {out} ({len(records)}/{total} kayıt)")
    if len(records) != total:
        print(f"UYARI: {total - len(records)} üretim boş yanıt verdi ya da koşu kesildi.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
