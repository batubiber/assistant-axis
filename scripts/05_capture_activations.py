#!/usr/bin/env python3
"""Aşama 1 — üretilmiş rollout'ların aktivasyonlarını yakala.

vLLM süreçten çıkmış olmalı: iki motor aynı anda VRAM'e sığmaz.

Kullanım:
    uv run --extra ml python scripts/05_capture_activations.py
    uv run --extra ml python scripts/05_capture_activations.py --batch-size 4  # OOM olursa
    uv run --extra ml python scripts/05_capture_activations.py --start-row 4800  # kaldığı yerden

ÇIKIŞ KODLARI:
    0  tamamlandı
    2  koşulamadı ya da yarıda kaldı (girdi eksik/pilot/bayat, batch hatası)

Bu geçiş ~16.000 satır ve ~2.000 batch sürer. Üç şey buna göre kurulu:
ilerleme her batch'te BASILIR (sessiz saatler yok), bir batch patlarsa hangi
batch olduğu ve nereden devam edileceği yazılır, ve kısmi sonuç diske
kaydedilir — `--start-row` ile üretim tekrarlanmadan devam edilebilir.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from aax import config
from aax.activations import mean_response_activations
from aax.model import free_vram_mib, load_hf_model
from aax.prompts import RolloutSpec, to_chat_messages
from aax.rollouts import load_rollouts_meta, read_rollouts, rollouts_run_id

ACTS_PATH = config.model_data_dir() / "activations.npy"
INDEX_PATH = config.model_data_dir() / "activations_index.json"
# Kısmi koşunun işareti. `activations_index.json` YALNIZCA geçiş eksiksiz
# bittiğinde yazılır; bu dosya ise "matriste ilk k satır gerçek, gerisi
# sıfır" der ve `--start-row`'un dayanağıdır.
PARTIAL_PATH = config.model_data_dir() / "activations_partial.json"
ROLLOUTS_PATH = config.model_data_dir() / "rollouts.jsonl"
ROLLOUTS_META_PATH = config.model_data_dir() / "rollouts_meta.json"


def compute_run_id(records: list[dict]) -> str:
    """Yakalanan rollout kayıtlarından türetilen koşu kimliği.

    `scripts/00_generate_role_data.py::compute_run_id` ile aynı desen (bkz.
    orada): saatten değil İÇERİKTEN türetilir. Öncesinde
    `activations_index.json` hiç `run_id` yazmıyordu — `07_extract_axis.py`
    `index.get("run_id")` okuyup `criterion_a.json`'a `null` basıyordu ve
    verdict artefaktının kaynak koşuya (bu `rollouts.jsonl` kümesine) geri
    bağlantısı hiç kurulmuyordu.

    Gövde artık `aax.rollouts.rollouts_run_id`'de: aynı kimliği
    `04_generate_rollouts.py` (rollouts_meta.json) ve
    `06_label_and_train_probe.py` (role_expression.json) da yazıyor ve
    `07_extract_axis.py` son ikisinin EŞİT olmasını şart koşuyor. Üç ayrı
    kopya, aralarından birinin sessizce ayrışması demekti.
    """
    return rollouts_run_id(records)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="aktivasyon yakalamada satır başına batch boyutu (OOM olursa düşür)",
    )
    parser.add_argument(
        "--start-row",
        type=int,
        default=0,
        help=(
            "bu satırdan itibaren yakala; öncesi mevcut activations.npy'den okunur "
            "(yarıda kalmış bir koşuyu üretimi tekrarlamadan sürdürmek için)"
        ),
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=250,
        help=(
            "kaç batch'te bir kısmi sonuç diske yazılsın (0 = yalnızca hata anında). "
            "Önemli 4: planlanan ölçekte (16.000 satır, batch 8) 25'in varsayılan olduğu "
            "sürümde bu 200 satırda bir, yani ~80 tam matris yeniden yazımı (~290 GB) "
            "demekti; 250 bunu ~6-7 yazıma indirir"
        ),
    )
    parser.add_argument(
        "--allow-pilot",
        action="store_true",
        help=(
            "rollouts.jsonl bir --limit koşusundan gelse bile devam et "
            "(Aşama 0'ın --allow-partial'ıyla aynı bilinçli geçersiz kılma)"
        ),
    )
    return parser


def _format_seconds(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    """`np.save`'i `aax.rollouts.write_rollouts`'un tempfile+fsync+rename
    deseniyle yaz — düz `np.save(path, array)` bir plain overwrite'tır.

    Önemli 4: planlanan ölçekte (16.000 × 28 × 2048 float32, ~3.67 GB)
    checkpoint bu dosyayı ~80 kez baştan yazıyordu; her yazım yarıda kesilirse
    (OOM, Ctrl-C, disk dolması) `activations.npy` YARIM ve BOZUK kalırdı —
    üstelik `--start-row` ile bir sonraki koşu tam olarak bu dosyayı okumaya
    çalışır (`_load_resume_prefix`). Tempfile + `os.replace` ile hedef dosya
    ya TAMAMEN eski hâlinde ya da TAMAMEN yeni hâlinde kalır, hiçbir zaman
    yarım kalmaz.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".npy.tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, array)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _save_partial(acts: np.ndarray, rows_done: int, run_id: str) -> None:
    """Kısmi matrisi ve ilerleme işaretini yaz.

    `activations_index.json` BİLEREK yazılmaz: o dosya "bu matris eksiksiz
    ve şu satırlara karşılık geliyor" anlamına gelir ve `07_extract_axis.py`
    ona güvenir. Yarım bir matrisin yanında duran eksiksiz görünümlü bir
    indeks, kuyruğu sıfır olan satırları gerçek aktivasyon sanan bir eksen
    hesabı demek olurdu.
    """
    _atomic_save_npy(ACTS_PATH, acts)
    PARTIAL_PATH.write_text(
        json.dumps(
            {"rows_done": int(rows_done), "n_rows": int(acts.shape[0]), "run_id": run_id},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _load_resume_prefix(acts: np.ndarray, start_row: int, run_id: str) -> int:
    """`--start-row` için mevcut matrisin ilk `start_row` satırını yerine koy.

    Dönüş: kopyalanan satır sayısı. Uyuşmazlıkta `ValueError`.

    Önemli 2: `PARTIAL_PATH` artık `--start-row` için ZORUNLU, önceden yalnızca
    "varsa" doğrulanıyordu. Şekil eşleşmesi TEK BAŞINA yeterli değil: aynı
    satır SAYISINA sahip ama FARKLI bir rollout kümesi bu kontrolü sessizce
    geçer. Somut senaryo: önceki tam bir koşu `activations.npy`'yi bırakır
    (başarıyla bittiği için `PARTIAL_PATH` başarı sonunda silinir);
    rollout'lar aynı kayıt sayısıyla yeniden üretilir (`04` tekrar koşulur);
    yeni yakalama geçişi ilk checkpoint'ten ÖNCE OS düzeyinde öldürülür (`kill
    -9`, OOM-killer) — hiçbir `_save_partial` çağrılmaz, `PARTIAL_PATH` hiç
    yazılmaz. Operatör "kaldığı yerden devam" niyetiyle `--start-row N`
    verir: eski kod yalnızca şekle bakıp satır 0..N'i ESKİ (önceki tam)
    koşudan alır, yeni bir `run_id` ile devam eder — `07`'nin künye
    kontrolleri bunu YAKALAYAMAZ çünkü indeks tek bir tutarlı `run_id`
    taşıyacak şekilde yazılır. Marker zorunlu kılınınca bu senaryo baştan
    reddedilir: operatör bilir bilmez baştan koşar, sessizce karışık veri
    üretilmez.
    """
    if not ACTS_PATH.exists():
        raise ValueError(
            f"--start-row {start_row} verildi ama {ACTS_PATH} yok — devam edilecek "
            "kısmi sonuç bulunamadı."
        )
    if not PARTIAL_PATH.exists():
        raise ValueError(
            f"--start-row {start_row} verildi ama {PARTIAL_PATH} yok. Şekil eşleşmesi "
            f"TEK BAŞINA yeterli değil: {ACTS_PATH} önceki TAMAMLANMIŞ bir koşudan kalmış "
            "olabilir (başarı sonunda kısmi işaret silinir) ve aynı satır sayısına sahip "
            "ama FARKLI bir rollout kümesiyle karışabilir. Baştan koşun — üretim (Aşama 1) "
            "tekrarlanmak zorunda değil, yalnızca bu yakalama geçişi."
        )
    marker = json.loads(PARTIAL_PATH.read_text(encoding="utf-8"))
    if marker.get("run_id") != run_id:
        raise ValueError(
            f"--start-row {start_row}: {PARTIAL_PATH} farklı bir koşuyu işaretliyor "
            f"(run_id={marker.get('run_id')!r}, beklenen {run_id!r}). Baştan koşun."
        )
    if marker.get("rows_done", 0) < start_row:
        raise ValueError(
            f"--start-row {start_row}: kısmi dosyada yalnızca "
            f"{marker.get('rows_done')} satır hesaplanmış. En fazla oradan devam "
            "edilebilir."
        )
    existing = np.load(ACTS_PATH, mmap_mode="r")
    if existing.shape != acts.shape:
        raise ValueError(
            f"--start-row {start_row}: mevcut {ACTS_PATH} şekli {tuple(existing.shape)}, "
            f"bu koşunun beklediği {tuple(acts.shape)}. Aradaki rollout kümesi ya da "
            "model değişmiş; baştan koşun."
        )
    acts[:start_row] = existing[:start_row]
    return start_row


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    try:
        records = read_rollouts(ROLLOUTS_PATH)
    except FileNotFoundError:
        print(
            f"BAŞARISIZ: {ROLLOUTS_PATH} yok.\n"
            "  Bu dosya Aşama 1 üretiminin çıktısıdır — önce "
            "scripts/04_generate_rollouts.py çalıştırılmalı.",
            file=sys.stderr,
        )
        return 2
    except ValueError as exc:
        print(f"BAŞARISIZ: {ROLLOUTS_PATH} okunamadı.\n  {exc}", file=sys.stderr)
        return 2

    if not records:
        print(f"BAŞARISIZ: {ROLLOUTS_PATH} boş — yakalanacak rollout yok.", file=sys.stderr)
        return 2

    # Aşama 0'ın `load_role_catalog` deseninin AYNISI: pilot bir artefakt
    # sessizce kanonik sayılmaz. `--limit 100` ile üretilmiş 100 satırlık bir
    # duman testi, tam üretimle aynı yolda (`data/rollouts.jsonl`) durur ve
    # dosyanın kendisinde bunu belli eden hiçbir şey yoktur.
    try:
        meta = load_rollouts_meta(ROLLOUTS_META_PATH, records, allow_pilot=args.allow_pilot)
    except ValueError as exc:
        print(f"BAŞARISIZ: rollout kümesi kanonik değil.\n  {exc}", file=sys.stderr)
        return 2
    if meta.get("limit") is not None:
        print(
            f"UYARI: PİLOT rollout kümesi (--limit={meta['limit']}, {meta['n']} kayıt) "
            "--allow-pilot ile kabul edildi. Bu aktivasyonlar A kriteri için kullanılamaz."
        )

    run_id = compute_run_id(records)
    print(f"{len(records)} rollout okundu (run_id={run_id})")

    if not 0 <= args.start_row < len(records):
        print(
            f"BAŞARISIZ: --start-row {args.start_row} geçersiz — "
            f"0 ile {len(records) - 1} arasında olmalı.",
            file=sys.stderr,
        )
        return 2

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

    acts = np.zeros((len(items), bundle.n_layers, bundle.d_model), dtype=np.float32)

    if args.start_row:
        try:
            _load_resume_prefix(acts, args.start_row, run_id)
        except ValueError as exc:
            print(f"BAŞARISIZ: kaldığı yerden devam edilemedi.\n  {exc}", file=sys.stderr)
            return 2
        print(f"İlk {args.start_row} satır mevcut {ACTS_PATH.name} dosyasından alındı")

    # Diskteki indeks bu andan itibaren YALAN olur: matrisi üzerine yazmaya
    # başlıyoruz. Eksiksiz görünümlü bayat bir indeksin yarım bir matrisin
    # yanında kalması, 07'nin sıfır satırları gerçek aktivasyon sanması
    # demekti — bütünlük kontrolleri (n_rows, run_id) satır sayısı ve rollout
    # kümesi aynı kaldığında bunu YAKALAYAMAZDI.
    if INDEX_PATH.exists():
        INDEX_PATH.unlink()
        print(f"Bayat {INDEX_PATH.name} silindi — geçiş bitince yeniden yazılacak")

    batch_size = max(1, args.batch_size)
    total_batches = -(-(len(items) - args.start_row) // batch_size)
    started = time.monotonic()
    rows_done = args.start_row
    batch_index = 0

    for start in range(args.start_row, len(items), batch_size):
        batch_index += 1
        stop = min(start + batch_size, len(items))
        try:
            acts[start:stop] = mean_response_activations(
                bundle, items[start:stop], batch_size=batch_size
            )
        except Exception as exc:  # noqa: BLE001 — hangi batch'te patladığı operatöre lazım
            print()
            _save_partial(acts, rows_done, run_id)
            print(
                f"BAŞARISIZ: batch {batch_index}/{total_batches} (satır {start}-{stop - 1}) "
                "yakalanamadı.\n"
                f"  Ayrıntı: {type(exc).__name__}: {exc}\n"
                f"  {rows_done} satır hesaplanmıştı ve {ACTS_PATH} içine kaydedildi.\n"
                f"  Devam etmek için: uv run --extra ml python scripts/05_capture_activations.py "
                f"--start-row {rows_done}"
                + (f" --batch-size {args.batch_size}" if args.batch_size != 8 else "")
                + "\n"
                "  OOM ise --batch-size'ı düşürerek devam edin; üretim (Aşama 1) "
                "tekrarlanmak zorunda DEĞİL.",
                file=sys.stderr,
            )
            return 2

        rows_done = stop
        elapsed = time.monotonic() - started
        done_now = rows_done - args.start_row
        remaining = len(items) - rows_done
        eta = (elapsed / done_now) * remaining if done_now else 0.0
        print(
            f"\r  batch {batch_index}/{total_batches} — {rows_done}/{len(items)} satır, "
            f"geçen {_format_seconds(elapsed)}, tahmini kalan {_format_seconds(eta)}",
            end="",
            flush=True,
        )

        if args.checkpoint_every and batch_index % args.checkpoint_every == 0:
            _save_partial(acts, rows_done, run_id)

    print()

    _atomic_save_npy(ACTS_PATH, acts)
    INDEX_PATH.write_text(
        json.dumps(
            {
                "n_rows": int(acts.shape[0]),
                "n_layers": int(acts.shape[1]),
                "d_model": int(acts.shape[2]),
                "model": config.TARGET_MODEL,
                "run_id": run_id,
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
    # Geçiş eksiksiz bitti: kısmi işaret artık yanıltıcı, kalkmalı.
    PARTIAL_PATH.unlink(missing_ok=True)
    print(f"Yazıldı: {ACTS_PATH} {acts.shape} float32 (~{acts.nbytes/1e9:.2f} GB)")
    print(f"Yazıldı: {INDEX_PATH}")
    print(f"Toplam süre: {_format_seconds(time.monotonic() - started)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
