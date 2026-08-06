"""Rollout kayıtlarının şeması ve diske yazımı.

Yazım atomik: 16.000 kayıtlık bir dosyanın yarısı diskte kalırsa aşağı akış
sessizce eksik veriyle çalışır.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from aax.prompts import RolloutSpec


def rollouts_run_id(records: list[dict]) -> str:
    """Rollout kayıtlarından türetilen koşu kimliği.

    `scripts/00_generate_role_data.py::compute_run_id` ile aynı ilke: saatten
    değil İÇERİKTEN türetilir, aynı içerik her koşuda aynı kimliği verir.

    Burada, `rollouts.py` içinde durmasının nedeni ÜÇ tüketicisinin olması ve
    üçünün de AYNI sayıyı üretmek zorunda olması:

    * `04_generate_rollouts.py`  -> `data/rollouts_meta.json`
    * `05_capture_activations.py`-> `data/activations_index.json`
    * `06_label_and_train_probe.py` -> `data/role_expression.json`

    `07_extract_axis.py` son ikisinin eşit olmasını ŞART koşar. Aynı kimliği
    üç ayrı yerde elle kopyalanmış bir ifadeyle hesaplamak, aralarından
    birinin sessizce ayrışması demekti.

    Blob, satır sırasıyla `kind`/`role`/`system_prompt`/`question` alanlarını
    birleştirir: aynı rollout kümesi (aynı sıra, aynı içerik) her zaman aynı
    kimliği üretir; roller, sistem promptları ya da sorular değişirse kimlik
    de değişir. `answer` bilerek DIŞARIDA: kimlik "hangi rollout kümesi"
    sorusunu yanıtlar, "model o gün ne üretti" sorusunu değil.
    """
    blob = "\n".join(
        f"{r['kind']}\t{r['role']}\t{r['system_prompt']}\t{r['question']}" for r in records
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


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


def rollouts_meta_payload(records: list[dict], limit: int | None) -> dict:
    """`data/rollouts_meta.json` içeriği — `rollouts.jsonl`'ın yanındaki künye.

    Neden ayrı bir dosya: `rollouts.jsonl` satır-başına-kayıt bir formattır,
    zarf alanı taşıyacak yeri yok. Ama Aşama 0 bu problemi zaten çözmüş
    durumda (`roles.json` içindeki `complete`/`limit` ve
    `prompts.load_role_catalog`'un pilot artefaktı sert reddi) ve aynı
    tehlike Aşama 1 için de geçerliydi: `--limit 100` ile üretilen bir duman
    testi, KANONİK yola (`data/rollouts.jsonl`) hiçbir işaret bırakmadan
    yazıyordu. Tam üretimin ardından koşulan bir duman testi ~25 dakikalık
    GPU çıktısını sessizce siliyor ve geriye ondan ayırt edilemez bir dosya
    bırakıyordu.

    Alanlar:
      * `n`      — kayıt sayısı (boş yanıtlar atıldıktan SONRA)
      * `limit`  — `--limit` değeri; `None` ise tam koşu, değilse PİLOT
      * `run_id` — içerikten türetilen kimlik (`rollouts_run_id`)
    """
    return {"n": len(records), "limit": limit, "run_id": rollouts_run_id(records)}


def write_rollouts_meta(path: str | Path, records: list[dict], limit: int | None) -> dict:
    path = Path(path)
    payload = rollouts_meta_payload(records, limit)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def load_rollouts_meta(
    path: str | Path, records: list[dict], *, allow_pilot: bool = False
) -> dict:
    """`rollouts_meta.json`'ı yükle ve `records`'u gerçekten tarif ettiğini doğrula.

    `prompts.load_role_catalog` ile aynı fail-closed desen: künye yoksa,
    kayıtlarla uyuşmuyorsa ya da bir PİLOT artefaktı tarif ediyorsa
    `ValueError`. Sessizce kabul etmek, tüm aşağı akış ölçümünü 100 satırlık
    bir duman testi üzerinde yapmak demektir — ve `activations.npy` bunu
    hiçbir yerde belli etmez.
    """
    path = Path(path)
    if not path.exists():
        raise ValueError(
            f"{path} yok — rollouts.jsonl'ın kanonik mi pilot mu olduğu bilinemiyor. "
            "scripts/04_generate_rollouts.py bu künyeyi rollouts.jsonl ile birlikte "
            "yazar; dosya yoksa rollout'lar künye yazmayan eski bir sürümden kalmıştır "
            "ve pilot olup olmadıkları belirlenemez."
        )
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError(f"{path} bozuk — JSON ayrıştırılamadı: {exc}") from exc

    expected_id = rollouts_run_id(records)
    if meta.get("n") != len(records) or meta.get("run_id") != expected_id:
        raise ValueError(
            f"{path} yanındaki rollouts.jsonl'ı tarif etmiyor — "
            f"künye n={meta.get('n')}, run_id={meta.get('run_id')!r}; "
            f"dosyada {len(records)} kayıt, run_id={expected_id!r}. "
            "İki dosya farklı koşulardan geliyor; "
            "scripts/04_generate_rollouts.py'yi tekrar çalıştırıp ikisini birlikte "
            "yeniden üretin."
        )
    if meta.get("limit") is not None and not allow_pilot:
        raise ValueError(
            f"{path}: limit={meta['limit']} — bu bir PİLOT artefaktı ({meta['n']} kayıt), "
            "kanonik rollout kümesi değil. Aşama 1'i --limit OLMADAN tekrar çalıştırın; "
            "pilot veriyle bilinçli olarak devam etmek için --allow-pilot verin."
        )
    return meta


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
