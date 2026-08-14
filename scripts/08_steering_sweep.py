#!/usr/bin/env python3
"""Aşama 4 — steering sweep'i. Gateway'e DOKUNMAZ, sadece yerel üretim.

İki katmanda koşar: orta katman (makalenin seçimi) ve varsayılanın uç
desile girdiği katman (bizim A kriteri bulgumuz). Steering gücü her
katmanın KENDİ ortalama residual normunun oranıdır — L14=137, L19=436,
mutlak ölçek karşılaştırmayı anlamsız kılardı.

Kullanım:
    uv run --extra ml python scripts/08_steering_sweep.py --layers 14 19
    uv run --extra ml python scripts/08_steering_sweep.py --layers 14 --limit-roles 3

Dayanıklılık düzeltmesi (Fix Round 1; bkz.
`.superpowers/sdd/p3-task-4-fix1-brief.md`): operatör bu script'i ~2 saat
GÖZETİMSİZ koşturuyor — 3500 üretimlik bir sweep'in 3400'ünde bir CUDA OOM
ya da geçici cihaz hatası (ikisi de `RuntimeError` alt sınıfı), düzeltme
öncesi hiçbir artefakt yazmadan `main()`'den dışarı çıkıyordu. Bu, tam
olarak `06_label_and_train_probe.py`'de commit 44dd90e ile çözülen sınıfın
aynısı (bkz. o dosyanın "Etiketleme geçişi DAYANIKLIDIR" paragrafı) ama bu
script bu dersi plan metninden (3ddb783) SONRA öğrendiği için ilk sürüme
yansımamıştı. Artık:

  - üretim döngüsü her `PROGRESS_PERIOD` (100) üretimde bir `records`'ı
    `write_sweep` ile diske yazıyor (tam yeniden yazım — 3500 kaydın
    ~2 MB'lık dosyasını 35 kez yeniden yazmak bedava, append modunun
    kısmi-satır riski hiç doğmuyor);
  - tek bir üretim çağrısı beklenmeyen bir `Exception` fırlatırsa (CUDA
    OOM, geçici cihaz hatası) koşu o ana kadar üretilenleri yazıp temiz bir
    Türkçe teşhisle çıkış 2 döner — `KeyboardInterrupt` (operatörün
    Ctrl-C'si) eskisi gibi ele alınmaya devam eder;
  - meta dosyası da (`write_sweep`'in zaten kullandığı tempfile +
    `os.replace` deseniyle) ATOMİK yazılıyor — düz `Path.write_text` bir
    kill/OOM-kill/disk dolması sırasında geçerli bir `.jsonl`'in yanına
    budanmış bir meta bırakabiliyordu;
  - `select_assistant_end_roles` ve `planned_generation_count`'un attığı
    `ValueError`'lar (taban commit cec3483, plan metninden SONRA eklendi)
    artık sarmalı — sarmasız hâlleri traceback + çıkış kodu 1 veriyordu,
    oysa bu projede 1 "kriter değerlendirildi ve düştü" demek, bir kullanım
    hatası (ör. `--n-roles` mevcut rol sayısından büyük) değil;
  - `default` türünde satırı olmayan (ya da sonlu olmayan bir norm üreten)
    bir `activations_index.json` artık model YÜKLENMEDEN önce temiz bir
    Türkçe mesajla reddediliyor — eskiden `nan` basıp GPU'ya modeli yükler,
    ilk `generate_steered` çağrısında sarmalanmamış bir `ValueError` alırdı.

Dayanıklılık düzeltmesi, ikinci tur (Fix Round 2; bkz.
`.superpowers/sdd/p3-task-4-fix2-brief.md`), operatörün BİRAZDAN koşacağı
~1.5 saatlik gözetimsiz koşuyu doğrudan ilgilendiren dört madde:

  - `main()` artık `07_extract_axis.py:609-637`'deki desenle bir tanı
    sarmalayıcısı: gerçek gövde `_run()`'a taşındı, `main()` yalnızca onu
    çağırıp öngörülmemiş bir `Exception`'ı (ör. `write_sweep`/
    `write_json_atomic` tam diskte ENOSPC ile patlarsa) yakalayıp temiz bir
    Türkçe teşhisle çıkış 2 döner — sarmasız hâli çıplak traceback + çıkış
    1 veriyordu, ve bu projede 1 "kriter değerlendirildi ve düştü" demek;
  - meta dosyası artık döngü BAŞLAMADAN ÖNCE bir kez (bu koşunun
    parametreleriyle, `attempted=0`) ve her periyodik `write_sweep`'in
    yanında `complete: false` ile yazılıyor — eskiden yalnızca döngü
    SONUNDA yazıldığı için bir SIGKILL/elektrik kesintisi taze bir kısmi
    `.jsonl`'in yanına ÖNCEKİ koşunun (farklı katman/rol/güç) meta'sını
    bırakabiliyordu;
  - hedef `.jsonl` ilk yazımdan önce zaten var ve boş değilse, üzerine
    yazılmadan önce `steering_sweep.jsonl.prev`'e kopyalanıyor ve stderr'e
    bir UYARI basılıyor — tam resume KAPSAM DIŞI kalmaya devam ediyor, bu
    yalnızca SESSİZ EZMEYİ önlüyor (ör. bitmiş 3500 kayıtlık bir sweep'in
    üstüne 105 kayıtlık bir duman testi çalıştırmak);
  - testler artık steering'in FİİLEN uygulandığını sabitliyor: üretici
    fonksiyona ulaşan güç kümesinin tam `set(STRENGTHS)` olduğunu, her
    çağrının `layer_norm`/`direction`'ının O katmanın (ilk katmanın değil)
    değerleri olduğunu, ve üretilen kayıtların `strength` alanının o kaydı
    üreten çağrının gücüyle aynı olduğunu doğruluyor.

Kontrol yönü desteği (Task 2; bkz. `.superpowers/sdd/p4-task-2-brief.md`),
`results/control_preregistration.json`'daki ön-tescilli kontrol koşusunu
AYNI script'le koşturmak için: `--direction {axis,gaussian,shuffled,rolespan}`
(varsayılan `axis` — davranış BİREBİR bugünküyle aynı kalır), `--seed` (yalnızca
kontrol yönleri için anlamlı), `--variant` (verilirse artefaktlar
`steering_sweep_<AD>(.jsonl/_meta.json)` olur; verilmezse bugünkü adlar
DEĞİŞMEDEN kalır) ve `--strengths` (varsayılan `aax.susceptibility.STRENGTHS`
— kontrol koşusu ön-tescilin 3 gücünü kullanır, 7'sini değil, aksi hâlde
225 yerine 525 hakem çağrısı gerekirdi). `axis` DIŞINDAKİ bir yön `--variant`
OLMADAN reddedilir (çıkış 2) — bir kontrol koşusu, committed Aşama 4
sweep'ini asla sessizce ezmemeli. Kontrol yönü `aax.controls.control_direction`
ile, model YÜKLENMEDEN ÖNCE (layer_norms hesabıyla aynı disiplinde) kurulur;
oradan gelen bir `ValueError` de aynı temiz-mesaj + çıkış 2 desenini izler.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import sys
import tempfile
from datetime import timedelta
from pathlib import Path
from time import monotonic

import numpy as np

from aax import config
from aax.controls import CONTROL_KINDS, control_direction, direction_fingerprint
from aax.model import load_hf_model
from aax.steering import generate_steered, mean_residual_norm
from aax.susceptibility import (
    INTROSPECTIVE_QUESTIONS,
    STRENGTHS,
    select_assistant_end_roles,
)

# İlerleme çıktısı VE artımlı kalıcılık (bkz. modül docstring'i) AYNI
# periyotla hizalı: ~2 saatlik bir koşuda operatör zaten bu periyotta bir
# ilerleme satırı görüyordu, artık aynı anda diske de bir kaydediliyor.
PROGRESS_PERIOD = 100


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


def write_json_atomic(path: str | Path, payload: dict) -> None:
    """`write_sweep`'in tempfile + `os.replace` deseninin JSON-metin hâli.

    Meta dosyası eskiden düz `Path.write_text` ile yazılıyordu; hemen
    yanındaki `write_sweep` çağrısı ise zaten atomikti. O yazım sırasında
    bir kill/OOM-kill/disk dolması, geçerli ve tam bir `steering_sweep.jsonl`
    yanına budanmış bir meta bırakabiliyordu — aşağı akıştaki okuyucunun bunu
    fark etmesinin yolu yoktu. Bu yardımcı ile süreç ya ESKİ ya YENİ tam
    içeriği görür, asla yarım bir JSON değil.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _meta_payload(
    *,
    layers: list[int],
    strengths: list[float],
    roles: list[str],
    layer_norms: dict[int, float],
    axis_run_id,
    planned: int,
    attempted: int,
    produced: int,
    complete: bool,
    direction_kind: str,
    direction_seed: int | None,
    direction_sha256: str,
) -> dict:
    """Meta gövdesi — üç çağrı yerinde (döngüden önce, her periyodik
    `write_sweep`'in yanında, döngü sonunda) TEKRARLANMASIN diye tek bir
    yerde üretilir (bkz. Fix Round 2 brief, M2).

    `strengths` artık sabit `STRENGTHS`'ten değil çağıranın kendi listesinden
    gelir (Task 2: `--strengths`) — varsayılanı hâlâ `STRENGTHS` ama artık
    çağıran belirliyor, burası varsaymıyor. `direction_kind`/`direction_seed`/
    `direction_sha256` de Task 2: hangi yönle koşulduğunu ve (kontrol
    yönlerinde) hangi tohumla, hangi vektöre çözüldüğünü artefakta yazar."""
    return {
        "layers": layers,
        "strengths": list(strengths),
        "n_roles": len(roles),
        "roles": roles,
        "questions": list(INTROSPECTIVE_QUESTIONS),
        "layer_norms": {str(k): v for k, v in layer_norms.items()},
        "axis_run_id": axis_run_id,
        "planned": planned,
        "attempted": attempted,
        "produced": produced,
        "complete": complete,
        "direction_kind": direction_kind,
        "direction_seed": direction_seed,
        "direction_sha256": direction_sha256,
    }


def _archive_existing_sweep(out: str | Path) -> None:
    """`out` yolunda ZATEN bir sweep varsa, ilk yazımdan önce kenara kopyalar.

    Tam resume ÖZELLİKLE KAPSAM DIŞI (bkz. Fix Round 2 brief, M3) — bu
    yalnızca SESSİZ EZMEYİ önler. Örnek: 3400 kayıt biriktirmiş bir koşunun
    ardından `--limit-roles 3` ile bir duman testi tekrar koşulursa, mevcut
    kod bu 3400 kaydı sessizce 105'e indirir; bu yardımcı önce onları
    `.jsonl.prev`'e taşıyıp stderr'e bir UYARI basar.
    """
    out = Path(out)
    if not out.exists() or out.stat().st_size == 0:
        return
    existing = sum(
        1 for line in out.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    prev = out.with_name(out.name + ".prev")
    shutil.copyfile(out, prev)
    print(
        f"UYARI: {out} ZATEN VAR ({existing} kayıt) — {prev} OLARAK KOPYALANDI, "
        "BU KOŞU ÜZERİNE YAZACAK.",
        file=sys.stderr,
    )


def _run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, nargs="+", required=True,
                        help="steering yapılacak katmanlar, ör. --layers 14 19")
    parser.add_argument("--n-roles", type=int, default=50,
                        help="Assistant ucuna en yakın kaç rol (varsayılan 50)")
    parser.add_argument("--limit-roles", type=int, default=None,
                        help="duman testi: yalnızca ilk N rol")
    parser.add_argument("--max-new-tokens", type=int, default=120,
                        help="yanıt başına üretilecek azami token")
    parser.add_argument("--direction", choices=("axis", *CONTROL_KINDS), default="axis",
                        help="steering yönü: axis (Assistant Axis, varsayılan) ya da "
                             "bir kontrol yönü (gaussian/shuffled/rolespan)")
    parser.add_argument("--seed", type=int, default=0,
                        help="kontrol yönü RNG tohumu — yalnızca --direction "
                             "axis DIŞINDAYKEN anlamlı")
    parser.add_argument("--variant", type=str, default=None,
                        help="verilirse artefaktlar steering_sweep_<AD>.jsonl / "
                             "steering_sweep_<AD>_meta.json olarak yazılır; "
                             "verilmezse bugünkü adlar DEĞİŞMEDEN kalır")
    parser.add_argument("--strengths", type=float, nargs="+", default=list(STRENGTHS),
                        help="steering güç ızgarası (varsayılan: "
                             "aax.susceptibility.STRENGTHS)")
    args = parser.parse_args(argv)

    # `axis` DIŞINDAKİ bir yön `--variant` OLMADAN reddedilir — bir kontrol
    # koşusu committed Aşama 4 sweep'ini (`steering_sweep.jsonl`) asla
    # sessizce ezmemeli. Bu kontrol hiçbir Stage 3 artifact'i OKUMADAN,
    # hiçbir dosyaya DOKUNMADAN önce çalışır: en ucuz, en erken kapı.
    if args.direction != "axis" and not args.variant:
        print(
            "BAŞARISIZ: --direction axis dışında bir değer verildi ama "
            "--variant eksik — bir kontrol koşusu mevcut Aşama 4 sweep'ini "
            "asla sessizce ezmemeli.\n"
            "  --variant <ad> ekleyin (ör. --variant gaussian_seed0) ya da "
            "--direction axis kullanın.",
            file=sys.stderr,
        )
        return 2

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
    #
    # `select_assistant_end_roles` (`src/aax/susceptibility.py`) `n < 1`,
    # `n > len(names)` ve isim/vektör uzunluk uyuşmazlığında Türkçe
    # `ValueError` atıyor (taban commit cec3483). Sarmasız bırakılırsa bu
    # traceback + çıkış kodu 1 verir — bu projede 1 "kriter değerlendirildi
    # ve düştü" demek, `--n-roles`'a mevcut rol sayısından büyük bir değer
    # verilmesi gibi bir kullanım hatası değil.
    try:
        roles = select_assistant_end_roles(vectors, names, axis, args.layers[0], args.n_roles)
    except ValueError as exc:
        print(
            f"BAŞARISIZ: rol seçimi kurulamadı.\n  {exc}\n"
            "  --n-roles değerini mevcut rol vektörü sayısına göre küçültün.",
            file=sys.stderr,
        )
        return 2
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

    # `default` türünde hiç satır yoksa `mean_residual_norm` boş bir dilim
    # üzerinde çalışır ve yalnızca bir `RuntimeWarning` ile `nan` döner —
    # fırlatmaz. Düzeltme öncesi bu `nan` kontrolsüz kalıyor, script
    # "L14 ortalama residual normu: nan" basıp DEVAM ediyor, modeli
    # yüklüyor, ve ancak ilk `generate_steered` çağrısında (`aax.steering`
    # `strength ve layer_norm sonlu olmalı` der) sarmalanmamış bir
    # `ValueError` alıyordu. Bu guard model YÜKLENMEDEN önce, ucuzca çalışır.
    default_rows = [i for i, r in enumerate(index["rows"]) if r["kind"] == "default"]
    if not default_rows:
        print(
            "BAŞARISIZ: activations_index.json içinde 'default' türünde hiç satır "
            "yok — steering ölçeği (mean_residual_norm) tanımsız.\n"
            "  Kontrol edin: scripts/04_generate_rollouts.py'nin default rollout'ları "
            "ürettiğini ve scripts/05_capture_activations.py'nin bunları "
            "activations_index.json'a yazdığını.\n"
            "  Model YÜKLENMEDİ.",
            file=sys.stderr,
        )
        return 2
    try:
        layer_norms = {
            L: mean_residual_norm(np.asarray(acts[default_rows[:1000]]), L)
            for L in args.layers
        }
    except ValueError as exc:
        print(
            f"BAŞARISIZ: residual normu hesaplanamadı.\n  {exc}\n"
            "  Model YÜKLENMEDİ.",
            file=sys.stderr,
        )
        return 2
    non_finite = {L: n for L, n in layer_norms.items() if not np.isfinite(n)}
    if non_finite:
        print(
            "BAŞARISIZ: şu katmanlarda ortalama residual normu sonlu değil "
            f"(nan/inf): {non_finite} — steering ölçeği tanımsız.\n"
            "  Girdi (activations.npy / activations_index.json) bozuk olabilir; "
            "scripts/05_capture_activations.py'yi tekrar çalıştırıp yeniden üretin.\n"
            "  Model YÜKLENMEDİ.",
            file=sys.stderr,
        )
        return 2

    # `planned_generation_count` de aynı biçimde sarmasız çağrılıyordu —
    # ör. `--limit-roles 0` rol boyutunu sıfıra indirdiğinde attığı
    # `ValueError` de traceback + çıkış 1 veriyordu.
    try:
        total = planned_generation_count(
            n_layers=len(args.layers), n_strengths=len(args.strengths),
            n_roles=len(role_keys), n_questions=len(INTROSPECTIVE_QUESTIONS),
        )
    except ValueError as exc:
        print(
            f"BAŞARISIZ: üretim planı kurulamadı.\n  {exc}\n"
            "  --limit-roles / --n-roles değerini sıfırdan farklı ve mevcut rol "
            "kümesine sığacak şekilde ayarlayın.",
            file=sys.stderr,
        )
        return 2
    print(f"{total} üretim planlandı "
          f"({len(args.layers)} katman × {len(args.strengths)} güç × "
          f"{len(role_keys)} rol × {len(INTROSPECTIVE_QUESTIONS)} soru)")
    for L, n in layer_norms.items():
        print(f"  L{L} ortalama residual normu: {n:.1f}")

    # Yön kurulumu (Task 2) model YÜKLENMEDEN ÖNCE olur — layer_norms
    # hesabıyla AYNI disiplin: `control_direction`'ın attığı bir `ValueError`
    # GPU'ya bir model yüklemeden önce yakalanmalı. `axis` koşusunda
    # `directions[L]` bugünküyle BİREBİR aynı (`axis[L]`) — davranış değişmez.
    try:
        if args.direction == "axis":
            directions = {L: axis[L] for L in args.layers}
        else:
            # `role_vectors_layer` TÜM rol vektörleri (`vectors`), yalnızca
            # seçili `role_keys` DEĞİL — ön-tescil bunu tüm rol span'i
            # (rank 92) üzerinden tanımlıyor, `select_assistant_end_roles`'ın
            # daha sonra bu span'den seçtiği alt kümeden değil.
            #
            # M1 (Fix Round 1): HER katman kendisinin eksenini ve rol
            # vektörlerini kullanır. Test (`test_steering_sweep.py`)
            # `axis_layer=axis[L]` yerine `axis_layer=axis[args.layers[0]]`
            # mutasyonunu yakalar — tüm katmanları ilk katmanınkisine
            # sabitlemeyi görerek hatayı test seviyesinde davranış farklılığı
            # ile algılar.
            directions = {
                L: control_direction(
                    args.direction,
                    axis_layer=axis[L],
                    role_vectors_layer=vectors[:, L, :],
                    seed=args.seed,
                )
                for L in args.layers
            }
    except ValueError as exc:
        print(
            f"BAŞARISIZ: kontrol yönü kurulamadı.\n  {exc}\n"
            "  Model YÜKLENMEDİ.",
            file=sys.stderr,
        )
        return 2
    direction_seed = None if args.direction == "axis" else args.seed
    # Birden fazla katman verilirse (ör. --layers 14 19) her katmanın kendi
    # yönü var; tek bir parmak izi için katmanlar SİRALANIP istiflenir —
    # aynı tohum + aynı katman KÜMESİ HER ZAMAN aynı sha üretir (katman
    # sırasından bağımsız olarak). M2 (Fix Round 1): Sıralama `--layers 14 19`
    # ile `--layers 19 14`'ü farklı parmak izleri vermelerini sağlar — BİREBİR
    # aynı yönler, farklı sıra, ön-tescilin layer SET tanımıyla uyumlu parmak
    # izi sabitliği.
    direction_sha256 = direction_fingerprint(
        np.stack([directions[L] for L in sorted(args.layers)])
    )

    bundle = load_hf_model()
    records: list[dict] = []
    started = monotonic()
    done = 0
    # `--variant` verilmezse `suffix` boş string olur ve adlar bugünküyle
    # BİREBİR aynı kalır (`steering_sweep.jsonl` / `steering_sweep_meta.json`)
    # — committed Aşama 4 artefaktları hiçbir koşulda değişmez.
    suffix = f"_{args.variant}" if args.variant else ""
    out = D / f"steering_sweep{suffix}.jsonl"
    meta_path = D / f"steering_sweep{suffix}_meta.json"
    # M3: hedef `.jsonl` önceki bir koşudan kalma ve boş değilse, ilk
    # yazımdan önce kenara kopyala — aksi hâlde biraz aşağıdaki ilk periyodik
    # (ya da döngü hiç girmezse döngü sonrası) `write_sweep` onu SESSİZCE
    # ezerdi.
    _archive_existing_sweep(out)
    # M2: meta döngü BAŞLAMADAN ÖNCE bir kez (bu koşunun parametreleriyle,
    # `attempted=0`) yazılır — aksi hâlde meta yalnızca döngü SONUNDA
    # yazıldığı için bir SIGKILL/elektrik kesintisi taze bir kısmi `.jsonl`'in
    # yanına ÖNCEKİ koşunun (farklı katman/rol/güç) meta'sını bırakabilirdi.
    write_json_atomic(meta_path, _meta_payload(
        layers=args.layers, strengths=args.strengths, roles=role_keys,
        layer_norms=layer_norms, axis_run_id=index.get("run_id"), planned=total,
        attempted=0, produced=0, complete=False,
        direction_kind=args.direction, direction_seed=direction_seed,
        direction_sha256=direction_sha256,
    ))
    # Yalnızca üretim çağrısı sırasında fırlayan beklenmeyen bir `Exception`
    # (CUDA OOM, geçici cihaz hatası — ikisi de `RuntimeError` alt sınıfı)
    # burada tutulur; `KeyboardInterrupt` `BaseException`dır ve içteki
    # `except Exception`'ı ATLAYIP dıştaki `except KeyboardInterrupt`'a
    # düşer — operatörün Ctrl-C'si bir "BAŞARISIZ" tanısına dönüşmez.
    crashed: Exception | None = None
    try:
        for layer, strength, role, question in itertools.product(
            args.layers, args.strengths, role_keys, INTROSPECTIVE_QUESTIONS
        ):
            direction = directions[layer]
            # Her katalog rolü üç sistem promptu varyantı taşır; sweep
            # boyunca sabit ilkini kullanmak koşuyu deterministik tutar
            # (varyant başına 3× daha fazla üretim yerine).
            system_prompt = catalog[role]["instructions"][0]
            try:
                answer = generate_steered(
                    bundle,
                    [{"role": "system", "content": system_prompt},
                     {"role": "user", "content": question}],
                    layer=layer, direction=direction, strength=strength,
                    layer_norm=layer_norms[layer],
                    max_new_tokens=args.max_new_tokens,
                )
            except Exception as exc:
                print(
                    f"\nBAŞARISIZ: üretim sırasında beklenmeyen bir hata oluştu "
                    f"({done}/{total} tamamlanmıştı) — o ana kadar üretilenler "
                    "diske yazılacak.\n"
                    f"  {type(exc).__name__}: {exc}\n"
                    "  Olası neden: CUDA bellek yetersizliği "
                    "(torch.cuda.OutOfMemoryError) ya da geçici bir cihaz hatası — "
                    "ikisi de RuntimeError alt sınıfıdır.\n"
                    "  Koşu planlanan sonucu üretemedi; kısmi kayıtlar ve meta "
                    "yine de yazıldı.",
                    file=sys.stderr,
                )
                crashed = exc
                break
            done += 1
            if answer.strip():
                records.append(sweep_record(
                    layer=layer, strength=strength, role=role,
                    question=question, answer=answer))
            if done % PROGRESS_PERIOD == 0:
                # Artımlı kalıcılık: 3500 kaydın ~2 MB'lık dosyasını 35 kez
                # tam yeniden yazmak bedava — bir sonraki çökme/kesinti bu
                # ana kadarki ilerlemeyi kaybettirmez.
                write_sweep(out, records)
                # M2: meta'yı jsonl'in HER periyodik yazımının yanında
                # tazele (`complete: false` ile) — pencereyi (jsonl güncel,
                # meta bayat) döngünün tamamı yerine milisaniyelere indirir.
                write_json_atomic(meta_path, _meta_payload(
                    layers=args.layers, strengths=args.strengths, roles=role_keys,
                    layer_norms=layer_norms, axis_run_id=index.get("run_id"),
                    planned=total, attempted=done, produced=len(records),
                    complete=False, direction_kind=args.direction,
                    direction_seed=direction_seed, direction_sha256=direction_sha256,
                ))
                el = monotonic() - started
                eta = el / done * (total - done)
                print(f"\r  {done}/{total} — geçen {timedelta(seconds=int(el))}, "
                      f"kalan ~{timedelta(seconds=int(eta))}", end="", flush=True)
    except KeyboardInterrupt:
        print("\nKESİLDİ — o ana kadar üretilenler yazılıyor.", file=sys.stderr)

    print()
    write_sweep(out, records)
    # `planned`: dört boyutun (katman × güç × rol × soru) çarpımı — koşu HİÇ
    # kesilmese/çökmese kaç üretim yapılacaktı (`total`, yukarıda).
    # `attempted`: döngünün fiilen kaç kez `generate_steered`'ı TAMAMLADIĞI
    # (`done`) — bir kesinti/çökme bunu `planned`'dan KÜÇÜK bırakır.
    # `produced`: bunlardan kaçının boş OLMAYAN bir yanıtla kayda dönüştüğü
    # (`len(records)`) — `attempted`'tan küçük olması KESİNTİ değil, birkaç
    # üretimin boş yanıt verdiği anlamına gelir. `complete` bu yüzden
    # `attempted == planned`e bakar (`produced == planned`e DEĞİL): hiç
    # kesilmemiş ama birkaç boş yanıt üretmiş tam bir koşu artık
    # `complete: false` görünmez.
    write_json_atomic(meta_path, _meta_payload(
        layers=args.layers, strengths=args.strengths, roles=role_keys,
        layer_norms=layer_norms, axis_run_id=index.get("run_id"), planned=total,
        attempted=done, produced=len(records), complete=done == total,
        direction_kind=args.direction, direction_seed=direction_seed,
        direction_sha256=direction_sha256,
    ))

    print(f"Yazıldı: {out} ({len(records)}/{total} kayıt)")
    if len(records) != total:
        print(f"UYARI: {total - len(records)} üretim boş yanıt verdi ya da koşu kesildi/çöktü.")
    if crashed is not None:
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    """Tanı sarmalayıcısı — çıkış 1'i bu script hiç kullanmaz, ama `_run()`'ın
    İÇİNDE öngörülmemiş bir istisna (ör. `write_sweep`/`write_json_atomic`
    tam diskte ENOSPC ile patlarsa) sarmalanmadan yorumlayıcıdan çıkarsa
    Python traceback basıp çıkış kodu 1 ile döner — ve bu projede 1 "kriter
    değerlendirildi ve düştü" demek (bkz. modül docstring'i, Fix Round 2 /
    M1). Bir I/O hatasının böyle yorumlanması kabul edilemez: aynı sınıftan
    bir hata `06_label_and_train_probe.py`'de commit 44dd90e ile ve
    `07_extract_axis.py:609-637`'de zaten düzeltilmişti; bu script aynı
    dersi bu turda alıyor ve AYNI deseni uyguluyor.

    `_run()`'ın kendi gövdesindeki her adım (rol seçimi, norm hesabı, üretim
    döngüsü) zaten kendi Türkçe teşhisini ve çıkış kodu 2'sini üretiyor —
    buradaki `except Exception` yalnızca ÖNGÖRÜLMEMİŞ bir çökme içindir.
    `KeyboardInterrupt` (operatörün Ctrl-C'si) BİLEREK ayrı tutulur ve
    yeniden fırlatılır: bir "BAŞARISIZ" tanısına dönüşmemeli. Not: tam disk
    durumunda `write_json_atomic` da aynı sebeple düşer — bu sarmalayıcı
    yalnızca çıkış kodunu ve mesajı düzeltir, eksik meta'yı KURTARMAZ; bu
    beklenen ve kapsam dışıdır.
    """
    try:
        return _run(argv)
    except KeyboardInterrupt:
        raise
    except Exception as exc:  # noqa: BLE001 — kasıtlı geniş yakalama, gerekçe docstring'de
        print(
            "BAŞARISIZ: beklenmeyen bir hata yüzünden sweep tamamlanamadı.\n"
            f"  Ayrıntı: {type(exc).__name__}: {exc}\n"
            "  Bu bir kriter kararı DEĞİLDİR (çıkış 1 değil 2): hesaplama "
            "tamamlanmadı, üretilmiş olabilecek kısmi veriler B kriteri "
            "değerlendirmesi için tam sayılmamalı.\n"
            "  Olası neden: disk dolu (ENOSPC), izin hatası ya da beklenmeyen "
            "bir I/O sorunu. Diski/izinleri kontrol edip tekrar koşun.",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
