"""`scripts/07_extract_axis.py` karar mantığı testleri.

Bu script'in ürettiği sayı projenin nihai hükmüdür (A kriteri). Buradaki
testler o hükmü bozan üç yolu kapatır: eksenin YANLIŞ nicelikten kurulması
(ham "fully" satırları, rol vektörleri yerine), sıfır "fully" durumunda
tanımsız veriden sahte bir "GEÇTİ" ve bayat bir `role_expression.json` ile
sessiz kayma.

Model, GPU, ağ yok: tüm veri sentetik, tüm yollar `tmp_path`'e yönlendirilir.
Script dosya adı bir rakamla başladığı için normal `import` ile içe
aktarılamaz; `importlib` ile dosya yolundan yüklenir ve repo kuralı gereği
`sys.modules`'e kaydedilir (bkz. `tests/test_label_and_train_probe.py`).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from aax.axis import contrast_axis

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "07_extract_axis.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("extract_axis", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ea = _load_script()


def test_module_is_registered_in_sys_modules():
    assert sys.modules["extract_axis"] is ea


# --- ortak yardımcılar --------------------------------------------------------


# Testlerin çoğu 2-3 rol vektörüyle çalışıyor; `--min-role-vectors`'ın
# varsayılanı 40 (spec Bölüm 9). Taban BİLİNÇLİ olarak geçiliyor — tabanın
# kendisi ayrıca `test_exits_2_when_role_vector_count_is_below_the_floor` ile
# sınanıyor.
_ARGS = ["--min-role-vectors", "1"]

# `activations_index.json` ile `role_expression.json` aynı koşudan geldiğini
# içerikten türetilen bu kimlikle kanıtlar (07 eşit olmalarını ŞART koşar).
_RUN_ID = "testrun00000001"


def _patch_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(ea.config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(ea, "ROLE_EXPRESSION_PATH", tmp_path / "role_expression.json")
    monkeypatch.setattr(ea, "OUT_DIR", tmp_path / "axis")


def _write_dataset(
    tmp_path,
    *,
    role_spec: list[tuple[str, str, int, float]],
    n_default: int = 6,
    default_value: float = 7.0,
    n_layers: int = 2,
    d_model: int = 3,
    expression_override: dict[str, str] | None = None,
    index_extra: dict | None = None,
    run_id: str | None = _RUN_ID,
    expression_run_id: str | None = _RUN_ID,
):
    """Sentetik aktivasyon + indeks + ifade haritası yaz.

    `role_spec`: (rol, kategori, satır sayısı, taban değer) listesi. Her rolün
    satırları d_model boyutunda `taban değer`e dayalı, katmanlar arası hafifçe
    farklı bir vektör alır — böylece katman başına eksen ayrı ayrı anlamlı olur.
    """
    rows = []
    blocks = []
    for role, category, count, value in role_spec:
        for _ in range(count):
            rows.append({"kind": "role", "role": role, "system_prompt": f"{role} ol"})
        block = np.zeros((count, n_layers, d_model), dtype=np.float32)
        for layer in range(n_layers):
            block[:, layer, :] = [value, value / 2 + layer, -value]
        blocks.append(block)

    default_block = np.zeros((n_default, n_layers, d_model), dtype=np.float32)
    for layer in range(n_layers):
        default_block[:, layer, :] = [default_value, default_value * 3 - layer, 1.0]
    for _ in range(n_default):
        rows.append({"kind": "default", "role": None, "system_prompt": ""})
    blocks.append(default_block)

    acts = np.concatenate(blocks, axis=0)
    np.save(tmp_path / "activations.npy", acts)

    index = {
        "n_rows": int(acts.shape[0]),
        "n_layers": n_layers,
        "d_model": d_model,
        "model": "test/Model-1.7B",
        "run_id": run_id,
        "middle_layer": n_layers // 2,
        "rows": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
    }
    if index_extra:
        index.update(index_extra)
    (tmp_path / "activations_index.json").write_text(
        json.dumps(index, ensure_ascii=False), encoding="utf-8"
    )

    expression: dict[str, str] = {}
    cursor = 0
    for _role, category, count, _value in role_spec:
        for offset in range(count):
            expression[str(cursor + offset)] = category
        cursor += count
    if expression_override is not None:
        expression = expression_override
    (tmp_path / "role_expression.json").write_text(
        json.dumps(
            {"run_id": expression_run_id, "expression": expression}, ensure_ascii=False
        ),
        encoding="utf-8",
    )
    return acts, index, expression


# --- Kritik 1: eksen rol vektörlerinden kurulur, ham satırlardan değil --------


def test_axis_is_built_from_fully_role_vectors_not_raw_rows(tmp_path, monkeypatch):
    """Ham "fully" satırlarını havuzlamak iki hata birden yapardı.

    Burada `az` rolünün yalnızca 4 "fully" satırı var — >=10 kuralıyla elenir,
    ama ham havuzlamada satırları yine ortalamaya karışırdı. `cok` rolünün 30,
    `orta` rolünün 10 satırı var — ham havuzlamada `cok` ortalamayı ele
    geçirirdi, oysa tanım her nitelikli rolün EŞİT katkısını ister.
    Beklenen rol ortalaması: mean(vec_cok, vec_orta).
    """
    _patch_paths(monkeypatch, tmp_path)
    acts, index, _ = _write_dataset(
        tmp_path,
        role_spec=[
            ("cok", "fully", 30, 1.0),
            ("orta", "fully", 10, 5.0),
            ("az", "fully", 4, 100.0),
            ("yalniz_somewhat", "somewhat", 10, 9.0),
        ],
    )

    assert ea.main(_ARGS) in (0, 1)  # karar ne olursa olsun artefakt yazılmalı

    axis = np.load(tmp_path / "axis" / "assistant_axis.npy")
    names = json.loads((tmp_path / "axis" / "role_names.json").read_text(encoding="utf-8"))
    assert "az" not in names  # >=10 kuralıyla elendi

    n_layers, d_model = index["n_layers"], index["d_model"]
    default_mean = acts[-6:].astype(np.float64).mean(axis=0)
    vec_cok = acts[:30].astype(np.float64).mean(axis=0)
    vec_orta = acts[30:40].astype(np.float64).mean(axis=0)
    expected_role_mean = (vec_cok + vec_orta) / 2

    for layer in range(n_layers):
        assert axis[layer] == pytest.approx(
            contrast_axis(default_mean[layer], expected_role_mean[layer]), rel=1e-5
        )

    # ham satır havuzlamasıyla AÇIKÇA farklı olmalı (regresyonun kendisi)
    raw_pool = acts[:44].astype(np.float64).mean(axis=0)
    wrong_axis = contrast_axis(default_mean[0], raw_pool[0])
    assert axis[0] != pytest.approx(wrong_axis, rel=1e-3)

    report = json.loads((tmp_path / "axis" / "criterion_a.json").read_text(encoding="utf-8"))
    assert report["n_role_vectors"] == 3  # cok, orta, yalniz_somewhat
    assert report["n_fully_role_vectors"] == 2  # az elendi, somewhat sayılmaz
    assert report["n_layers"] == n_layers
    assert report["d_model"] == d_model


# --- Kritik 2: sıfır "fully" gürültülü başarısızlık, sahte GEÇTİ değil -------


def test_fails_loudly_when_no_role_vector_is_fully(tmp_path, monkeypatch, capsys):
    """Çalışmanın kendi hipotezine yakın senaryo: 1.7B model bir role hiç
    TAM girmiyorsa `fully` rol vektörü yoktur. Eski kod boş dilimin
    ortalamasından NaN üretip A KRİTERİ: GEÇTİ basardı."""
    _patch_paths(monkeypatch, tmp_path)
    _write_dataset(
        tmp_path,
        role_spec=[
            ("a", "somewhat", 12, 1.0),
            ("b", "somewhat", 12, 4.0),
            ("c", "no", 12, 8.0),
        ],
    )

    assert ea.main(_ARGS) == 2

    captured = capsys.readouterr()
    assert "GEÇTİ" not in captured.out
    assert "BAŞARISIZ" in captured.err
    assert "fully" in captured.err
    assert not (tmp_path / "axis" / "criterion_a.json").exists()


# --- Önemli 1: boş default_idx çıkış 1'e (DÜŞTÜ) değil 2'ye düşmeli ---------


def test_empty_default_idx_exits_2_not_1(tmp_path, monkeypatch, capsys):
    """`fully` tarafındaki boş-dilim NaN'ının İKİZİ: hiç 'default' satırı
    yoksa `acts[default_idx].mean(axis=0)` NaN döner. Düzeltme öncesi kod bu
    NaN'ı korumasız bırakıp `contrast_axis`'e taşıyordu; orada fırlayan
    `ValueError` yakalanmadığı için yorumlayıcı çıkış kodu 1 ile dönüyordu —
    "A KRİTERİ DÜŞTÜ" anlamına gelen kod. Bir çökme asla bilimsel bir sonuç
    olarak kaydedilemez; doğru kod 2'dir (BAŞARISIZ, karar DEĞİL)."""
    _patch_paths(monkeypatch, tmp_path)
    _write_dataset(
        tmp_path,
        role_spec=[("a", "fully", 12, 1.0), ("b", "fully", 12, 4.0)],
        n_default=0,
    )

    assert ea.main(_ARGS) == 2

    captured = capsys.readouterr()
    assert "GEÇTİ" not in captured.out
    assert "DÜŞTÜ" not in captured.out
    assert "BAŞARISIZ" in captured.err
    assert "default" in captured.err
    assert not (tmp_path / "axis" / "criterion_a.json").exists()
    assert not (tmp_path / "axis" / "assistant_axis.npy").exists()


def test_wraps_a_numeric_valueerror_as_exit_2_not_1(tmp_path, monkeypatch, capsys):
    """`contrast_axis`/`cosine`/`n_components_for_variance`'ın fırlattığı HER
    `ValueError` çıkış koduna 1 (DÜŞTÜ) değil 2'ye (BAŞARISIZ) çevrilmeli —
    ör. default ve fully ortalamaları tesadüfen eşitse (sıfır normlu
    kontrast) ya da aktivasyon verisinde başka bir nedenle NaN/inf varsa.
    Doğrudan `contrast_axis`'i patlatarak sarmalayıcının genel olduğunu (yalnızca
    boş-default özel durumuna bağlı olmadığını) doğrular."""
    _patch_paths(monkeypatch, tmp_path)
    _write_dataset(
        tmp_path,
        role_spec=[("a", "fully", 12, 1.0), ("b", "fully", 12, 4.0)],
    )

    def boom(*_args, **_kwargs):
        raise ValueError("simüle edilmiş sayısal hata")

    monkeypatch.setattr(ea, "contrast_axis", boom)

    assert ea.main(_ARGS) == 2

    captured = capsys.readouterr()
    assert "GEÇTİ" not in captured.out
    assert "DÜŞTÜ" not in captured.out
    assert "BAŞARISIZ" in captured.err
    assert "simüle edilmiş sayısal hata" in captured.err
    assert not (tmp_path / "axis" / "criterion_a.json").exists()


def test_no_partial_artifact_when_a_late_numeric_step_raises(tmp_path, monkeypatch, capsys):
    """Düzeltme öncesi `assistant_axis.npy`/`role_vectors.npy`,
    `n_components_for_70pct` hesaplanmadan ÖNCE yazılıyordu — geç bir raise,
    önceki bir koşudan kalma bir `criterion_a.json` yanında yarım bir
    `assistant_axis.npy`/`role_vectors.npy` bırakabilirdi. Artık hiçbir
    değer yazılmadan ÖNCE TÜM sayısal hesap tamamlanmış olmalı: bu testte
    `n_components_for_variance` (bloktaki SON çağrı) patlatılır ve HİÇBİR
    artefaktın diske gitmediği doğrulanır."""
    _patch_paths(monkeypatch, tmp_path)
    _write_dataset(
        tmp_path,
        role_spec=[("a", "fully", 12, 1.0), ("b", "fully", 12, 4.0)],
    )

    def boom(*_args, **_kwargs):
        raise ValueError("simüle edilmiş geç hata")

    monkeypatch.setattr(ea, "n_components_for_variance", boom)

    assert ea.main(_ARGS) == 2

    captured = capsys.readouterr()
    assert "BAŞARISIZ" in captured.err
    for name in ("assistant_axis.npy", "role_vectors.npy", "role_names.json", "criterion_a.json"):
        assert not (tmp_path / "axis" / name).exists()


# --- Bulgu 6: bayat role_expression.json ------------------------------------


def test_fails_when_expression_map_size_does_not_match_role_rows(tmp_path, monkeypatch, capsys):
    """Farklı bir --limit ile üretilmiş eski harita sessizce kısmi hizasızlık
    yaratırdı: eşleşmeyen her satır "no" sayılır."""
    _patch_paths(monkeypatch, tmp_path)
    _write_dataset(
        tmp_path,
        role_spec=[("a", "fully", 12, 1.0), ("b", "fully", 12, 4.0)],
        expression_override={str(i): "fully" for i in range(10)},  # 24 yerine 10
    )

    assert ea.main(_ARGS) == 2

    captured = capsys.readouterr()
    assert "BAŞARISIZ" in captured.err
    assert "10" in captured.err and "24" in captured.err
    assert "GEÇTİ" not in captured.out


def test_fails_when_expression_keys_do_not_cover_role_rows(tmp_path, monkeypatch, capsys):
    """Anahtar sayısı tutsa bile satır numaraları kaymış olabilir."""
    _patch_paths(monkeypatch, tmp_path)
    _write_dataset(
        tmp_path,
        role_spec=[("a", "fully", 12, 1.0), ("b", "fully", 12, 4.0)],
        expression_override={str(i + 100): "fully" for i in range(24)},
    )

    assert ea.main(_ARGS) == 2

    captured = capsys.readouterr()
    assert "BAŞARISIZ" in captured.err
    assert "kapsamıyor" in captured.err


# --- Bulgu 5: n_components_for_70pct doyuma ulaşmamalı -----------------------


def test_n_components_for_70pct_is_computed_against_the_full_spectrum(tmp_path, monkeypatch):
    """60 rol vektörü, 30 boyut, izotropik varyans: %70'e ulaşmak 10'dan fazla
    bileşen ister. Yalnızca ilk 10 orandan hesaplansaydı `searchsorted` doyuma
    ulaşıp hep 11 derdi — "persona uzayı düşük boyutlu" iddiasını yapay olarak
    destekleyen, desteklenemeyecek kadar küçük bir sayı.
    """
    _patch_paths(monkeypatch, tmp_path)
    rng = np.random.default_rng(7)
    n_roles, d_model, n_layers, per_role = 60, 30, 2, 10

    rows, blocks = [], []
    for role_index in range(n_roles):
        center = rng.normal(scale=1.0, size=d_model)
        block = np.zeros((per_role, n_layers, d_model), dtype=np.float32)
        for layer in range(n_layers):
            block[:, layer, :] = center
        blocks.append(block)
        rows += [
            {"kind": "role", "role": f"r{role_index}", "system_prompt": "x"}
        ] * per_role
    default_block = np.full((5, n_layers, d_model), 0.25, dtype=np.float32)
    blocks.append(default_block)
    rows += [{"kind": "default", "role": None, "system_prompt": ""}] * 5

    acts = np.concatenate(blocks, axis=0)
    np.save(tmp_path / "activations.npy", acts)
    (tmp_path / "activations_index.json").write_text(
        json.dumps(
            {
                "n_rows": int(acts.shape[0]),
                "n_layers": n_layers,
                "d_model": d_model,
                "model": "test/Model-1.7B",
                "run_id": _RUN_ID,
                "middle_layer": 1,
                "rows": rows,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "role_expression.json").write_text(
        json.dumps(
            {
                "run_id": _RUN_ID,
                "expression": {str(i): "fully" for i in range(n_roles * per_role)},
            }
        ),
        encoding="utf-8",
    )

    # 60 rol vektörü var: burada taban (varsayılan 40) BİLİNÇLİ olarak
    # geçilmiyor, gerçek koşudaki gibi sağlanıyor.
    assert ea.main([]) in (0, 1)

    report = json.loads((tmp_path / "axis" / "criterion_a.json").read_text(encoding="utf-8"))
    ratios_first10 = np.asarray(report["explained_variance_ratio"])
    assert len(ratios_first10) == 10  # rapor yine ilk 10 bileşeni gösterir
    # eski (doyan) hesap tam olarak 11 derdi:
    assert int(np.searchsorted(np.cumsum(ratios_first10), 0.70) + 1) == 11
    assert np.cumsum(ratios_first10)[-1] < 0.70  # ilk 10 eşiğe hiç ulaşmıyor
    assert report["n_components_for_70pct"] > 11
    # D6: kesilmiş `explained_variance_ratio` ile tam spektruma karşı sayılan
    # `n_components_for_70pct` tek başına bağdaştırılamıyordu. Bu alan köprü:
    # ilk 10 bileşen %70'e ulaşmıyorsa cevabın 10'dan büyük olması ZORUNLU.
    assert report["cumulative_variance_at_10"] == pytest.approx(
        float(np.cumsum(ratios_first10)[-1])
    )
    assert report["cumulative_variance_at_10"] < 0.70


# --- Minor: künye -------------------------------------------------------------


def test_criterion_a_records_provenance_without_a_timestamp(tmp_path, monkeypatch):
    _patch_paths(monkeypatch, tmp_path)
    _write_dataset(
        tmp_path,
        role_spec=[("a", "fully", 12, 1.0), ("b", "fully", 12, 4.0), ("c", "fully", 12, 9.0)],
        run_id="abc123",
        expression_run_id="abc123",
    )

    assert ea.main(_ARGS) in (0, 1)

    report = json.loads((tmp_path / "axis" / "criterion_a.json").read_text(encoding="utf-8"))
    assert report["model"] == "test/Model-1.7B"
    assert report["run_id"] == "abc123"
    assert report["n_layers"] == 2
    assert report["d_model"] == 3
    assert report["middle_layer"] == 1
    # saatten türetilen hiçbir alan yok
    assert not [k for k in report if "time" in k or "date" in k or "stamp" in k]


# --- B1: çökme çıkış 1 (DÜŞTÜ) değil, çıkış 2 (BAŞARISIZ) --------------------
#
# Düzeltme öncesi ÖLÇÜLDÜ: `main()` gövdesinin yalnızca sayısal bölümü
# sarılıydı. `activations.npy` yokken `FileNotFoundError` sarmalayıcının
# DIŞINDA kalıyor, `raise SystemExit(main())` hiç çalışmıyor ve yorumlayıcı
# yakalanmamış istisna için çıkış kodu 1 döndürüyordu — bu projede "A KRİTERİ
# DÜŞTÜ" demek olan kod. Hiçbir tanı da basılmıyordu.


def test_missing_activations_file_exits_2_with_a_diagnostic(tmp_path, monkeypatch, capsys):
    _patch_paths(monkeypatch, tmp_path)
    _write_dataset(tmp_path, role_spec=[("a", "fully", 12, 1.0), ("b", "fully", 12, 4.0)])
    (tmp_path / "activations.npy").unlink()

    assert ea.main(_ARGS) == 2

    captured = capsys.readouterr()
    assert "BAŞARISIZ" in captured.err
    assert "activations.npy" in captured.err
    assert "05_capture_activations.py" in captured.err
    assert "DÜŞTÜ" not in captured.out


def test_missing_activations_index_exits_2_with_a_diagnostic(tmp_path, monkeypatch, capsys):
    _patch_paths(monkeypatch, tmp_path)
    _write_dataset(tmp_path, role_spec=[("a", "fully", 12, 1.0), ("b", "fully", 12, 4.0)])
    (tmp_path / "activations_index.json").unlink()

    assert ea.main(_ARGS) == 2

    captured = capsys.readouterr()
    assert "BAŞARISIZ" in captured.err
    assert "activations_index.json" in captured.err


def test_corrupt_activations_index_exits_2(tmp_path, monkeypatch, capsys):
    _patch_paths(monkeypatch, tmp_path)
    _write_dataset(tmp_path, role_spec=[("a", "fully", 12, 1.0), ("b", "fully", 12, 4.0)])
    (tmp_path / "activations_index.json").write_text("{bozuk json", encoding="utf-8")

    assert ea.main(_ARGS) == 2

    assert "BAŞARISIZ" in capsys.readouterr().err


def test_index_rows_longer_than_activations_exits_2_not_1(tmp_path, monkeypatch, capsys):
    """Ölçülen ikinci çökme yolu: `rows` matristen uzunsa `acts[role_idx]`
    bir `IndexError` fırlatıyordu. `IndexError` bir `ValueError` DEĞİLDİR —
    sayısal bloğun sarmalayıcısına da takılmadan çıkış 1'e düşüyordu."""
    _patch_paths(monkeypatch, tmp_path)
    _, index, _ = _write_dataset(
        tmp_path, role_spec=[("a", "fully", 12, 1.0), ("b", "fully", 12, 4.0)]
    )
    index["rows"] = index["rows"] + [{"kind": "role", "role": "hayalet", "system_prompt": "x"}] * 5
    (tmp_path / "activations_index.json").write_text(
        json.dumps(index, ensure_ascii=False), encoding="utf-8"
    )

    assert ea.main(_ARGS) == 2

    captured = capsys.readouterr()
    assert "BAŞARISIZ" in captured.err
    assert "DÜŞTÜ" not in captured.out


def test_middle_layer_out_of_range_exits_2(tmp_path, monkeypatch, capsys):
    """`middle >= n_layers` de bir `IndexError`'dı ve mevcut `except ValueError`
    sarmalayıcısından KAÇIYORDU."""
    _patch_paths(monkeypatch, tmp_path)
    _, index, _ = _write_dataset(
        tmp_path, role_spec=[("a", "fully", 12, 1.0), ("b", "fully", 12, 4.0)]
    )
    index["middle_layer"] = 99
    (tmp_path / "activations_index.json").write_text(
        json.dumps(index, ensure_ascii=False), encoding="utf-8"
    )

    assert ea.main(_ARGS) == 2

    assert "middle_layer" in capsys.readouterr().err


def test_an_unexpected_exception_becomes_exit_2(tmp_path, monkeypatch, capsys):
    """Sarmalayıcı GENEL olmalı: öngörülmemiş bir istisna türü de 2'ye
    çevrilmeli, asla 1'e düşmemeli."""
    _patch_paths(monkeypatch, tmp_path)
    _write_dataset(tmp_path, role_spec=[("a", "fully", 12, 1.0), ("b", "fully", 12, 4.0)])

    def boom(*_args, **_kwargs):
        raise RuntimeError("öngörülmemiş çökme")

    monkeypatch.setattr(ea, "role_vectors", boom)

    assert ea.main(_ARGS) == 2

    captured = capsys.readouterr()
    assert "BAŞARISIZ" in captured.err
    assert "öngörülmemiş çökme" in captured.err
    assert "DÜŞTÜ" not in captured.out


def test_keyboard_interrupt_is_not_swallowed(tmp_path, monkeypatch):
    """`except Exception`, `BaseException`'ı KAPSAMAMALI: operatörün Ctrl-C'si
    bir 'BAŞARISIZ' tanısına dönüşmemeli."""
    _patch_paths(monkeypatch, tmp_path)
    _write_dataset(tmp_path, role_spec=[("a", "fully", 12, 1.0), ("b", "fully", 12, 4.0)])

    def boom(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(ea, "role_vectors", boom)

    with pytest.raises(KeyboardInterrupt):
        ea.main(_ARGS)


# --- B2: çok az rol vektöründen hüküm çıkmaz ---------------------------------


def test_exits_2_when_role_vector_count_is_below_the_floor(tmp_path, monkeypatch, capsys):
    """Düzeltme öncesi ÖLÇÜLDÜ: 2 rol vektörü ve span dışı bir default ile
    `passed: True`, `cos_magnitude: 0.99995`, çıkış kodu 0. `n` vektörle
    persentil yalnızca `k/n` değerlerini alabildiği için uç desil koşulu
    neredeyse otomatiktir; ölçülen şey veri değil, örneklem büyüklüğüdür.
    Eski davranış yalnızca bir `UYARI:` basıp devam ediyordu."""
    _patch_paths(monkeypatch, tmp_path)
    _write_dataset(
        tmp_path,
        role_spec=[("a", "fully", 12, 1.0), ("b", "fully", 12, 4.0), ("c", "fully", 12, 9.0)],
    )

    assert ea.main([]) == 2  # varsayılan taban 40

    captured = capsys.readouterr()
    assert "BAŞARISIZ" in captured.err
    assert "3 rol vektörü" in captured.err
    assert "40" in captured.err
    assert "--min-role-vectors" in captured.err
    assert "GEÇTİ" not in captured.out
    assert "DÜŞTÜ" not in captured.out
    assert not (tmp_path / "axis" / "criterion_a.json").exists()


def test_min_role_vectors_override_is_respected(tmp_path, monkeypatch):
    """Taban bilinçli olarak düşürülebilmeli — ama yalnızca AÇIKÇA."""
    _patch_paths(monkeypatch, tmp_path)
    _write_dataset(
        tmp_path,
        role_spec=[("a", "fully", 12, 1.0), ("b", "fully", 12, 4.0), ("c", "fully", 12, 9.0)],
    )

    assert ea.main(["--min-role-vectors", "3"]) in (0, 1)
    assert (tmp_path / "axis" / "criterion_a.json").exists()


# --- B3: bütünlük alanları yazılıyordu ama okunmuyordu -----------------------


def test_n_rows_mismatch_between_index_and_matrix_exits_2(tmp_path, monkeypatch, capsys):
    _patch_paths(monkeypatch, tmp_path)
    _, index, _ = _write_dataset(
        tmp_path, role_spec=[("a", "fully", 12, 1.0), ("b", "fully", 12, 4.0)]
    )
    index["n_rows"] = index["n_rows"] + 7
    (tmp_path / "activations_index.json").write_text(
        json.dumps(index, ensure_ascii=False), encoding="utf-8"
    )

    assert ea.main(_ARGS) == 2

    captured = capsys.readouterr()
    assert "BAŞARISIZ" in captured.err
    assert "n_rows" in captured.err


def test_run_id_mismatch_between_index_and_expression_exits_2(tmp_path, monkeypatch, capsys):
    """Sayı ve kapsama kontrollerinin İKİSİNİ de geçen bayatlık senaryosu:
    Aşama 1, aynı satır sayısı ve aynı sırayla ama FARKLI bir rol kümesiyle
    yeniden koşturulmuş. Tek yakalayan şey içerikten türetilen kimlik."""
    _patch_paths(monkeypatch, tmp_path)
    _write_dataset(
        tmp_path,
        role_spec=[("a", "fully", 12, 1.0), ("b", "fully", 12, 4.0)],
        run_id="koşu-A",
        expression_run_id="koşu-B",
    )

    assert ea.main(_ARGS) == 2

    captured = capsys.readouterr()
    assert "BAŞARISIZ" in captured.err
    assert "koşu-A" in captured.err  # her iki kimlik de adlandırılmalı
    assert "koşu-B" in captured.err
    assert "GEÇTİ" not in captured.out


def test_missing_run_id_in_expression_exits_2(tmp_path, monkeypatch, capsys):
    """Kimlik alanı yazmayan eski bir `role_expression.json` sessizce
    geçmemeli — iki `None` birbirine eşit sayılırsa kontrol hiç yoktur."""
    _patch_paths(monkeypatch, tmp_path)
    _write_dataset(
        tmp_path,
        role_spec=[("a", "fully", 12, 1.0), ("b", "fully", 12, 4.0)],
        expression_run_id=None,
    )

    assert ea.main(_ARGS) == 2

    assert "run_id" in capsys.readouterr().err


def test_matching_run_ids_pass_the_integrity_check(tmp_path, monkeypatch):
    """Pozitif kontrol: aynı kimlik taşıyan iki dosya kabul edilmeli."""
    _patch_paths(monkeypatch, tmp_path)
    _write_dataset(
        tmp_path,
        role_spec=[("a", "fully", 12, 1.0), ("b", "fully", 12, 4.0)],
        run_id="aynı",
        expression_run_id="aynı",
    )

    assert ea.main(_ARGS) in (0, 1)
    report = json.loads((tmp_path / "axis" / "criterion_a.json").read_text(encoding="utf-8"))
    assert report["run_id"] == "aynı"
