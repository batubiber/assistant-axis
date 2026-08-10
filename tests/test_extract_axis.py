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
    monkeypatch.setattr(ea, "ACTS_PATH", tmp_path / "activations.npy")
    monkeypatch.setattr(ea, "INDEX_PATH", tmp_path / "activations_index.json")
    monkeypatch.setattr(ea, "OUT_DIR", tmp_path / "axis")


def _write_dataset(
    tmp_path,
    *,
    role_spec: list[tuple[str, str | None, int, float]],
    n_default: int = 6,
    default_value: float = 7.0,
    n_layers: int = 2,
    d_model: int = 3,
    expression_override: dict[str, str] | None = None,
    index_extra: dict | None = None,
    run_id: str | None = _RUN_ID,
    expression_run_id: str | None = _RUN_ID,
    dropped_roles_override: list[str] | None = None,
    method: str | None = None,
    n_roles_dropped: int | None = None,
    probe_holdout_agreement: float | None = None,
):
    """Sentetik aktivasyon + indeks + ifade haritası yaz.

    `role_spec`: (rol, kategori, satır sayısı, taban değer) listesi. Her rolün
    satırları d_model boyutunda `taban değer`e dayalı, katmanlar arası hafifçe
    farklı bir vektör alır — böylece katman başına eksen ayrı ayrı anlamlı olur.

    `kategori` `None` olabilir: bu, `--role-level-fallback`'in ATTIĞI bir rolü
    modeller — indeks satırları YAZILIR (kind="role") ama `expression`'da o
    rolün HİÇBİR satırı görünmez, rol adı otomatik olarak yazılan
    `dropped_roles` listesine eklenir (bkz. `run_role_level_fallback()`).
    """
    rows = []
    blocks = []
    dropped_roles_auto: list[str] = []
    for role, category, count, value in role_spec:
        for _ in range(count):
            rows.append({"kind": "role", "role": role, "system_prompt": f"{role} ol"})
        block = np.zeros((count, n_layers, d_model), dtype=np.float32)
        for layer in range(n_layers):
            block[:, layer, :] = [value, value / 2 + layer, -value]
        blocks.append(block)
        if category is None:
            dropped_roles_auto.append(role)

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
        if category is not None:
            for offset in range(count):
                expression[str(cursor + offset)] = category
        cursor += count
    if expression_override is not None:
        expression = expression_override

    payload: dict = {"run_id": expression_run_id, "expression": expression}
    if method is not None:
        payload["method"] = method
    if dropped_roles_override is not None:
        payload["dropped_roles"] = dropped_roles_override
    elif dropped_roles_auto:
        payload["dropped_roles"] = sorted(dropped_roles_auto)
    if n_roles_dropped is not None:
        payload["n_roles_dropped"] = n_roles_dropped
    if probe_holdout_agreement is not None:
        payload["probe_holdout_agreement"] = probe_holdout_agreement

    (tmp_path / "role_expression.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
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


# --- Bulgu 6: bayat role_expression.json (+ rol düzeyi geri çekilmenin ------
# BİLİNÇLİ olarak dışarıda bıraktığı roller) -----------------------------------
#
# `--role-level-fallback` (scripts/06_label_and_train_probe.py) hiçbir
# kategoride >=10 etiket toplayamayan bir rolü ATAR: o rolün satırları
# `expression`'da hiç görünmez, rol adı `dropped_roles`'a yazılır. Aşağıdaki
# testler bunu "bayat artefakt" ile karıştırmadığını, ama aynı zamanda
# `dropped_roles` beyanının kendisinin de doğrulandığını kanıtlıyor.


def test_probe_path_without_dropped_roles_still_requires_full_coverage(
    tmp_path, monkeypatch, capsys
):
    """`dropped_roles` alanı YOKSA (probe yolu) davranış eskisiyle birebir
    aynı kalmalı: eksik HER anahtar bayatlık işaretidir, hiçbir rol muaf
    değildir. Farklı bir --limit ile üretilmiş eski bir harita sessizce
    kısmi hizasızlık yaratırdı."""
    _patch_paths(monkeypatch, tmp_path)
    _write_dataset(
        tmp_path,
        role_spec=[("a", "fully", 12, 1.0), ("b", "fully", 12, 4.0)],
        expression_override={str(i): "fully" for i in range(10)},  # 24 yerine 10
    )

    assert ea.main(_ARGS) == 2

    captured = capsys.readouterr()
    assert "BAŞARISIZ" in captured.err
    assert "14" in captured.err  # 24 rol satırından 10'u kapsandı, 14'ü eksik
    assert "atılmamış" in captured.err
    assert "GEÇTİ" not in captured.out


def test_fails_when_an_expression_entry_points_at_a_nonexistent_row(
    tmp_path, monkeypatch, capsys
):
    """Anahtar sayısı tutsa bile satır numaraları kaymış olabilir — hiçbiri
    indeksteki gerçek bir satıra karşılık gelmiyor."""
    _patch_paths(monkeypatch, tmp_path)
    _write_dataset(
        tmp_path,
        role_spec=[("a", "fully", 12, 1.0), ("b", "fully", 12, 4.0)],
        expression_override={str(i + 100): "fully" for i in range(24)},
    )

    assert ea.main(_ARGS) == 2

    captured = capsys.readouterr()
    assert "BAŞARISIZ" in captured.err
    assert "geçersiz" in captured.err
    assert "24" in captured.err  # 24 anahtarın HEPSİ geçersiz
    assert "GEÇTİ" not in captured.out


def test_fails_when_an_expression_entry_points_at_a_non_role_row(tmp_path, monkeypatch, capsys):
    """Anahtar gerçekten var olan bir satırı işaret ediyor ama o satır
    'default' türünde — rol satırı DEĞİL. Var olmayan satırdan farklı ama
    aynı hard-failure sınıfı."""
    _patch_paths(monkeypatch, tmp_path)
    _, index, _ = _write_dataset(
        tmp_path, role_spec=[("a", "fully", 12, 1.0), ("b", "fully", 12, 4.0)]
    )
    default_row_index = next(i for i, r in enumerate(index["rows"]) if r["kind"] == "default")
    payload = json.loads((tmp_path / "role_expression.json").read_text(encoding="utf-8"))
    payload["expression"][str(default_row_index)] = "fully"
    (tmp_path / "role_expression.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )

    assert ea.main(_ARGS) == 2

    captured = capsys.readouterr()
    assert "BAŞARISIZ" in captured.err
    assert "geçersiz" in captured.err
    assert "GEÇTİ" not in captured.out


def test_fails_when_a_non_dropped_role_row_is_missing_an_entry(tmp_path, monkeypatch, capsys):
    """Fallback artefaktında bile ATILMAMIŞ bir rolün her satırı zorunludur —
    yalnızca `dropped_roles`'ta adı geçen roller muaf. Burada 'b'nin son
    satırı (23) eksik ama 'b' dropped_roles'ta değil: hâlâ hard failure."""
    _patch_paths(monkeypatch, tmp_path)
    _write_dataset(
        tmp_path,
        role_spec=[
            ("a", "fully", 12, 1.0),
            ("b", "fully", 12, 4.0),
            ("dusuruldu", None, 5, 50.0),  # gerçekten atılmış rol, dokunulmuyor
        ],
    )
    payload = json.loads((tmp_path / "role_expression.json").read_text(encoding="utf-8"))
    assert payload["dropped_roles"] == ["dusuruldu"]  # ön koşul: otomatik yazıldı
    del payload["expression"]["23"]
    (tmp_path / "role_expression.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )

    assert ea.main(_ARGS) == 2

    captured = capsys.readouterr()
    assert "BAŞARISIZ" in captured.err
    assert "atılmamış 1 rol satırının" in captured.err
    assert "23" in captured.err
    assert "GEÇTİ" not in captured.out


def test_fails_when_a_dropped_role_still_has_entries(tmp_path, monkeypatch, capsys):
    """`dropped_roles` bir rolü ATILDI diye listeliyorsa o rolün HİÇBİR
    satırı `expression`'da olmamalı — varsa artefakt kendi içinde
    tutarsızdır (probe reddi öncesi bir role ait etiketler yanlışlıkla
    sızmış olabilir)."""
    _patch_paths(monkeypatch, tmp_path)
    _write_dataset(
        tmp_path,
        role_spec=[
            ("a", "fully", 12, 1.0),
            ("b", "fully", 12, 4.0),
            ("dusuruldu", None, 5, 50.0),
        ],
    )
    payload = json.loads((tmp_path / "role_expression.json").read_text(encoding="utf-8"))
    payload["expression"]["24"] = "fully"  # "dusuruldu"nun ilk satırına kaçak kayıt
    (tmp_path / "role_expression.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )

    assert ea.main(_ARGS) == 2

    captured = capsys.readouterr()
    assert "BAŞARISIZ" in captured.err
    assert "dusuruldu" in captured.err
    assert "atıldı" in captured.err
    assert "GEÇTİ" not in captured.out


def test_fails_when_a_dropped_role_name_is_not_in_the_catalog(tmp_path, monkeypatch, capsys):
    """`dropped_roles`'ta adı geçen bir rol indeksin rol kataloğunda hiç
    yoksa iki dosya farklı rol kümelerinden geliyor demektir."""
    _patch_paths(monkeypatch, tmp_path)
    _write_dataset(
        tmp_path,
        role_spec=[("a", "fully", 12, 1.0), ("b", "fully", 12, 4.0)],
        dropped_roles_override=["hayalet"],
    )

    assert ea.main(_ARGS) == 2

    captured = capsys.readouterr()
    assert "BAŞARISIZ" in captured.err
    assert "hayalet" in captured.err
    assert "GEÇTİ" not in captured.out


def test_role_level_fallback_artifact_with_dropped_roles_is_accepted(tmp_path, monkeypatch):
    """Görevin tam senaryosu: gerçek fallback artefaktı gibi bazı roller
    (bilerek) hiç kapsanmıyor ama geri kalan kapsama tam — kabul edilmeli,
    atılan rol eksene hiç katkı vermemeli."""
    _patch_paths(monkeypatch, tmp_path)
    _write_dataset(
        tmp_path,
        role_spec=[
            ("a", "fully", 12, 1.0),
            ("b", "fully", 12, 4.0),
            ("dusuruldu", None, 5, 50.0),
        ],
        method="role_level_fallback",
        n_roles_dropped=1,
        probe_holdout_agreement=0.635,
    )

    assert ea.main(_ARGS) in (0, 1)  # karar ne olursa olsun kabul edilmeli

    names = json.loads((tmp_path / "axis" / "role_names.json").read_text(encoding="utf-8"))
    assert "dusuruldu" not in names  # atılan rol hiç değerlendirilmedi

    report = json.loads((tmp_path / "axis" / "criterion_a.json").read_text(encoding="utf-8"))
    assert report["n_role_vectors"] == 2  # yalnızca a, b
    assert report["role_expression_method"] == "role_level_fallback"
    assert report["role_expression_n_roles_dropped"] == 1
    assert report["role_expression_probe_holdout_agreement"] == pytest.approx(0.635)


def test_criterion_a_records_probe_method_by_default(tmp_path, monkeypatch):
    """`method` alanı YOKSA (probe yolu, bkz. 06'nın probe dalının
    write_text'i) `criterion_a.json` bunu açıkça "probe" olarak kaydetmeli,
    fallback'e özgü iki alan `None` kalmalı."""
    _patch_paths(monkeypatch, tmp_path)
    _write_dataset(
        tmp_path,
        role_spec=[("a", "fully", 12, 1.0), ("b", "fully", 12, 4.0)],
    )

    assert ea.main(_ARGS) in (0, 1)

    report = json.loads((tmp_path / "axis" / "criterion_a.json").read_text(encoding="utf-8"))
    assert report["role_expression_method"] == "probe"
    assert report["role_expression_n_roles_dropped"] is None
    assert report["role_expression_probe_holdout_agreement"] is None


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
    assert report["cumulative_variance_top_components"] == pytest.approx(
        float(np.cumsum(ratios_first10)[-1])
    )
    assert report["cumulative_variance_top_components"] < 0.70
    # Minor: burada toplanan bileşen sayısı tam olarak 10 — alan adı artık
    # bu sayıyı hardcode etmiyor, ayrı bir alanda AÇIKÇA duruyor.
    assert report["cumulative_variance_n_components"] == 10


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


def test_criterion_a_records_the_min_role_vectors_floor_actually_used(tmp_path, monkeypatch):
    """Minor: gevşetilmiş bir taban ön kaydedilmiş bir hüküm için MADDİ bir
    sapmadır ve eskiden yalnızca `n_role_vectors >= 40` mı diye BAKARAK
    dolaylı çıkarılabilirdi. `criterion_a.json` artık fiilen KULLANILAN
    `--min-role-vectors` değerini AÇIKÇA taşıyor."""
    _patch_paths(monkeypatch, tmp_path)
    _write_dataset(
        tmp_path,
        role_spec=[("a", "fully", 12, 1.0), ("b", "fully", 12, 4.0), ("c", "fully", 12, 9.0)],
    )

    assert ea.main(["--min-role-vectors", "3"]) in (0, 1)

    report = json.loads((tmp_path / "axis" / "criterion_a.json").read_text(encoding="utf-8"))
    assert report["min_role_vectors"] == 3


def test_cumulative_variance_n_components_reflects_fewer_than_10_role_vectors(
    tmp_path, monkeypatch
):
    """Minor: `cumulative_variance_at_10` adı sabit "10" varsayıyordu ama
    `ratios_mid = ratios_full[:10]` yalnızca EN FAZLA 10 eleman taşır —
    `--min-role-vectors` spec tabanının (40) altına bilinçli gevşetilirse
    (burada 3 rol vektörüyle) bu sayı 10'un ALTINDA kalır. Alan adı artık
    sayıyı hardcode etmiyor; kaç bileşenin fiilen toplandığı ayrı bir alanda
    (`cumulative_variance_n_components`) AÇIKÇA duruyor."""
    _patch_paths(monkeypatch, tmp_path)
    _write_dataset(
        tmp_path,
        role_spec=[("a", "fully", 12, 1.0), ("b", "fully", 12, 4.0), ("c", "fully", 12, 9.0)],
    )

    assert ea.main(["--min-role-vectors", "3"]) in (0, 1)

    report = json.loads((tmp_path / "axis" / "criterion_a.json").read_text(encoding="utf-8"))
    # Yalnızca 3 rol vektörü var: PCA en fazla 3 bileşen üretebilir.
    assert len(report["explained_variance_ratio"]) == 3
    assert report["cumulative_variance_n_components"] == 3
    assert report["cumulative_variance_n_components"] < 10


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


# --- Kritik 1: aynı spec'ler + farklı cevaplar -> farklı run_id -> reddedilir -


def test_rejects_when_index_and_expression_come_from_identical_specs_but_different_answers(
    tmp_path, monkeypatch, capsys
):
    """Kritik 1'in tam senaryosu, uçtan uca: `04` `temperature=1.0` ile
    örnekler, yani aynı spec'lerle (`kind`/`role`/`system_prompt`/`question`)
    iki ayrı üretim koşusu FARKLI `answer` üretebilir — hayatta kalan kayıt
    listesi (boş yanıtlar hariç) değişmediği sürece bu, düzeltme öncesi
    `run_id`'yi DEĞİŞTİRMEZDİ. `05` cevabı aktivasyona tokenize eder, `06`'nın
    hakem etiketleri cevap metni üzerinden verilir; `run_id` cevaba duyarlı
    olmayınca `activations_index.json` (cevap kümesi A'dan) ve
    `role_expression.json` (cevap kümesi B'den) AYNI `run_id`'yi taşıyıp
    `07`'nin üç bütünlük kontrolünün (satır sayısı, anahtar kapsaması,
    `run_id` eşitliği) HEPSİNİ geçebilirdi — satır *i*'nin etiketi cevap B'yi
    tarif ederken satır *i*'nin aktivasyonu cevap A'yı kodlardı, sessizce.

    Düzeltmeden sonra `rollouts_run_id` cevabı da hash'e katıyor: aynı
    spec'lerin iki farklı cevap kümesi FARKLI bir `run_id` üretir ve `07`
    bunu (zaten var olan) `run_id` eşleşmezliği kontrolüyle reddeder."""
    from aax.prompts import RolloutSpec
    from aax.rollouts import rollout_record, rollouts_run_id

    def _records_for_answers(answers: list[str]) -> list[dict]:
        return [
            rollout_record(
                RolloutSpec(
                    kind="role",
                    role="a",
                    system_prompt="a ol",
                    question="q?",
                    sample_index=0,
                ),
                answer,
            )
            for answer in answers
        ]

    # Aynı spec'ler (kind/role/system_prompt/question), FARKLI cevaplar —
    # tıpkı `04`'ün temperature=1.0 ile iki ayrı üretiminin verdiği gibi.
    run_a = rollouts_run_id(_records_for_answers([f"cevap-{i}" for i in range(12)]))
    run_b = rollouts_run_id(_records_for_answers([f"BAMBAŞKA-{i}" for i in range(12)]))
    assert run_a != run_b, "Kritik 1'in ön koşulu: farklı cevap -> farklı run_id"

    _patch_paths(monkeypatch, tmp_path)
    _write_dataset(
        tmp_path,
        role_spec=[("a", "fully", 12, 1.0), ("b", "fully", 12, 4.0)],
        run_id=run_a,  # activations_index.json: 05'in yakaladığı cevap kümesi
        expression_run_id=run_b,  # role_expression.json: 06'nın etikettelediği cevap kümesi
    )

    assert ea.main(_ARGS) == 2

    captured = capsys.readouterr()
    assert "GEÇTİ" not in captured.out
    assert "BAŞARISIZ" in captured.err
    assert run_a in captured.err
    assert run_b in captured.err
    assert not (tmp_path / "axis" / "criterion_a.json").exists()
