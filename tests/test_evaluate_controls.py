"""`scripts/10_evaluate_controls.py` testleri.

Bu script AĞA ÇIKMAZ: girdileri yalnızca `09_evaluate_steering.py`'nin diske
yazdığı `rate_by_strength(_<AD>).json` dosyaları ve `08_steering_sweep.py`'nin
yazdığı `steering_sweep_<AD>_meta.json`'lardır — hiçbir test sahte bir gateway
istemcisine ihtiyaç duymaz (`tests/conftest.py`'nin soket kilidi yine de ikinci
bir savunma katmanı olarak duruyor).

Script dosya adı bir rakamla başladığı için normal `import` ile içe
aktarılamaz; `importlib` ile dosya yolundan yüklenir (bkz.
`tests/test_evaluate_steering.py`, `tests/test_steering_sweep.py`).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_P = Path(__file__).resolve().parents[1] / "scripts" / "10_evaluate_controls.py"


def _load():
    spec = importlib.util.spec_from_file_location("evaluate_controls", _P)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ec = _load()


def test_module_is_registered_in_sys_modules():
    assert "evaluate_controls" in sys.modules


def test_control_kinds_match_the_three_preregistered_directions():
    assert ec.CONTROL_KINDS == ("gaussian", "shuffled", "rolespan")


def test_far_strength_is_the_preregistered_minus_zero_point_six():
    assert ec.FAR_STRENGTH == -0.6


# --- saf yardımcılar: load_rate_by_strength ---------------------------------


def test_load_rate_by_strength_parses_string_keys_to_numeric(tmp_path):
    path = tmp_path / "rate_by_strength.json"
    path.write_text(json.dumps({"14": {"-0.6": 0.94, "0.0": 0.456}}), encoding="utf-8")
    out = ec.load_rate_by_strength(path)
    assert out == {14: {-0.6: 0.94, 0.0: 0.456}}
    assert isinstance(next(iter(out)), int)
    assert isinstance(next(iter(out[14])), float)


def test_load_rate_by_strength_rejects_non_object_top_level(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(ValueError, match="nesne"):
        ec.load_rate_by_strength(path)


def test_load_rate_by_strength_rejects_non_object_layer_value(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"14": [1, 2, 3]}), encoding="utf-8")
    with pytest.raises(ValueError, match="14"):
        ec.load_rate_by_strength(path)


# --- saf yardımcılar: direction_delta ---------------------------------------


def test_direction_delta_computes_delta_and_marks_below_threshold():
    out = ec.direction_delta({-0.6: 0.30}, baseline_rate=0.10)
    assert out["delta"] == pytest.approx(0.20)
    assert out["reached_threshold"] is False
    assert out["passed"] is True


def test_direction_delta_reaches_threshold_exactly():
    """B kriterindeki 'tam eşikte GEÇER' semantiğinin AYNISI — C için bu
    'ULAŞTI' (kontrolün kendisi B kriterini geçebiliyor) demek, yani C
    kriteri açısından KÖTÜ (`passed: False`)."""
    out = ec.direction_delta({-0.6: 0.35}, baseline_rate=0.10)
    assert out["delta"] == pytest.approx(0.25)
    assert out["reached_threshold"] is True
    assert out["passed"] is False


def test_direction_delta_epsilon_absorbs_the_same_float_noise_as_criterion_b():
    """`0.35 - 0.10 == 0.24999999999999997` — B kriterindeki AYNI gürültü.
    Yeni bir tolerans İCAT EDİLMEDİĞİNİ, `B_THRESHOLD_EPS`'in AYNEN
    kullanıldığını doğrular."""
    assert 0.35 - 0.10 != 0.25
    out = ec.direction_delta({-0.6: 0.35}, baseline_rate=0.10)
    assert out["reached_threshold"] is True
    assert out["threshold_eps"] == pytest.approx(1e-9)


def test_direction_delta_missing_far_strength_raises():
    with pytest.raises(ValueError, match="-0.6"):
        ec.direction_delta({-0.4: 0.30}, baseline_rate=0.10)


def test_direction_delta_rejects_non_finite_rates():
    with pytest.raises(ValueError, match="sonlu"):
        ec.direction_delta({-0.6: float("nan")}, baseline_rate=0.10)


def test_direction_delta_returns_the_full_documented_schema():
    out = ec.direction_delta({-0.6: 0.40}, baseline_rate=0.10)
    assert set(out.keys()) == {
        "baseline_strength", "baseline_rate", "far_strength", "far_rate",
        "delta", "threshold", "threshold_eps", "reached_threshold",
        "passed", "reason",
    }


# --- saf yardımcılar: evaluate_direction (paylaşılan taban) -----------------


def test_evaluate_direction_uses_the_axis_baseline_not_the_controls_own():
    """Kontrolün KENDİ 0.0'ı varsa bile (normalde hiç olmaz) YOK SAYILIR —
    taban HER ZAMAN eksenin `axis_rates`'inden okunur."""
    control_rates = {14: {-0.6: 0.90}}
    axis_rates = {14: {0.0: 0.456, -0.6: 0.94}}
    out = ec.evaluate_direction(control_rates, axis_rates)
    assert out[14]["baseline_rate"] == pytest.approx(0.456)
    assert out[14]["delta"] == pytest.approx(0.90 - 0.456)


def test_evaluate_direction_raises_when_axis_has_no_such_layer():
    control_rates = {19: {-0.6: 0.90}}
    axis_rates = {14: {0.0: 0.456}}
    with pytest.raises(ValueError, match="L19"):
        ec.evaluate_direction(control_rates, axis_rates)


def test_evaluate_direction_raises_when_axis_layer_has_no_zero_baseline():
    control_rates = {14: {-0.6: 0.90}}
    axis_rates = {14: {-0.6: 0.94}}  # 0.0 hiç yok
    with pytest.raises(ValueError, match="0.0"):
        ec.evaluate_direction(control_rates, axis_rates)


def test_evaluate_direction_rejects_empty_control_rates():
    with pytest.raises(ValueError, match="hiç"):
        ec.evaluate_direction({}, {14: {0.0: 0.1}})


# --- saf yardımcılar: overall_exit_code -------------------------------------


def test_overall_exit_code_zero_when_nothing_reaches_the_threshold():
    verdicts = {
        "gaussian": {14: {"reached_threshold": False}},
        "shuffled": {14: {"reached_threshold": False}},
    }
    assert ec.overall_exit_code(verdicts) == 0


def test_overall_exit_code_one_when_any_direction_reaches_the_threshold():
    verdicts = {
        "gaussian": {14: {"reached_threshold": False}},
        "rolespan": {14: {"reached_threshold": True}},
    }
    assert ec.overall_exit_code(verdicts) == 1


def test_overall_exit_code_two_for_empty_verdicts():
    assert ec.overall_exit_code({}) == 2
    assert ec.overall_exit_code({"gaussian": {}}) == 2


# --- ortak test yardımcıları (I/O seviyesi) ---------------------------------


def _patch_paths(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    results_dir = tmp_path / "results"
    monkeypatch.setattr(ec.config, "model_data_dir", lambda: data_dir)
    monkeypatch.setattr(ec.config, "model_results_dir", lambda: results_dir)
    return data_dir, results_dir


def _write_axis_rates(results_dir: Path, rates: dict) -> None:
    out_dir = results_dir / "steering"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rate_by_strength.json").write_text(json.dumps(rates), encoding="utf-8")


def _write_control_rates(results_dir: Path, kind: str, rates: dict) -> None:
    out_dir = results_dir / "steering"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"rate_by_strength_{kind}.json").write_text(json.dumps(rates), encoding="utf-8")


def _write_control_meta(
    data_dir: Path, kind: str, *, seed: int = 0, sha256: str = "deadbeef00000000"
) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / f"steering_sweep_{kind}_meta.json").write_text(
        json.dumps({
            "direction_kind": kind, "direction_seed": seed, "direction_sha256": sha256,
        }),
        encoding="utf-8",
    )


# Ölçülen gerçek Aşama 4 taban/artış büyüklüklerine yakın, ama küçük — testler
# hız için gerçek 250 öğelik oranları TEKRARLAMAZ.
_AXIS_RATES = {"14": {"0.0": 0.456, "-0.6": 0.94}}


def _write_all_three_controls_below_threshold(data_dir: Path, results_dir: Path) -> None:
    for kind in ec.CONTROL_KINDS:
        _write_control_rates(
            results_dir, kind,
            {"14": {"-0.6": 0.50, "-0.4": 0.48, "-0.2": 0.47}},  # taban 0.456 -> delta ~0.044
        )
        _write_control_meta(data_dir, kind, seed=1)


# --- I/O: eksik veri -> 2, ASLA 0/1 değil ------------------------------------


def test_missing_axis_baseline_file_exits_two(tmp_path, monkeypatch, capsys):
    data_dir, results_dir = _patch_paths(monkeypatch, tmp_path)
    # Eksen dosyası HİÇ yazılmadı.

    exit_code = ec.main([])

    assert exit_code == 2
    err = capsys.readouterr().err
    assert "BAŞARISIZ" in err
    assert "Traceback" not in err
    assert "rate_by_strength.json" in err


def test_missing_control_file_exits_two_never_zero_or_one(tmp_path, monkeypatch, capsys):
    data_dir, results_dir = _patch_paths(monkeypatch, tmp_path)
    _write_axis_rates(results_dir, _AXIS_RATES)
    # yalnızca gaussian ve shuffled yazıldı — rolespan'ın rate dosyası EKSİK.
    for kind in ("gaussian", "shuffled"):
        _write_control_rates(results_dir, kind, {"14": {"-0.6": 0.50}})
        _write_control_meta(data_dir, kind)

    exit_code = ec.main([])

    assert exit_code == 2
    err = capsys.readouterr().err
    assert "BAŞARISIZ" in err
    assert "rolespan" in err
    assert "rate_by_strength_rolespan.json" in err
    assert not (results_dir / "steering" / "criterion_c.json").exists()


def test_missing_control_meta_exits_two(tmp_path, monkeypatch, capsys):
    """`rate_by_strength_<AD>.json` var ama `steering_sweep_<AD>_meta.json`
    yoksa `direction_sha256` okunamaz — bu da 2."""
    data_dir, results_dir = _patch_paths(monkeypatch, tmp_path)
    _write_axis_rates(results_dir, _AXIS_RATES)
    for kind in ec.CONTROL_KINDS:
        _write_control_rates(results_dir, kind, {"14": {"-0.6": 0.50}})
        if kind != "shuffled":
            _write_control_meta(data_dir, kind)
    # 'shuffled' için meta KASITLI yazılmadı.

    exit_code = ec.main([])

    assert exit_code == 2
    err = capsys.readouterr().err
    assert "BAŞARISIZ" in err
    assert "shuffled" in err
    assert "steering_sweep_shuffled_meta.json" in err


def test_missing_minus_zero_point_six_in_a_control_exits_two(tmp_path, monkeypatch, capsys):
    data_dir, results_dir = _patch_paths(monkeypatch, tmp_path)
    _write_axis_rates(results_dir, _AXIS_RATES)
    for kind in ec.CONTROL_KINDS:
        if kind == "gaussian":
            # -0.6 EKSİK — yalnızca -0.4/-0.2 ölçülmüş gibi davran.
            _write_control_rates(results_dir, kind, {"14": {"-0.4": 0.50, "-0.2": 0.47}})
        else:
            _write_control_rates(results_dir, kind, {"14": {"-0.6": 0.50}})
        _write_control_meta(data_dir, kind)

    exit_code = ec.main([])

    assert exit_code == 2
    err = capsys.readouterr().err
    assert "BAŞARISIZ" in err
    assert "gaussian" in err
    assert "-0.6" in err
    assert not (results_dir / "steering" / "criterion_c.json").exists()


def test_missing_baseline_layer_in_axis_exits_two(tmp_path, monkeypatch, capsys):
    """Eksen dosyası VAR ama kontrolün ölçtüğü katmanı (14) hiç İÇERMİYOR
    (yalnızca 19'u ölçmüş gibi) — bu da 'eksik taban' sayılır."""
    data_dir, results_dir = _patch_paths(monkeypatch, tmp_path)
    _write_axis_rates(results_dir, {"19": {"0.0": 0.416, "-0.6": 0.896}})
    for kind in ec.CONTROL_KINDS:
        _write_control_rates(results_dir, kind, {"14": {"-0.6": 0.50}})
        _write_control_meta(data_dir, kind)

    exit_code = ec.main([])

    assert exit_code == 2
    err = capsys.readouterr().err
    assert "BAŞARISIZ" in err
    assert "L14" in err


def test_corrupt_axis_file_exits_two_not_traceback(tmp_path, monkeypatch, capsys):
    data_dir, results_dir = _patch_paths(monkeypatch, tmp_path)
    out_dir = results_dir / "steering"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rate_by_strength.json").write_text("{bozuk json", encoding="utf-8")

    exit_code = ec.main([])

    assert exit_code == 2
    err = capsys.readouterr().err
    assert "BAŞARISIZ" in err
    assert "Traceback" not in err


# --- I/O: gerçek kararlar -----------------------------------------------------


def test_all_controls_below_threshold_exits_zero_and_writes_artifact(tmp_path, monkeypatch):
    data_dir, results_dir = _patch_paths(monkeypatch, tmp_path)
    _write_axis_rates(results_dir, _AXIS_RATES)
    _write_all_three_controls_below_threshold(data_dir, results_dir)

    exit_code = ec.main([])

    assert exit_code == 0
    payload = json.loads(
        (results_dir / "steering" / "criterion_c.json").read_text(encoding="utf-8")
    )
    assert payload["passed"] is True
    for kind in ec.CONTROL_KINDS:
        assert payload["directions"][kind]["passed"] is True
        assert payload["directions"][kind]["layers"]["14"]["reached_threshold"] is False


def test_one_control_reaching_threshold_exits_one(tmp_path, monkeypatch):
    data_dir, results_dir = _patch_paths(monkeypatch, tmp_path)
    _write_axis_rates(results_dir, _AXIS_RATES)
    _write_all_three_controls_below_threshold(data_dir, results_dir)
    # rolespan'ı KASITLI olarak eşiğe ULAŞTIR: 0.456 taban + >=0.25 = >=0.706.
    _write_control_rates(results_dir, "rolespan", {"14": {"-0.6": 0.94}})
    _write_control_meta(data_dir, "rolespan", seed=1)

    exit_code = ec.main([])

    assert exit_code == 1
    payload = json.loads(
        (results_dir / "steering" / "criterion_c.json").read_text(encoding="utf-8")
    )
    assert payload["passed"] is False
    assert payload["directions"]["rolespan"]["passed"] is False
    assert payload["directions"]["rolespan"]["layers"]["14"]["reached_threshold"] is True
    assert payload["directions"]["gaussian"]["passed"] is True
    assert payload["directions"]["shuffled"]["passed"] is True


def test_shared_baseline_is_read_from_the_axis_run_not_from_a_control(tmp_path, monkeypatch):
    """Bir kontrolün kendi dosyasına YANLIŞLIKLA/başka bir yoldan bir 0.0
    hücresi sızmış olsa bile (normalde hiç olmaz), taban YİNE DE eksenin
    kendi `rate_by_strength.json`'ından okunmalı — kontrolün 0.0'ı TAMAMEN
    YOK SAYILMALI."""
    data_dir, results_dir = _patch_paths(monkeypatch, tmp_path)
    _write_axis_rates(results_dir, _AXIS_RATES)  # eksenin 0.0'ı: 0.456
    # gaussian dosyasına YANLIŞ bir 0.0 (0.05) sızdırılmış gibi davran.
    _write_control_rates(results_dir, "gaussian", {"14": {"0.0": 0.05, "-0.6": 0.50}})
    _write_control_meta(data_dir, "gaussian")
    for kind in ("shuffled", "rolespan"):
        _write_control_rates(results_dir, kind, {"14": {"-0.6": 0.50}})
        _write_control_meta(data_dir, kind)

    exit_code = ec.main([])

    assert exit_code in (0, 1)
    payload = json.loads(
        (results_dir / "steering" / "criterion_c.json").read_text(encoding="utf-8")
    )
    gaussian = payload["directions"]["gaussian"]["layers"]["14"]
    assert gaussian["baseline_rate"] == pytest.approx(0.456), (
        "taban EKSENDEN gelmeli — kontrolün kendi (sızmış) 0.0'ı YOK SAYILMALI"
    )
    assert gaussian["delta"] == pytest.approx(0.50 - 0.456)


def test_direction_sha256_and_seed_recorded_from_variant_meta(tmp_path, monkeypatch):
    data_dir, results_dir = _patch_paths(monkeypatch, tmp_path)
    _write_axis_rates(results_dir, _AXIS_RATES)
    _write_all_three_controls_below_threshold(data_dir, results_dir)
    _write_control_meta(data_dir, "shuffled", seed=42, sha256="cafef00ddeadbeef")

    exit_code = ec.main([])

    assert exit_code == 0
    payload = json.loads(
        (results_dir / "steering" / "criterion_c.json").read_text(encoding="utf-8")
    )
    shuffled = payload["directions"]["shuffled"]
    assert shuffled["direction_seed"] == 42
    assert shuffled["direction_sha256"] == "cafef00ddeadbeef"
    assert shuffled["direction_kind"] == "shuffled"


def test_ratio_is_reported_but_does_not_gate_pass_fail(tmp_path, monkeypatch):
    """`ratio_axis_to_control` İKİNCİL bilgidir — C kararını (`passed`)
    ETKİLEMEMELİ, yalnızca raporlanmalı. Burada oran küçük (kontrol ekseninkine
    yakın) olsa bile kontrol eşiğin altında kaldığı için C GEÇER."""
    data_dir, results_dir = _patch_paths(monkeypatch, tmp_path)
    _write_axis_rates(results_dir, _AXIS_RATES)  # eksen deltası: 0.94 - 0.456 = 0.484
    for kind in ec.CONTROL_KINDS:
        # kontrol deltası 0.20 (eşiğin altında) — oran ~2.4, ama pass/fail
        # SADECE mutlak eşiğe bakmalı.
        _write_control_rates(results_dir, kind, {"14": {"-0.6": 0.656}})
        _write_control_meta(data_dir, kind)

    exit_code = ec.main([])

    assert exit_code == 0
    payload = json.loads(
        (results_dir / "steering" / "criterion_c.json").read_text(encoding="utf-8")
    )
    for kind in ec.CONTROL_KINDS:
        cell = payload["directions"][kind]["layers"]["14"]
        assert cell["passed"] is True
        assert cell["ratio_axis_to_control"] == pytest.approx(0.484 / 0.20, rel=1e-3)


def test_criterion_c_json_carries_every_documented_field(tmp_path, monkeypatch):
    data_dir, results_dir = _patch_paths(monkeypatch, tmp_path)
    _write_axis_rates(results_dir, _AXIS_RATES)
    _write_all_three_controls_below_threshold(data_dir, results_dir)

    exit_code = ec.main([])
    assert exit_code == 0

    payload = json.loads(
        (results_dir / "steering" / "criterion_c.json").read_text(encoding="utf-8")
    )
    assert set(payload.keys()) >= {
        "model", "far_strength", "threshold", "threshold_eps",
        "baseline_source", "directions", "passed", "note",
    }
    assert payload["far_strength"] == -0.6
    assert payload["threshold"] == pytest.approx(0.25)
    assert "rate_by_strength.json" in payload["baseline_source"]
    assert set(payload["directions"]) == {"gaussian", "shuffled", "rolespan"}
    for kind in ec.CONTROL_KINDS:
        d = payload["directions"][kind]
        assert set(d.keys()) >= {
            "layers", "passed", "direction_kind", "direction_seed", "direction_sha256",
        }
        cell = d["layers"]["14"]
        assert set(cell.keys()) >= {
            "baseline_strength", "baseline_rate", "far_strength", "far_rate",
            "delta", "threshold", "threshold_eps", "reached_threshold", "passed",
            "reason", "axis_delta", "ratio_axis_to_control",
        }


def test_failed_run_never_overwrites_a_previous_real_artifact(tmp_path, monkeypatch, capsys):
    """Karar üretemeyen bir koşu (burada: eksik bir kontrol dosyası) var olan
    GERÇEK bir `criterion_c.json`'ı EZMEMELİ — `09_evaluate_steering.py`'nin
    F5 ilkesinin AYNISI."""
    data_dir, results_dir = _patch_paths(monkeypatch, tmp_path)
    steering_dir = results_dir / "steering"
    steering_dir.mkdir(parents=True, exist_ok=True)
    previous_payload = {"passed": True, "directions": {"gaussian": {"passed": True}}}
    (steering_dir / "criterion_c.json").write_text(
        json.dumps(previous_payload), encoding="utf-8"
    )
    _write_axis_rates(results_dir, _AXIS_RATES)
    # kontrol dosyaları KASITLI eksik.

    exit_code = ec.main([])

    assert exit_code == 2
    after = json.loads((steering_dir / "criterion_c.json").read_text(encoding="utf-8"))
    assert after == previous_payload


def test_main_does_not_touch_the_network(tmp_path, monkeypatch):
    """Script'in HİÇ gateway istemcisi kurmadığını doğrular — `build_default_
    client` gibi bir isim modülde hiç YOK, dolayısıyla ağa çıkacak bir yol da
    yok."""
    assert not hasattr(ec, "build_default_client")
    assert not hasattr(ec, "GatewayClient")


def test_missing_axis_far_strength_yields_none_ratio_not_a_crash(tmp_path, monkeypatch):
    """Eksenin KENDİ -0.6 ölçümü (normalde Aşama 4'ün B kriteri zaten bunu
    şart koşar, ama savunma yine de test edilir) eksikse `axis_delta`/
    `ratio_axis_to_control` sessizce `None` olmalı — C KARARINI etkilememeli,
    koşuyu ÇÖKERTMEMELİ."""
    data_dir, results_dir = _patch_paths(monkeypatch, tmp_path)
    _write_axis_rates(results_dir, {"14": {"0.0": 0.456}})  # -0.6 eksik
    _write_all_three_controls_below_threshold(data_dir, results_dir)

    exit_code = ec.main([])

    assert exit_code == 0
    payload = json.loads(
        (results_dir / "steering" / "criterion_c.json").read_text(encoding="utf-8")
    )
    for kind in ec.CONTROL_KINDS:
        cell = payload["directions"][kind]["layers"]["14"]
        assert cell["axis_delta"] is None
        assert cell["ratio_axis_to_control"] is None
        assert cell["passed"] is True
