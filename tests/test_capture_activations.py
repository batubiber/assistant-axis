"""`scripts/05_capture_activations.py` içindeki saf mantığın testleri.

Ağa çıkmaz, HF/torch model yüklemeye hiç dokunmaz: `build_arg_parser`
(argparse tanımı) ve `compute_run_id` (içerikten türetilen koşu kimliği) test
edilir. Script dosya adı bir rakamla başladığı için (`05_capture_activations.py`)
normal `import` ile içe aktarılamaz; `importlib` ile dosya yolundan yüklenir
(bkz. `tests/test_judge_gate.py`).
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "05_capture_activations.py"
)


def _load_script():
    spec = importlib.util.spec_from_file_location("capture_activations", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ca = _load_script()


def test_module_is_registered_in_sys_modules():
    """Repo kuralı (bkz. test_judge_gate.py, test_generate_role_data.py):
    rakamla başlayan script'i importlib ile yüklerken modülü sys.modules'e
    de kaydet."""
    assert sys.modules["capture_activations"] is ca


def test_batch_size_arg_has_help_text():
    parser = ca.build_arg_parser()
    actions = {a.dest: a for a in parser._actions if a.dest != "help"}
    assert actions["batch_size"].help


# --- run_id: activations_index.json'ı kaynağa geri bağla ---------------------


def _make_records(roles: list[str]) -> list[dict]:
    return [
        {"kind": "role", "role": role, "system_prompt": f"{role} ol", "question": "q?"}
        for role in roles
    ]


def test_compute_run_id_is_deterministic_for_the_same_content():
    """`00_generate_role_data.py::compute_run_id` ile aynı desen: içerikten
    türetilir, aynı içerik her koşuda aynı kimliği verir."""
    a = _make_records(["pirate", "sage"])
    b = _make_records(["pirate", "sage"])
    assert ca.compute_run_id(a) == ca.compute_run_id(b)
    assert len(ca.compute_run_id(a)) == 16


def test_compute_run_id_changes_when_the_row_content_changes():
    """Farklı bir rollout kümesi farklı bir kimlik üretmeli — aksi hâlde
    `criterion_a.json`'daki `run_id` kaynağı ayırt edemez."""
    a = _make_records(["pirate", "sage"])
    b = _make_records(["pirate", "ghost"])
    assert ca.compute_run_id(a) != ca.compute_run_id(b)


def test_compute_run_id_is_not_clock_based():
    """Kimlik içerikten türetilir — aynı içerik her zaman aynı kimlik
    (bkz. `00_generate_role_data.py::test_run_id_is_not_clock_based`)."""
    records = _make_records(["pirate"])
    first = ca.compute_run_id(records)
    time.sleep(0.01)
    assert ca.compute_run_id(records) == first
