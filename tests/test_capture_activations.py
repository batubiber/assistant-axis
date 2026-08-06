"""`scripts/05_capture_activations.py` içindeki saf mantığın testleri.

Ağa çıkmaz, HF/torch model yüklemeye hiç dokunmaz: yalnızca `build_arg_parser`
(argparse tanımı) test edilir. Script dosya adı bir rakamla başladığı için
(`05_capture_activations.py`) normal `import` ile içe aktarılamaz; `importlib`
ile dosya yolundan yüklenir (bkz. `tests/test_judge_gate.py`).
"""
from __future__ import annotations

import importlib.util
import sys
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
