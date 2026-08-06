"""`scripts/02_pilot_rollouts.py` testleri.

Model, GPU, ağ yok: `load_hf_model` sahtelenir ve `torch` `sys.modules`
üzerinden sahte bir modülle değiştirilir (script `torch`'u `main()` İÇİNDE
import ediyor). Script dosya adı bir rakamla başladığı için normal `import`
ile içe aktarılamaz; `importlib` ile dosya yolundan yüklenir (bkz.
`tests/test_judge_gate.py`).
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from contextlib import contextmanager
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "02_pilot_rollouts.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("pilot_rollouts", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


pr = _load_script()


def test_module_is_registered_in_sys_modules():
    assert sys.modules["pilot_rollouts"] is pr


class _FakeEncoding(dict):
    def to(self, _device):
        return self


class _FakeIds:
    shape = (1, 3)


class _FakeTokenizer:
    eos_token_id = 0

    def __init__(self, answers: list[str]) -> None:
        self._answers = answers
        self._decoded = 0

    def apply_chat_template(self, messages, **kwargs):
        return " | ".join(f"{m['role']}:{m['content']}" for m in messages)

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        return _FakeEncoding(input_ids=_FakeIds(), attention_mask=_FakeIds())

    def decode(self, _sequence, skip_special_tokens=True):
        answer = self._answers[self._decoded % len(self._answers)]
        self._decoded += 1
        return answer


class _FakeModel:
    device = "cpu"

    def generate(self, **kwargs):
        return [[0, 1, 2, 3, 4]]


class _FakeBundle:
    def __init__(self, answers: list[str]) -> None:
        self.tokenizer = _FakeTokenizer(answers)
        self.model = _FakeModel()


def _setup(monkeypatch, tmp_path, answers: list[str], *, n_roles: int = 2, n_questions: int = 2):
    monkeypatch.setattr(pr.config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(pr, "OUT_PATH", tmp_path / "pilot_rollouts.jsonl")
    monkeypatch.setattr(pr, "load_hf_model", lambda *a, **k: _FakeBundle(answers))

    fake_torch = types.ModuleType("torch")

    @contextmanager
    def no_grad():
        yield

    fake_torch.no_grad = no_grad
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    (tmp_path / "roles.json").write_text(
        json.dumps(
            {
                "complete": True,
                "limit": None,
                "requested": n_roles,
                "catalog_size": n_roles,
                "roles": [
                    {
                        "role": f"rol{i}",
                        "description": f"aciklama {i}",
                        "instructions": [f"You are rol{i}."],
                        "questions": ["q"],
                    }
                    for i in range(n_roles)
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "questions.json").write_text(
        json.dumps({"shared_questions": [f"soru {i}?" for i in range(n_questions)]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["02_pilot_rollouts.py", "--roles", str(n_roles),
                                      "--questions", str(n_questions)])


def _read_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_pilot_records_have_the_rollout_record_schema(tmp_path, monkeypatch):
    """D5: kayıtlar burada ELLE kuruluyordu (`{"role": ..., "system_prompt":
    ..., "question": ..., "answer": ...}`), yani `rollout_record`'ın şemasını
    ve korumasını atlıyordu."""
    _setup(monkeypatch, tmp_path, ["cevap"])

    assert pr.main() == 0

    records = _read_records(pr.OUT_PATH)
    assert records, "hiç kayıt yazılmadı"
    for record in records:
        assert set(record) == {
            "kind",
            "role",
            "system_prompt",
            "question",
            "sample_index",
            "answer",
        }
        assert record["kind"] == "role"


def test_empty_pilot_answers_never_reach_the_worksheet(tmp_path, monkeypatch, capsys):
    """Boş bir pilot yanıtı, BLOKLAYICI kapının ~40 slotundan birini yer ve
    insan etiketleyiciye boş bir satır olarak çıkardı — kapının ölçtüğü uyum
    oranı o satıra göre kayardı. `rollout_record` boş yanıtı reddeder; script
    onu artık sayıp atlıyor."""
    _setup(monkeypatch, tmp_path, ["dolu", "   "])  # her ikinci yanıt boş

    assert pr.main() == 0

    records = _read_records(pr.OUT_PATH)
    assert records
    assert all(r["answer"].strip() for r in records)
    out = capsys.readouterr().out
    assert "boş yanıt atlandı" in out


# --- Önemli 6: pilot varsayılanı kapının tabanına sıfır pay bırakıyordu ------


def test_default_pilot_size_is_9_roles_by_5_questions_end_to_end(tmp_path, monkeypatch):
    """Uçtan uca: bayraksız (`--roles`/`--questions` verilmeden) bir koşu
    9 rol × 5 soru = 45 kayıt üretmeli — kapının 40'lık tabanının 5 üstünde."""
    _setup(monkeypatch, tmp_path, ["cevap"], n_roles=9, n_questions=5)
    monkeypatch.setattr(sys, "argv", ["02_pilot_rollouts.py"])  # bayraksız, saf varsayılan

    assert pr.main() == 0

    records = _read_records(pr.OUT_PATH)
    assert len(records) == 45


def test_warns_when_pilot_output_falls_below_the_gate_floor(tmp_path, monkeypatch, capsys):
    """Önemli 6: pilot boş yanıtlar yüzünden kapının tabanının (40) altına
    düşerse operatör bunu insan etiketlemesine BAŞLAMADAN önce görmeli."""
    # 9x5=45 spec, yanıtların yarısı boş -> ~22-23 kayıt, 40'ın altında.
    _setup(monkeypatch, tmp_path, ["dolu", "   "], n_roles=9, n_questions=5)
    monkeypatch.setattr(sys, "argv", ["02_pilot_rollouts.py"])

    assert pr.main() == 0

    records = _read_records(pr.OUT_PATH)
    assert len(records) < 40
    out = capsys.readouterr().out
    assert "UYARI" in out
    assert "40" in out


def test_no_warning_when_pilot_output_meets_the_gate_floor(tmp_path, monkeypatch, capsys):
    _setup(monkeypatch, tmp_path, ["cevap"], n_roles=9, n_questions=5)
    monkeypatch.setattr(sys, "argv", ["02_pilot_rollouts.py"])

    assert pr.main() == 0

    out = capsys.readouterr().out
    assert "UYARI" not in out
