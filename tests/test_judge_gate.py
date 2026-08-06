"""`scripts/03_judge_gate.py` karar mantığı VE dosya işleme testleri.

Ağa çıkmaz: ölçüm fonksiyonlarını (`collapse_to_category`, `agreement_rate`,
`gate_passed`) doğrudan çağırır; `run_machine`/`run_score`'u `tmp_path` altına
yazılmış sahte dosyalarla ve ağsız `FakeJudgeClient` ile uçtan uca dener.
Gerçek `GatewayClient`'a hiç dokunulmaz, `build_default_client` gerektiğinde
monkeypatch'lenir. Script dosya adı bir rakamla başladığı için
(`03_judge_gate.py`) normal `import` ile içe aktarılamaz; `importlib` ile
dosya yolundan yüklenir (bkz. `tests/test_smoke_gateway.py`).
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from aax.gateway import BudgetExceeded, CircuitOpen, GatewayError
from aax.judge import JudgeParseError

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "03_judge_gate.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("judge_gate", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


judge_gate = _load_script()


def test_module_is_registered_in_sys_modules():
    """Repo kuralı (bkz. test_smoke_gateway.py, test_generate_role_data.py):
    rakamla başlayan script'i importlib ile yüklerken modülü sys.modules'e
    de kaydet."""
    assert sys.modules["judge_gate"] is judge_gate


# --- collapse_to_category / agreement_rate / gate_passed (saf, ağsız) -------


def test_collapse_maps_paper_rubric_to_three_categories():
    assert judge_gate.collapse_to_category(3) == "fully"
    assert judge_gate.collapse_to_category(2) == "somewhat"
    assert judge_gate.collapse_to_category(1) == "no"
    assert judge_gate.collapse_to_category(0) == "no"


def test_collapse_rejects_out_of_range():
    with pytest.raises(ValueError):
        judge_gate.collapse_to_category(4)


def test_agreement_is_computed_on_collapsed_categories():
    """0 ve 1 aynı kategoriye düştüğü için bu çift UYUŞUR."""
    assert judge_gate.agreement_rate([0], [1]) == 1.0


def test_agreement_counts_category_mismatch():
    assert judge_gate.agreement_rate([3, 3], [3, 2]) == 0.5


def test_agreement_rejects_length_mismatch():
    with pytest.raises(ValueError, match="uzunluk"):
        judge_gate.agreement_rate([1, 2], [1])


def test_agreement_rejects_empty_input():
    with pytest.raises(ValueError, match="boş"):
        judge_gate.agreement_rate([], [])


def test_gate_passes_at_threshold_exactly():
    assert judge_gate.gate_passed(0.75) is True
    assert judge_gate.gate_passed(0.7499) is False


# --- _parse_human_score (saf, Bulgu 3) --------------------------------------


def test_parse_human_score_accepts_valid_values():
    assert judge_gate._parse_human_score("0") == 0
    assert judge_gate._parse_human_score("3") == 3
    assert judge_gate._parse_human_score(" 2 ") == 2


def test_parse_human_score_rejects_non_numeric_text():
    with pytest.raises(ValueError, match="sayı değil"):
        judge_gate._parse_human_score("n/a")


def test_parse_human_score_rejects_trailing_dot():
    with pytest.raises(ValueError, match="sayı değil"):
        judge_gate._parse_human_score("3.")


def test_parse_human_score_rejects_out_of_range():
    with pytest.raises(ValueError, match="0-3 aralığı dışında"):
        judge_gate._parse_human_score("9")
    with pytest.raises(ValueError, match="0-3 aralığı dışında"):
        judge_gate._parse_human_score("-1")


# --- ortak test yardımcıları -------------------------------------------------


class FakeJudgeClient:
    """Ağsız sahte hakem istemcisi — `.chat()` çağrılarını sırayla
    `responses`'tan karşılar. Bir öğe `BaseException` ise fırlatılır."""

    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self.calls = 0
        self.sends_made = 0

    def chat(self, messages, *, stage, temperature=0.0, max_tokens=1024):
        outcome = self._responses[self.calls]
        self.calls += 1
        self.sends_made += 1
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _patch_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(judge_gate, "PILOT_PATH", tmp_path / "pilot_rollouts.jsonl")
    monkeypatch.setattr(judge_gate, "LABELS_PATH", tmp_path / "judge_gate_labels.csv")
    monkeypatch.setattr(judge_gate, "MACHINE_PATH", tmp_path / "judge_gate_machine.json")
    monkeypatch.setattr(judge_gate, "RESULT_PATH", tmp_path / "judge_gate.json")


def _write_pilot(path: Path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )


def _write_labels_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["idx", "role", "question", "answer", "human_score"])
        for row in rows:
            writer.writerow(
                [row["idx"], row["role"], row["question"], row["answer"], row["human_score"]]
            )


def _write_machine_json(path: Path, mapping: dict[int, int]) -> None:
    path.write_text(
        json.dumps({str(k): v for k, v in mapping.items()}, ensure_ascii=False),
        encoding="utf-8",
    )


# --- run_machine: kör çalışma sayfası (Bulgu 1) -----------------------------


def test_run_machine_worksheet_has_no_machine_score_column(tmp_path, monkeypatch):
    _patch_paths(monkeypatch, tmp_path)
    _write_pilot(
        judge_gate.PILOT_PATH,
        [
            {"role": "pirate", "question": "q1", "answer": "a1"},
            {"role": "sage", "question": "q2", "answer": "a2"},
        ],
    )
    client = FakeJudgeClient(["[3]", "[1]"])

    exit_code = judge_gate.run_machine(client)

    assert exit_code == 0
    with judge_gate.LABELS_PATH.open(encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    assert header == ["idx", "role", "question", "answer", "human_score"]
    assert "machine_score" not in header
    # Bulgu 1: makine puanı satırın hiçbir yerinde (sondaki fazladan sütun,
    # yorum satırı vb. olarak da) sızmamalı.
    worksheet_text = judge_gate.LABELS_PATH.read_text(encoding="utf-8")
    assert "machine_score" not in worksheet_text


def test_run_machine_worksheet_human_score_column_is_blank(tmp_path, monkeypatch):
    _patch_paths(monkeypatch, tmp_path)
    _write_pilot(judge_gate.PILOT_PATH, [{"role": "pirate", "question": "q1", "answer": "a1"}])
    client = FakeJudgeClient(["[3]"])

    judge_gate.run_machine(client)

    with judge_gate.LABELS_PATH.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {"idx": "0", "role": "pirate", "question": "q1", "answer": "a1", "human_score": ""}
    ]


def test_run_machine_writes_machine_scores_to_separate_json_keyed_by_idx(
    tmp_path, monkeypatch
):
    _patch_paths(monkeypatch, tmp_path)
    _write_pilot(
        judge_gate.PILOT_PATH,
        [
            {"role": "pirate", "question": "q1", "answer": "a1"},
            {"role": "sage", "question": "q2", "answer": "a2"},
        ],
    )
    client = FakeJudgeClient(["[3]", "[1]"])

    judge_gate.run_machine(client)

    machine = json.loads(judge_gate.MACHINE_PATH.read_text(encoding="utf-8"))
    assert machine == {"0": 3, "1": 1}


def test_run_machine_prints_blind_worksheet_instructions(tmp_path, monkeypatch, capsys):
    _patch_paths(monkeypatch, tmp_path)
    _write_pilot(judge_gate.PILOT_PATH, [{"role": "pirate", "question": "q1", "answer": "a1"}])
    client = FakeJudgeClient(["[2]"])

    judge_gate.run_machine(client)

    out = capsys.readouterr().out
    assert "kördür" in out
    assert str(judge_gate.MACHINE_PATH) in out


def test_run_machine_missing_pilot_file_raises_systemexit(tmp_path, monkeypatch):
    _patch_paths(monkeypatch, tmp_path)
    client = FakeJudgeClient([])

    with pytest.raises(SystemExit, match="pilot_rollouts.jsonl yok"):
        judge_gate.run_machine(client)


def test_run_machine_handles_judge_parse_error(tmp_path, monkeypatch, capsys):
    _patch_paths(monkeypatch, tmp_path)
    _write_pilot(judge_gate.PILOT_PATH, [{"role": "pirate", "question": "q1", "answer": "a1"}])
    client = FakeJudgeClient([JudgeParseError("bozuk JSON")])

    exit_code = judge_gate.run_machine(client)

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "BAŞARISIZ" in err
    assert "ayrıştırılamadı" in err
    assert not judge_gate.LABELS_PATH.exists()
    assert not judge_gate.MACHINE_PATH.exists()


def test_run_machine_handles_budget_exceeded(tmp_path, monkeypatch, capsys):
    _patch_paths(monkeypatch, tmp_path)
    _write_pilot(judge_gate.PILOT_PATH, [{"role": "pirate", "question": "q1", "answer": "a1"}])
    client = FakeJudgeClient(
        [BudgetExceeded("'stage05_judge_gate' aşama bütçesi doldu: 15/15")]
    )

    exit_code = judge_gate.run_machine(client)

    assert exit_code == 2
    assert "DURDURULDU" in capsys.readouterr().err


def test_run_machine_handles_circuit_open(tmp_path, monkeypatch, capsys):
    _patch_paths(monkeypatch, tmp_path)
    _write_pilot(judge_gate.PILOT_PATH, [{"role": "pirate", "question": "q1", "answer": "a1"}])
    client = FakeJudgeClient([CircuitOpen("devre kesici açık")])

    exit_code = judge_gate.run_machine(client)

    assert exit_code == 2
    assert "DURDURULDU" in capsys.readouterr().err


def test_run_machine_handles_gateway_error(tmp_path, monkeypatch, capsys):
    _patch_paths(monkeypatch, tmp_path)
    _write_pilot(judge_gate.PILOT_PATH, [{"role": "pirate", "question": "q1", "answer": "a1"}])
    client = FakeJudgeClient([GatewayError("HTTP 500")])

    exit_code = judge_gate.run_machine(client)

    assert exit_code == 2
    err = capsys.readouterr().err
    assert "BAŞARISIZ" in err
    assert "gateway çağrısı" in err


# --- run_score: idx birleştirme ve uyuşmazlık (Bulgu 1) ---------------------


def test_run_score_missing_labels_file_raises_systemexit(tmp_path, monkeypatch):
    _patch_paths(monkeypatch, tmp_path)
    with pytest.raises(SystemExit, match="judge_gate_labels.csv yok"):
        judge_gate.run_score()


def test_run_score_missing_machine_file_raises_systemexit(tmp_path, monkeypatch):
    _patch_paths(monkeypatch, tmp_path)
    _write_labels_csv(
        judge_gate.LABELS_PATH,
        [{"idx": 0, "role": "pirate", "question": "q", "answer": "a", "human_score": "3"}],
    )
    with pytest.raises(SystemExit, match="judge_gate_machine.json yok"):
        judge_gate.run_score()


def test_run_score_fails_loudly_when_labels_has_idx_missing_from_machine(
    tmp_path, monkeypatch
):
    """Bulgu 1: idx yalnızca worksheet'te varsa sessizce atlanmaz, patlar."""
    _patch_paths(monkeypatch, tmp_path)
    _write_labels_csv(
        judge_gate.LABELS_PATH,
        [
            {"idx": 0, "role": "pirate", "question": "q0", "answer": "a0", "human_score": "3"},
            {"idx": 1, "role": "sage", "question": "q1", "answer": "a1", "human_score": "2"},
        ],
    )
    _write_machine_json(judge_gate.MACHINE_PATH, {0: 3})  # idx 1 makine dosyasında yok

    with pytest.raises(SystemExit) as exc_info:
        judge_gate.run_score()

    message = str(exc_info.value)
    assert "uyuşmazlığı" in message
    assert "1" in message
    assert not judge_gate.RESULT_PATH.exists()


def test_run_score_fails_loudly_when_machine_has_extra_idx(tmp_path, monkeypatch):
    """Aynı bulgu, ters yön: idx yalnızca makine dosyasında var."""
    _patch_paths(monkeypatch, tmp_path)
    _write_labels_csv(
        judge_gate.LABELS_PATH,
        [{"idx": 0, "role": "pirate", "question": "q0", "answer": "a0", "human_score": "3"}],
    )
    _write_machine_json(judge_gate.MACHINE_PATH, {0: 3, 1: 2})

    with pytest.raises(SystemExit) as exc_info:
        judge_gate.run_score()

    assert "uyuşmazlığı" in str(exc_info.value)
    assert not judge_gate.RESULT_PATH.exists()


# --- run_score: elle yazılmış human_score doğrulaması (Bulgu 3) ------------


def test_run_score_reports_all_bad_human_score_rows_at_once(tmp_path, monkeypatch):
    _patch_paths(monkeypatch, tmp_path)
    _write_labels_csv(
        judge_gate.LABELS_PATH,
        [
            {"idx": 0, "role": "pirate", "question": "q0", "answer": "a0", "human_score": "3."},
            {"idx": 1, "role": "sage", "question": "q1", "answer": "a1", "human_score": "n/a"},
            {"idx": 2, "role": "ghost", "question": "q2", "answer": "a2", "human_score": "9"},
            {
                "idx": 3,
                "role": "engineer",
                "question": "q3",
                "answer": "a3",
                "human_score": "2",
            },
        ],
    )
    _write_machine_json(judge_gate.MACHINE_PATH, {0: 3, 1: 1, 2: 0, 3: 2})

    with pytest.raises(SystemExit) as exc_info:
        judge_gate.run_score()

    message = str(exc_info.value)
    assert "idx=0" in message
    assert "idx=1" in message
    assert "idx=2" in message
    assert "idx=3" not in message, "geçerli satır hata listesine girmemeli"
    assert not judge_gate.RESULT_PATH.exists(), "bozuk satır varken artifact yazılmamalı"


# --- run_score: boş satırlar ve mutlu yol -----------------------------------


def test_run_score_skips_blank_human_score_rows(tmp_path, monkeypatch):
    _patch_paths(monkeypatch, tmp_path)
    _write_labels_csv(
        judge_gate.LABELS_PATH,
        [
            {"idx": 0, "role": "pirate", "question": "q0", "answer": "a0", "human_score": "3"},
            {"idx": 1, "role": "sage", "question": "q1", "answer": "a1", "human_score": ""},
        ],
    )
    _write_machine_json(judge_gate.MACHINE_PATH, {0: 3, 1: 1})

    exit_code = judge_gate.run_score()

    assert exit_code == 0
    result = json.loads(judge_gate.RESULT_PATH.read_text(encoding="utf-8"))
    assert result["n"] == 1


def test_run_score_all_blank_raises_systemexit(tmp_path, monkeypatch):
    _patch_paths(monkeypatch, tmp_path)
    _write_labels_csv(
        judge_gate.LABELS_PATH,
        [{"idx": 0, "role": "pirate", "question": "q0", "answer": "a0", "human_score": ""}],
    )
    _write_machine_json(judge_gate.MACHINE_PATH, {0: 3})

    with pytest.raises(SystemExit, match="Hiç human_score doldurulmamış"):
        judge_gate.run_score()


def test_run_score_computes_agreement_and_writes_result(tmp_path, monkeypatch, capsys):
    _patch_paths(monkeypatch, tmp_path)
    _write_labels_csv(
        judge_gate.LABELS_PATH,
        [
            # makine 0, insan 1 -> ikisi de "no", UYUŞUR
            {"idx": 0, "role": "pirate", "question": "q0", "answer": "a0", "human_score": "1"},
            # makine 3, insan 2 -> UYUŞMAZ
            {"idx": 1, "role": "sage", "question": "q1", "answer": "a1", "human_score": "2"},
        ],
    )
    _write_machine_json(judge_gate.MACHINE_PATH, {0: 0, 1: 3})

    exit_code = judge_gate.run_score()

    assert exit_code == 1  # %50 < %75 eşik
    result = json.loads(judge_gate.RESULT_PATH.read_text(encoding="utf-8"))
    assert result["n"] == 2
    assert result["agreement"] == 0.5
    assert result["threshold"] == 0.75
    assert result["passed"] is False
    assert result["pairs"] == [
        {"idx": 0, "role": "pirate", "machine": 0, "human": 1},
        {"idx": 1, "role": "sage", "machine": 3, "human": 2},
    ]
    err = capsys.readouterr().err
    assert "KAPI KAPALI" in err


def test_run_score_gate_open_at_or_above_threshold(tmp_path, monkeypatch, capsys):
    _patch_paths(monkeypatch, tmp_path)
    rows = [
        {"idx": i, "role": "pirate", "question": f"q{i}", "answer": f"a{i}", "human_score": "3"}
        for i in range(4)
    ]
    _write_labels_csv(judge_gate.LABELS_PATH, rows)
    _write_machine_json(judge_gate.MACHINE_PATH, {i: 3 for i in range(4)})

    exit_code = judge_gate.run_score()

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "KAPI AÇIK" in out


# --- main(): eksik anahtar tanısı, --machine/--score ayrımı (Bulgu 2) ------


def test_main_reports_missing_api_key_without_traceback(monkeypatch, capsys):
    """Operatörün en olası hatası: iki ayrı kabuk çağrısının ikincisinde
    APP_KEY_JAILBREAK export edilmeyi unutulmuş."""

    def patlayan():
        raise RuntimeError(
            "APP_KEY_JAILBREAK ortam değişkeni tanımlı değil. "
            "Dağıtım ortamınızın .env dosyasından alıp kabuğunuzda export edin."
        )

    monkeypatch.setattr(judge_gate, "build_default_client", patlayan)

    exit_code = judge_gate.main(["--machine"])

    assert exit_code == 2
    err = capsys.readouterr().err
    assert "BAŞARISIZ" in err
    assert "gateway istemcisi kurulamadı" in err
    assert "APP_KEY_JAILBREAK" in err
    assert "Traceback" not in err
    assert "RuntimeError" not in err


@pytest.mark.parametrize(
    "istisna",
    [
        BudgetExceeded("'stage05_judge_gate' aşama bütçesi doldu: 15/15"),
        CircuitOpen("devre kesici açık"),
        GatewayError("HTTP 500"),
    ],
)
def test_main_budget_and_circuit_subclasses_are_not_swallowed_by_client_setup(
    tmp_path, monkeypatch, capsys, istisna
):
    """`BudgetExceeded`/`CircuitOpen`/`GatewayError` `RuntimeError` alt
    sınıflarıdır ama istemci KURULUMUNDA değil `chat()` sırasında oluşurlar —
    `main()`'in dar `except RuntimeError` bloğu istemci kurulumuna özel
    olmalı, `run_machine`'in kendi except zincirini ezmemeli."""
    _patch_paths(monkeypatch, tmp_path)
    _write_pilot(judge_gate.PILOT_PATH, [{"role": "pirate", "question": "q1", "answer": "a1"}])
    client = FakeJudgeClient([istisna])
    monkeypatch.setattr(judge_gate, "build_default_client", lambda: client)

    exit_code = judge_gate.main(["--machine"])

    assert exit_code == 2
    err = capsys.readouterr().err
    assert "BAŞARISIZ" in err or "DURDURULDU" in err


def test_main_score_does_not_build_a_gateway_client(tmp_path, monkeypatch):
    """`--score` ağa hiç çıkmamalı — istemci kurulmaya çalışılırsa patlar."""
    _patch_paths(monkeypatch, tmp_path)
    _write_labels_csv(
        judge_gate.LABELS_PATH,
        [{"idx": 0, "role": "pirate", "question": "q0", "answer": "a0", "human_score": "3"}],
    )
    _write_machine_json(judge_gate.MACHINE_PATH, {0: 3})

    def patlar():
        raise AssertionError("--score gateway istemcisi kurmamalı")

    monkeypatch.setattr(judge_gate, "build_default_client", patlar)

    exit_code = judge_gate.main(["--score"])

    assert exit_code == 0


def test_main_machine_happy_path_end_to_end(tmp_path, monkeypatch, capsys):
    _patch_paths(monkeypatch, tmp_path)
    _write_pilot(judge_gate.PILOT_PATH, [{"role": "pirate", "question": "q1", "answer": "a1"}])
    client = FakeJudgeClient(["[3]"])
    monkeypatch.setattr(judge_gate, "build_default_client", lambda: client)

    exit_code = judge_gate.main(["--machine"])

    assert exit_code == 0
    assert judge_gate.LABELS_PATH.exists()
    assert judge_gate.MACHINE_PATH.exists()
