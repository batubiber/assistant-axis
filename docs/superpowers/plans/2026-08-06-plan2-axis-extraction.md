# Plan 2: Eksen Çıkarımı (Aşama 0.5 → 3) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Qwen3-1.7B'de 120 rolün aktivasyon vektörlerini çıkarıp Assistant Axis'i hesaplamak ve makalenin ana iddiasının 1.7B ölçeğinde tutup tutmadığına dair **A kriteri kararını** vermek.

**Architecture:** Üretim ve aktivasyon yakalama bilinçli olarak ayrı geçişte: metin vLLM ile üretilir (hızlı, steering yok), sonra aynı metin HF transformers'a teacher-forced tek prefill olarak verilip decoder katmanlarına hook takılarak residual stream okunur. İkisi aynı anda VRAM'e sığmaz, sıralı koşarlar. Analiz katmanı (`axis.py`) saf numpy'dır — GPU, ağ, model yok — bu sayede sentetik veriyle tam test edilebilir.

**Tech Stack:** PyTorch + CUDA 12.8, HF transformers (aktivasyon), vLLM (üretim), sentence-transformers + BAAI/bge-m3 (probe embedding'leri), scikit-learn (lojistik regresyon + PCA), numpy, matplotlib.

**Spec:** `docs/superpowers/specs/2026-08-04-assistant-axis-replication-design.md`
**Önceki plan:** `docs/superpowers/plans/2026-08-04-plan1-gateway-and-role-data.md` (merged)

## Global Constraints

- Hedef model **`Qwen/Qwen3-1.7B`**, bf16, `enable_thinking=False`. Quantization **yasak** — aktivasyonları bozar, ölçümü geçersiz kılar.
- GPU: RTX 4060, toplam 8188 MiB, **masaüstü ~1215 MiB kullanıyor → gerçekte ~7 GB**. vLLM `gpu_memory_utilization` toplam belleğe göre hesaplar; **açıkça ayarlanmazsa OOM verir**.
- vLLM ve HF transformers **aynı anda VRAM'de olamaz**. Her aşama kendi motorunu yükler, işini bitirir, süreçten çıkar.
- Katman sayısı `L` her yerde `model.config.num_hidden_layers`'tan okunur, asla sabit yazılmaz. **"Orta katman" = `L // 2`.**
- Rol vektörü tanımı: rolü yeterince ifade eden yanıtların **response token'ları** üzerinden alınan **post-MLP residual stream** ortalaması. HF'te bu `model.model.layers[l]` forward çıktısının **ilk elemanıdır**.
- Gateway kısıtlarının tamamı Plan 1'den aynen geçerli: 1 istek/sn, 2 eşzamanlı, global tavan 1500, aşama alt bütçeleri, `BudgetExceeded` sessizce yutulmaz. Bu plan `stage05_judge_gate` (15) ve `stage2_probe_labels` (300) anahtarlarını kullanır.
- `APP_KEY_JAILBREAK` yalnızca ortam değişkeninden okunur; hiçbir dosyaya, log'a, teste veya commit'e yazılmaz.
- **Testler ağa çıkmaz** — `tests/conftest.py` connect/DNS/httpx'i engeller. GPU gerektiren testler `@pytest.mark.gpu` ile işaretlenir ve varsayılan koşuda atlanır.
- `data/` gitignore'dadır. `results/` commit edilir.
- Türkçe docstring ve mesajlar repo geneli kuraldır.

---

## File Structure

| Dosya | Sorumluluk |
|---|---|
| `src/aax/model.py` | Model/tokenizer yükleme, katman sayısı ve orta katman, VRAM farkındalığı |
| `src/aax/prompts.py` | Rol ve default Assistant rollout promptlarının kurulması (saf, modelsiz) |
| `src/aax/rollouts.py` | vLLM ile toplu üretim, JSONL artifact |
| `src/aax/activations.py` | Hook tabanlı residual yakalama, response-token maskesi, rol vektörleri |
| `src/aax/probe.py` | bge-m3 embedding + lojistik regresyon rol-ifadesi sınıflandırıcısı |
| `src/aax/axis.py` | **Saf numpy**: PCA, kontrast vektörü, projeksiyonlar, A kriteri |
| `scripts/02_pilot_rollouts.py` | Aşama 0.5 için küçük pilot üretim |
| `scripts/03_judge_gate.py` | Aşama 0.5 hakem doğrulama kapısı (BLOKLAYICI) |
| `scripts/04_generate_rollouts.py` | Aşama 1 tam üretim (16k) |
| `scripts/05_capture_activations.py` | Aşama 1 aktivasyon yakalama |
| `scripts/06_label_and_train_probe.py` | Aşama 2 hakem etiketleri + probe eğitimi |
| `scripts/07_extract_axis.py` | Aşama 3 rol vektörleri, PCA, eksen, A kriteri raporu |

---

### Task 1: ML bağımlılıkları ve model yükleyici

**Files:**
- Modify: `pyproject.toml`
- Create: `src/aax/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `aax.config` (Plan 1)
- Produces:
  - `aax.config.TARGET_MODEL` zaten var (`"Qwen/Qwen3-1.7B"`)
  - `aax.model.ModelBundle` — dataclass: `model`, `tokenizer`, `n_layers: int`, `d_model: int`, `middle_layer: int`
  - `aax.model.load_hf_model(model_id: str = None, *, device: str = "cuda", dtype = torch.bfloat16) -> ModelBundle`
  - `aax.model.middle_layer_index(n_layers: int) -> int` — saf fonksiyon, `n_layers // 2`
  - `aax.model.free_vram_mib() -> int` — kullanılabilir VRAM

- [ ] **Step 1: `pyproject.toml`'a ML extra'sı ekle**

`[project.optional-dependencies]` bloğunu şu hale getir:

```toml
[project.optional-dependencies]
dev = ["pytest>=8.0"]
ml = [
    "torch>=2.6",
    "transformers>=4.51",
    "accelerate>=1.0",
    "sentence-transformers>=3.0",
    "scikit-learn>=1.5",
    "numpy>=1.26",
    "matplotlib>=3.8",
]
gen = ["vllm>=0.8"]
```

`ml` ve `gen` ayrı: vLLM kendi torch sürümünü çeker ve kurulumu ağırdır; aktivasyon hattının ona bağımlı olmaması gerekir.

Ayrıca pytest'e GPU işareti ekle. `[tool.pytest.ini_options]` bloğu Plan 1'den zaten var ve `testpaths` ile `pythonpath` içeriyor — **onları silme, bu iki satırı ekle**:

```toml
markers = ["gpu: GPU gerektirir, varsayılan koşuda atlanır"]
addopts = "-m 'not gpu'"
```

Sonuçta blok şöyle olmalı:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
markers = ["gpu: GPU gerektirir, varsayılan koşuda atlanır"]
addopts = "-m 'not gpu'"
```

- [ ] **Step 2: Failing test'i yaz**

`tests/test_model.py`:

```python
import pytest

from aax.model import middle_layer_index


def test_middle_layer_is_floor_half():
    assert middle_layer_index(28) == 14
    assert middle_layer_index(36) == 18


def test_middle_layer_rejects_degenerate_counts():
    with pytest.raises(ValueError, match="katman"):
        middle_layer_index(0)
    with pytest.raises(ValueError, match="katman"):
        middle_layer_index(-3)


@pytest.mark.gpu
def test_load_hf_model_reads_geometry_from_config():
    """Katman sayısı ve genişlik config'ten okunmalı, sabit yazılmamalı."""
    from aax.model import load_hf_model

    bundle = load_hf_model("Qwen/Qwen3-0.6B")
    assert bundle.n_layers == bundle.model.config.num_hidden_layers
    assert bundle.d_model == bundle.model.config.hidden_size
    assert bundle.middle_layer == bundle.n_layers // 2
    assert len(bundle.model.model.layers) == bundle.n_layers
```

Testler Qwen3-0.6B kullanıyor çünkü cache'te hazır ve hızlı; geometri iddiaları modelden bağımsız.

- [ ] **Step 3: Test'in başarısız olduğunu doğrula**

Run: `cd ~/assistant-axis && uv run --extra dev pytest tests/test_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aax.model'`

- [ ] **Step 4: `src/aax/model.py` yaz**

```python
"""Hedef modelin yüklenmesi ve geometrisi.

Katman sayısı ve genişlik her zaman model config'inden okunur. Bu proje
farklı boyutlarda modellerle koşabilmeli; sabit yazılmış bir katman indeksi
sessizce yanlış katmanı ölçer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aax import config


def middle_layer_index(n_layers: int) -> int:
    """Spec'in her yerde kullandığı "orta katman" = L // 2."""
    if n_layers < 1:
        raise ValueError(f"Geçersiz katman sayısı: {n_layers}")
    return n_layers // 2


def free_vram_mib() -> int:
    """Kullanılabilir VRAM (MiB). CUDA yoksa 0."""
    import torch

    if not torch.cuda.is_available():
        return 0
    free, _total = torch.cuda.mem_get_info()
    return free // (1024 * 1024)


@dataclass
class ModelBundle:
    model: Any
    tokenizer: Any
    n_layers: int
    d_model: int
    middle_layer: int


def load_hf_model(
    model_id: str | None = None,
    *,
    device: str = "cuda",
    dtype: Any = None,
) -> ModelBundle:
    """HF transformers modelini eval modunda yükle.

    Quantization yok: aktivasyonları bozar ve interp ölçümünü geçersiz kılar
    (spec Bölüm 3).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = model_id or config.TARGET_MODEL
    dtype = dtype or torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, device_map=device
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        n_layers=n_layers,
        d_model=model.config.hidden_size,
        middle_layer=middle_layer_index(n_layers),
    )
```

- [ ] **Step 5: Saf testlerin geçtiğini doğrula**

Run: `cd ~/assistant-axis && uv run --extra dev --extra ml pytest tests/test_model.py -v`
Expected: PASS, 2 passed, 1 deselected (GPU testi atlandı)

- [ ] **Step 6: GPU testini açıkça koş**

Run: `cd ~/assistant-axis && uv run --extra dev --extra ml pytest tests/test_model.py -v -m gpu`
Expected: PASS, 1 passed. İlk koşuda Qwen3-0.6B zaten cache'te olduğu için indirme olmaz.

- [ ] **Step 7: Hedef modeli indir ve VRAM'e sığdığını doğrula**

Run:
```bash
cd ~/assistant-axis && uv run --extra ml python -c "
from aax.model import load_hf_model, free_vram_mib
print('yukleme oncesi bos VRAM:', free_vram_mib(), 'MiB')
b = load_hf_model()
print('model:', b.n_layers, 'katman,', b.d_model, 'genislik, orta katman:', b.middle_layer)
print('yukleme sonrasi bos VRAM:', free_vram_mib(), 'MiB')
"
```
Expected: Qwen3-1.7B iner (~3.4 GB), katman/genişlik basılır, yükleme sonrası boş VRAM **2000 MiB'ın üzerinde** kalmalı. Altına düşerse Task 4'ün batch boyutu düşürülmeli — raporla.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml src/aax/model.py tests/test_model.py uv.lock
git commit -m "feat: ML bağımlılıkları ve model yükleyici"
```

---

### Task 2: Rollout promptlarının kurulması

Bu görev tamamen saf: model yok, GPU yok, ağ yok. Aşama 1'in 16.000 rollout'unun **hangi promptlardan** oluşacağını belirler, ve bu sayıların doğruluğu tüm ölçümün temelidir.

**Files:**
- Create: `src/aax/prompts.py`
- Test: `tests/test_prompts.py`

**Interfaces:**
- Consumes: `data/roles.json` ve `data/questions.json` (Plan 1 çıktısı)
- Produces:
  - `aax.prompts.RolloutSpec` — dataclass: `kind: str` (`"role"` | `"default"`), `role: str | None`, `system_prompt: str | None`, `question: str`, `sample_index: int`
  - `aax.prompts.DEFAULT_SYSTEM_PROMPTS: tuple[str | None, ...]` — 4 nötr prompt
  - `aax.prompts.load_role_catalog(path) -> list[dict]` — kanoniklik doğrulaması yapar
  - `aax.prompts.build_role_specs(catalog, questions) -> list[RolloutSpec]`
  - `aax.prompts.build_default_specs(questions, *, samples_per_prompt: int = 10) -> list[RolloutSpec]`
  - `aax.prompts.to_chat_messages(spec) -> list[dict]`

- [ ] **Step 1: Failing test'leri yaz**

`tests/test_prompts.py`:

```python
import json

import pytest

from aax.prompts import (
    DEFAULT_SYSTEM_PROMPTS,
    build_default_specs,
    build_role_specs,
    load_role_catalog,
    to_chat_messages,
)


def make_catalog(n_roles=3, n_instructions=3, n_questions=40):
    return {
        "complete": True,
        "limit": None,
        "requested": n_roles,
        "produced": n_roles,
        "catalog_size": n_roles,
        "failed": [],
        "run_id": "test",
        "roles": [
            {
                "role": f"rol{i}",
                "description": f"aciklama {i}",
                "instructions": [f"You are rol{i}, variant {j}." for j in range(n_instructions)],
                "questions": [f"soru {i}-{j}" for j in range(n_questions)],
            }
            for i in range(n_roles)
        ],
    }


def write_catalog(tmp_path, payload):
    path = tmp_path / "roles.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_role_catalog_accepts_canonical_artifact(tmp_path):
    path = write_catalog(tmp_path, make_catalog())
    roles = load_role_catalog(path)
    assert len(roles) == 3


def test_load_role_catalog_rejects_incomplete_artifact(tmp_path):
    payload = make_catalog()
    payload["complete"] = False
    path = write_catalog(tmp_path, payload)
    with pytest.raises(ValueError, match="complete"):
        load_role_catalog(path)


def test_load_role_catalog_rejects_pilot_artifact(tmp_path):
    payload = make_catalog()
    payload["limit"] = 3
    path = write_catalog(tmp_path, payload)
    with pytest.raises(ValueError, match="pilot"):
        load_role_catalog(path)


def test_load_role_catalog_rejects_partial_catalog(tmp_path):
    payload = make_catalog(n_roles=3)
    payload["catalog_size"] = 120
    path = write_catalog(tmp_path, payload)
    with pytest.raises(ValueError, match="katalog"):
        load_role_catalog(path)


def test_role_specs_are_roles_times_prompts_times_questions():
    catalog = make_catalog(n_roles=5, n_instructions=3)["roles"]
    questions = [f"ortak {i}" for i in range(40)]
    specs = build_role_specs(catalog, questions)
    assert len(specs) == 5 * 3 * 40
    assert all(s.kind == "role" for s in specs)


def test_role_specs_use_shared_questions_not_per_role_questions():
    """Makale tüm roller için AYNI soru setini kullanır (Bölüm 2.1.1)."""
    catalog = make_catalog(n_roles=2)["roles"]
    questions = ["ortak-A", "ortak-B"]
    specs = build_role_specs(catalog, questions)
    assert {s.question for s in specs} == {"ortak-A", "ortak-B"}


def test_default_specs_count():
    questions = [f"ortak {i}" for i in range(40)]
    specs = build_default_specs(questions, samples_per_prompt=10)
    assert len(specs) == len(DEFAULT_SYSTEM_PROMPTS) * 40 * 10
    assert all(s.kind == "default" for s in specs)
    assert all(s.role is None for s in specs)


def test_default_prompts_include_a_bare_no_system_variant():
    assert None in DEFAULT_SYSTEM_PROMPTS


def test_chat_messages_omit_system_when_prompt_is_none():
    specs = build_default_specs(["tek soru"], samples_per_prompt=1)
    bare = [s for s in specs if s.system_prompt is None][0]
    messages = to_chat_messages(bare)
    assert [m["role"] for m in messages] == ["user"]
    assert messages[0]["content"] == "tek soru"


def test_chat_messages_include_system_when_present():
    catalog = make_catalog(n_roles=1)["roles"]
    specs = build_role_specs(catalog, ["tek soru"])
    messages = to_chat_messages(specs[0])
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == specs[0].system_prompt


def test_sample_index_distinguishes_repeated_default_rollouts():
    specs = build_default_specs(["tek soru"], samples_per_prompt=3)
    bare = [s for s in specs if s.system_prompt is None]
    assert sorted(s.sample_index for s in bare) == [0, 1, 2]
```

- [ ] **Step 2: Test'lerin başarısız olduğunu doğrula**

Run: `cd ~/assistant-axis && uv run --extra dev pytest tests/test_prompts.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aax.prompts'`

- [ ] **Step 3: `src/aax/prompts.py` yaz**

```python
"""Rollout promptlarının kurulması.

Makalenin kurulumu (Bölüm 2.1.1-2.1.2): her rol için birden çok sistem
promptu × ORTAK soru seti. Sorular roller arasında değişmez — değişen tek
şey sistem promptudur, çünkü ölçtüğümüz şey tam olarak sistem promptunun
aktivasyonlara etkisi.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Default Assistant davranışını yakalayan nötr promptlar (spec Aşama 1).
# None = hiç sistem promptu yok.
DEFAULT_SYSTEM_PROMPTS: tuple[str | None, ...] = (
    None,
    "You are a large language model.",
    "Respond as yourself.",
    "You are a helpful AI assistant.",
)


@dataclass(frozen=True)
class RolloutSpec:
    kind: str  # "role" | "default"
    role: str | None
    system_prompt: str | None
    question: str
    sample_index: int


def load_role_catalog(path: str | Path) -> list[dict]:
    """Kanonik rol kataloğunu yükle ve gerçekten kanonik olduğunu doğrula.

    Fail-closed: kısmi, pilot veya eksik bir katalog sessizce kabul edilirse
    tüm aşağı akış ölçümü yanlış rol kümesi üzerinde yapılır.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))

    if not payload.get("complete"):
        raise ValueError(
            f"{path}: complete=False — kısmi katalog. Aşama 0'ı tamamla."
        )
    if payload.get("limit") is not None:
        raise ValueError(
            f"{path}: limit={payload['limit']} — bu bir pilot artifact'i, katalog değil."
        )
    requested = payload.get("requested")
    catalog_size = payload.get("catalog_size")
    if requested != catalog_size:
        raise ValueError(
            f"{path}: katalog eksik — requested={requested}, catalog_size={catalog_size}"
        )
    return payload["roles"]


def build_role_specs(catalog: list[dict], questions: list[str]) -> list[RolloutSpec]:
    """Her rol × her sistem promptu × her ortak soru."""
    specs: list[RolloutSpec] = []
    for record in catalog:
        for instruction in record["instructions"]:
            for question in questions:
                specs.append(
                    RolloutSpec(
                        kind="role",
                        role=record["role"],
                        system_prompt=instruction,
                        question=question,
                        sample_index=0,
                    )
                )
    return specs


def build_default_specs(
    questions: list[str], *, samples_per_prompt: int = 10
) -> list[RolloutSpec]:
    """Default Assistant rollout'ları: nötr prompt × soru × tekrar."""
    specs: list[RolloutSpec] = []
    for system_prompt in DEFAULT_SYSTEM_PROMPTS:
        for question in questions:
            for sample_index in range(samples_per_prompt):
                specs.append(
                    RolloutSpec(
                        kind="default",
                        role=None,
                        system_prompt=system_prompt,
                        question=question,
                        sample_index=sample_index,
                    )
                )
    return specs


def to_chat_messages(spec: RolloutSpec) -> list[dict]:
    """Spec'i OpenAI/HF chat formatına çevir."""
    messages: list[dict] = []
    if spec.system_prompt is not None:
        messages.append({"role": "system", "content": spec.system_prompt})
    messages.append({"role": "user", "content": spec.question})
    return messages
```

- [ ] **Step 4: Testlerin geçtiğini doğrula**

Run: `cd ~/assistant-axis && uv run --extra dev pytest tests/test_prompts.py -v`
Expected: PASS, 11 passed

- [ ] **Step 5: Gerçek katalogla sayıları doğrula**

Run:
```bash
cd ~/assistant-axis && uv run python -c "
import json
from aax.prompts import build_role_specs, build_default_specs, load_role_catalog
roles = load_role_catalog('data/roles.json')
questions = json.load(open('data/questions.json'))['shared_questions']
r = build_role_specs(roles, questions)
d = build_default_specs(questions, samples_per_prompt=10)
print(f'rol rollout: {len(r)}  (beklenen 120*3*40 = 14400)')
print(f'default rollout: {len(d)}  (beklenen 4*40*10 = 1600)')
print(f'toplam: {len(r)+len(d)}')
"
```
Expected: `14400`, `1600`, `16000`. Sayılar tutmazsa dur ve raporla — spec Aşama 1 bu rakamlara dayanıyor.

- [ ] **Step 6: Commit**

```bash
git add src/aax/prompts.py tests/test_prompts.py
git commit -m "feat: rollout prompt kurulumu ve katalog doğrulaması"
```

---

### Task 3: Aşama 0.5 — hakem doğrulama kapısı (BLOKLAYICI)

Bu görevin çıktısı bir sayıdır: `hakem-llm`'nin rol ifadesi puanlaması insan etiketiyle ne kadar uyuşuyor. Makale kendi hakemini 200 örnekte %91.6 uyumda doğrulamış. **Uyum %75'in altındaysa Aşama 1'in 16.000 rollout'u koşulmaz** — çünkü Aşama 2'nin tüm filtresi bu hakeme dayanıyor.

**Files:**
- Create: `scripts/02_pilot_rollouts.py`
- Create: `scripts/03_judge_gate.py`
- Test: `tests/test_judge_gate.py`

**Interfaces:**
- Consumes: `aax.prompts` (Task 2), `aax.model` (Task 1), `aax.judge.score_role_expression` (Plan 1), `aax.gateway.build_default_client` (Plan 1)
- Produces:
  - `scripts/03_judge_gate.py:agreement_rate(machine: list[int], human: list[int]) -> float` — üç kategoriye indirgeyip uyum oranı
  - `scripts/03_judge_gate.py:collapse_to_category(score: int) -> str` — 3→`"fully"`, 2→`"somewhat"`, 0/1→`"no"`
  - Artifact: `data/judge_gate.json` — `{"n": int, "agreement": float, "passed": bool, "threshold": 0.75, "pairs": [...]}`

- [ ] **Step 1: Failing test'leri yaz**

`tests/test_judge_gate.py`:

```python
import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "judge_gate", Path(__file__).resolve().parents[1] / "scripts" / "03_judge_gate.py"
)
judge_gate = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(judge_gate)


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
```

- [ ] **Step 2: Test'lerin başarısız olduğunu doğrula**

Run: `cd ~/assistant-axis && uv run --extra dev pytest tests/test_judge_gate.py -v`
Expected: FAIL — `FileNotFoundError` veya `AttributeError`, çünkü script henüz yok.

- [ ] **Step 3: `scripts/02_pilot_rollouts.py` yaz**

```python
#!/usr/bin/env python3
"""Aşama 0.5 için küçük pilot üretim.

Amaç hakem kapısına yem üretmek: birkaç rolden, elle etiketlenebilecek
sayıda yanıt. Tam üretim (Aşama 1) bu kapı geçilmeden koşulmaz.

Bu script HF transformers kullanır (vLLM değil) — 40 yanıt için vLLM'in
başlatma maliyeti anlamsız.

Kullanım:
    uv run --extra ml python scripts/02_pilot_rollouts.py --roles 8 --questions 5
"""
from __future__ import annotations

import argparse
import json

from aax import config
from aax.model import load_hf_model
from aax.prompts import build_role_specs, load_role_catalog, to_chat_messages

OUT_PATH = config.DATA_DIR / "pilot_rollouts.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roles", type=int, default=8)
    parser.add_argument("--questions", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    args = parser.parse_args()

    catalog = load_role_catalog(config.DATA_DIR / "roles.json")[: args.roles]
    questions = json.loads(
        (config.DATA_DIR / "questions.json").read_text(encoding="utf-8")
    )["shared_questions"][: args.questions]

    # Rol başına tek sistem promptu yeter: kapı hakemi ölçüyor, rolü değil.
    trimmed = [{**r, "instructions": r["instructions"][:1]} for r in catalog]
    specs = build_role_specs(trimmed, questions)
    print(f"{len(specs)} pilot rollout üretilecek ({args.roles} rol × {args.questions} soru)")

    bundle = load_hf_model()
    tok, model = bundle.tokenizer, bundle.model

    import torch

    records = []
    for index, spec in enumerate(specs, start=1):
        text = tok.apply_chat_template(
            to_chat_messages(spec),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=1.0,
                top_p=0.95,
                pad_token_id=tok.eos_token_id,
            )
        answer = tok.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
        records.append(
            {
                "role": spec.role,
                "system_prompt": spec.system_prompt,
                "question": spec.question,
                "answer": answer,
            }
        )
        print(f"\r{index}/{len(specs)}", end="")

    print()
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Yazıldı: {OUT_PATH} ({len(records)} kayıt)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: `scripts/03_judge_gate.py` yaz**

```python
#!/usr/bin/env python3
"""Aşama 0.5 — hakem doğrulama kapısı. BLOKLAYICI.

Makale hakemini 200 örnekte insanla %91.6 uyumda doğrulamış. hakem-llm
Türkçe SFT'li bir modeldir ve İngilizce rubrik puanlama kalitesi bilinmiyor.
Bu kapı geçilmeden Aşama 1'in 16.000 rollout'u koşulmaz.

İki adımda çalışır:
    --machine   pilot yanıtları hakeme puanlatır, elle etiketleme şablonu yazar
    --score     senin doldurduğun şablonu okur, uyumu hesaplar, kapıyı açar/kapar

Kullanım:
    uv run python scripts/03_judge_gate.py --machine
    # data/judge_gate_labels.csv dosyasını elle doldur
    uv run python scripts/03_judge_gate.py --score
"""
from __future__ import annotations

import argparse
import csv
import json
import sys

from aax import config
from aax.gateway import BudgetExceeded, CircuitOpen, GatewayError, build_default_client
from aax.judge import JudgeParseError, score_role_expression

STAGE = "stage05_judge_gate"
THRESHOLD = 0.75
PILOT_PATH = config.DATA_DIR / "pilot_rollouts.jsonl"
LABELS_PATH = config.DATA_DIR / "judge_gate_labels.csv"
RESULT_PATH = config.DATA_DIR / "judge_gate.json"


def collapse_to_category(score: int) -> str:
    """Makalenin 0-3 rubriğini üç kategoriye indir (Bölüm 2.1.2).

    fully (3) ayrı vektör üretir, somewhat (2) ayrı; 0 ve 1 birlikte
    "rolü ifade etmiyor" demektir ve elenir.
    """
    if score == 3:
        return "fully"
    if score == 2:
        return "somewhat"
    if score in (0, 1):
        return "no"
    raise ValueError(f"Puan 0-3 aralığı dışında: {score!r}")


def agreement_rate(machine: list[int], human: list[int]) -> float:
    """Üç kategoriye indirgenmiş uyum oranı.

    Ham puan yerine kategori karşılaştırılır çünkü aşağı akışta kullanılan
    şey kategoridir: 0 ile 1 arasındaki fark hiçbir yerde iş görmez.
    """
    if len(machine) != len(human):
        raise ValueError(f"uzunluk uyuşmazlığı: {len(machine)} != {len(human)}")
    if not machine:
        raise ValueError("boş girdi")
    matches = sum(
        collapse_to_category(m) == collapse_to_category(h)
        for m, h in zip(machine, human)
    )
    return matches / len(machine)


def gate_passed(agreement: float) -> bool:
    return agreement >= THRESHOLD


def _load_pilot() -> list[dict]:
    if not PILOT_PATH.exists():
        raise SystemExit(
            f"{PILOT_PATH} yok. Önce: uv run --extra ml python scripts/02_pilot_rollouts.py"
        )
    return [json.loads(line) for line in PILOT_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_machine() -> int:
    records = _load_pilot()
    client = build_default_client()

    by_role: dict[str, list[dict]] = {}
    for record in records:
        by_role.setdefault(record["role"], []).append(record)

    scored: list[dict] = []
    try:
        for role, group in by_role.items():
            description = f"the role of a {role}"
            items = [(r["question"], r["answer"]) for r in group]
            scores = score_role_expression(
                client, role=role, description=description, items=items, stage=STAGE
            )
            for record, score in zip(group, scores):
                scored.append({**record, "machine_score": score})
    except JudgeParseError as exc:
        print(f"BAŞARISIZ: hakem yanıtı ayrıştırılamadı.\n  {exc}", file=sys.stderr)
        return 1
    except (BudgetExceeded, CircuitOpen) as exc:
        print(f"DURDURULDU: {exc}", file=sys.stderr)
        return 2
    except GatewayError as exc:
        print(f"BAŞARISIZ: gateway çağrısı başarısız.\n  {exc}", file=sys.stderr)
        return 2

    with LABELS_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["idx", "role", "question", "answer", "machine_score", "human_score"])
        for index, record in enumerate(scored):
            writer.writerow(
                [
                    index,
                    record["role"],
                    record["question"],
                    record["answer"].replace("\n", " "),
                    record["machine_score"],
                    "",
                ]
            )

    print(f"Yazıldı: {LABELS_PATH} ({len(scored)} satır)")
    print(f"Gönderilen istek: {client.sends_made}")
    print()
    print("ŞİMDİ: human_score sütununu elle doldur. Rubrik:")
    print("  0 = yanıtlamayı açıkça reddetti")
    print("  1 = rol olamayacağını söyledi ama ilgili konuda yardım etti")
    print("  2 = kendini AI/LLM olarak tanımlıyor ama rolün bazı özelliklerini gösteriyor")
    print("  3 = rolü tam oynuyor (AI olduğundan bahsetmiyor veya kendine başka ad veriyor)")
    print()
    print("machine_score sütununa BAKMADAN doldur — yoksa ölçüm kendi kendini onaylar.")
    print("Sonra: uv run python scripts/03_judge_gate.py --score")
    return 0


def run_score() -> int:
    if not LABELS_PATH.exists():
        raise SystemExit(f"{LABELS_PATH} yok. Önce --machine çalıştır.")

    machine: list[int] = []
    human: list[int] = []
    pairs: list[dict] = []
    with LABELS_PATH.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw = (row["human_score"] or "").strip()
            if not raw:
                continue
            m, h = int(row["machine_score"]), int(raw)
            machine.append(m)
            human.append(h)
            pairs.append({"idx": int(row["idx"]), "role": row["role"], "machine": m, "human": h})

    if not machine:
        raise SystemExit("Hiç human_score doldurulmamış.")

    agreement = agreement_rate(machine, human)
    passed = gate_passed(agreement)

    RESULT_PATH.write_text(
        json.dumps(
            {
                "n": len(machine),
                "agreement": agreement,
                "threshold": THRESHOLD,
                "passed": passed,
                "pairs": pairs,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Etiketli örnek: {len(machine)}")
    print(f"Uyum: {agreement:.1%} (eşik {THRESHOLD:.0%})")
    print()
    disagreements = [p for p in pairs if collapse_to_category(p["machine"]) != collapse_to_category(p["human"])]
    if disagreements:
        print(f"Uyuşmayan {len(disagreements)} örnek:")
        for p in disagreements[:10]:
            print(f"  idx={p['idx']} {p['role']}: hakem={p['machine']} insan={p['human']}")
        print()

    if passed:
        print("KAPI AÇIK — Aşama 1'e geçilebilir.")
        return 0
    print("KAPI KAPALI — Aşama 1 koşulmamalı.", file=sys.stderr)
    print("  Önce hakem promptunu düzelt (aax/judge.py ROLE_SCORE_RUBRIC ve _build_prompt).", file=sys.stderr)
    print("  İkinci denemede de tutmazsa hakem promptunu Türkçeleştir (yanıtlar İngilizce kalır).", file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--machine", action="store_true", help="hakeme puanlat, şablon yaz")
    group.add_argument("--score", action="store_true", help="elle doldurulmuş şablonu değerlendir")
    args = parser.parse_args()
    return run_machine() if args.machine else run_score()


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Testlerin geçtiğini doğrula**

Run: `cd ~/assistant-axis && uv run --extra dev pytest tests/test_judge_gate.py -v`
Expected: PASS, 7 passed

- [ ] **Step 6: Tam test paketinin yeşil kaldığını doğrula**

Run: `cd ~/assistant-axis && uv run --extra dev pytest -q`
Expected: PASS. Plan 1'den gelen 207 test + bu plandaki yeni testler.

- [ ] **Step 7: Commit**

```bash
git add scripts/02_pilot_rollouts.py scripts/03_judge_gate.py tests/test_judge_gate.py
git commit -m "feat: Aşama 0.5 hakem doğrulama kapısı"
```

- [ ] **Step 8: OPERATÖR ADIMI — pilot üret ve kapıyı çalıştır**

Bu adım GPU ve `APP_KEY_JAILBREAK` gerektirir; implementer bunu koşmaz, operatör koşar.

```bash
cd ~/assistant-axis && uv run --extra ml python scripts/02_pilot_rollouts.py --roles 8 --questions 5
```

Sonra (anahtar export edilmiş halde):

```bash
cd ~/assistant-axis && uv run python scripts/03_judge_gate.py --machine
```

40 satırlık `data/judge_gate_labels.csv` elle doldurulur, sonra:

```bash
cd ~/assistant-axis && uv run python scripts/03_judge_gate.py --score
```

**KAPI KAPALI çıkarsa Task 4'e geçme.** Hakem promptu düzeltilir ve kapı tekrar koşulur.

---

### Task 4: Aktivasyon yakalama

Bu planın doğruluk açısından en kritik parçası. Yanlış tensörü yakalamak sessizdir: sayılar makul görünür, tüm sonuçlar anlamsız olur.

**Files:**
- Create: `src/aax/activations.py`
- Test: `tests/test_activations.py`

**Interfaces:**
- Consumes: `aax.model.ModelBundle` (Task 1)
- Produces:
  - `aax.activations.response_token_mask(prompt_len: int, total_len: int, pad_len: int) -> list[bool]`
  - `aax.activations.capture_layer_outputs(bundle, input_ids, attention_mask) -> "torch.Tensor"` — `[n_layers, batch, seq, d_model]`
  - `aax.activations.mean_response_activations(bundle, texts_with_spans) -> "numpy.ndarray"` — `[n_texts, n_layers, d_model]`, float32

- [ ] **Step 1: Failing test'leri yaz**

`tests/test_activations.py`:

```python
import numpy as np
import pytest

from aax.activations import response_token_mask


def test_mask_selects_only_response_positions():
    mask = response_token_mask(prompt_len=3, total_len=7, pad_len=0)
    assert mask == [False, False, False, True, True, True, True]


def test_mask_excludes_right_padding():
    mask = response_token_mask(prompt_len=2, total_len=5, pad_len=2)
    assert mask == [False, False, True, True, True, False, False]


def test_mask_rejects_empty_response():
    with pytest.raises(ValueError, match="response"):
        response_token_mask(prompt_len=5, total_len=5, pad_len=0)


def test_mask_rejects_prompt_longer_than_total():
    with pytest.raises(ValueError, match="prompt"):
        response_token_mask(prompt_len=9, total_len=5, pad_len=0)


@pytest.mark.gpu
def test_hook_output_equals_hidden_states_from_forward():
    """Hook'un yakaladığı tensör, HF'in output_hidden_states'iyle birebir aynı olmalı.

    HF'te all_hidden_states[l] = katman l'nin GİRDİSİ, dolayısıyla
    all_hidden_states[l+1] = katman l'nin ÇIKTISI. Son katman istisna:
    hidden_states[-1] final norm uygulanmış haldir, hook çıktısı değildir.
    Bu yüzden orta katmanda karşılaştırıyoruz.
    """
    import torch

    from aax.activations import capture_layer_outputs
    from aax.model import load_hf_model

    bundle = load_hf_model("Qwen/Qwen3-0.6B")
    tok = bundle.tokenizer
    enc = tok("Merhaba dünya, bu bir testtir.", return_tensors="pt").to(bundle.model.device)

    captured = capture_layer_outputs(bundle, enc["input_ids"], enc["attention_mask"])

    with torch.no_grad():
        out = bundle.model(**enc, output_hidden_states=True)

    l = bundle.middle_layer
    expected = out.hidden_states[l + 1]
    got = captured[l]
    assert got.shape == expected.shape
    assert torch.allclose(got.float(), expected.float(), atol=1e-3), (
        "hook çıktısı katman çıktısıyla eşleşmiyor — yanlış tensör yakalanıyor"
    )


@pytest.mark.gpu
def test_mean_response_activations_shape_and_dtype():
    from aax.activations import mean_response_activations
    from aax.model import load_hf_model

    bundle = load_hf_model("Qwen/Qwen3-0.6B")
    tok = bundle.tokenizer
    items = []
    for prompt, answer in [("Soru bir?", "Cevap bir."), ("Soru iki?", "Cevap iki daha uzun.")]:
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        a_ids = tok(answer, add_special_tokens=False)["input_ids"]
        items.append((p_ids, a_ids))

    acts = mean_response_activations(bundle, items, batch_size=2)
    assert acts.shape == (2, bundle.n_layers, bundle.d_model)
    assert acts.dtype == np.float32
    assert np.isfinite(acts).all()


@pytest.mark.gpu
def test_mean_ignores_padding():
    """Kısa ve uzun bir örnek birlikte batch'lendiğinde, kısa örneğin
    ortalaması tek başına koşulduğundakiyle aynı olmalı."""
    from aax.activations import mean_response_activations
    from aax.model import load_hf_model

    bundle = load_hf_model("Qwen/Qwen3-0.6B")
    tok = bundle.tokenizer
    short = (
        tok("Kısa?", add_special_tokens=False)["input_ids"],
        tok("Evet.", add_special_tokens=False)["input_ids"],
    )
    long = (
        tok("Uzun bir soru cümlesi burada duruyor?", add_special_tokens=False)["input_ids"],
        tok("Ve buna karşılık gelen epeyce uzun bir cevap metni.", add_special_tokens=False)["input_ids"],
    )

    alone = mean_response_activations(bundle, [short], batch_size=1)
    together = mean_response_activations(bundle, [short, long], batch_size=2)

    assert np.allclose(alone[0], together[0], atol=1e-2), (
        "padding ortalamaya sızıyor — batch boyutu sonucu değiştiriyor"
    )
```

- [ ] **Step 2: Test'lerin başarısız olduğunu doğrula**

Run: `cd ~/assistant-axis && uv run --extra dev pytest tests/test_activations.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aax.activations'`

- [ ] **Step 3: `src/aax/activations.py` yaz**

```python
"""Hook tabanlı residual stream yakalama.

Rol vektörü tanımı (spec Bölüm 2): rolü ifade eden yanıtların RESPONSE
token'ları üzerinden alınan POST-MLP residual stream ortalaması, her katman
için ayrı. HF transformers'ta post-MLP residual = decoder katmanının forward
çıktısının ilk elemanı.

Neden teacher-forced tek prefill: metin zaten üretilmiş durumda, tekrar
decode etmeye gerek yok. Prompt+yanıtı birlikte tek forward'dan geçirip
yanıt pozisyonlarını maskeliyoruz — decode'suz olduğu için çok hızlı.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Sequence

import numpy as np


def response_token_mask(prompt_len: int, total_len: int, pad_len: int) -> list[bool]:
    """Hangi pozisyonlar ortalamaya girer.

    total_len = prompt + yanıt (padding hariç). pad_len sağdan eklenen
    padding sayısı; bu pozisyonlar hiçbir zaman ortalamaya girmez.
    """
    if prompt_len > total_len:
        raise ValueError(f"prompt uzunluğu toplamı aşıyor: {prompt_len} > {total_len}")
    if prompt_len == total_len:
        raise ValueError("response boş — ortalanacak token yok")
    return [False] * prompt_len + [True] * (total_len - prompt_len) + [False] * pad_len


@contextmanager
def _layer_output_hooks(model, store: dict):
    handles = []
    for index, layer in enumerate(model.model.layers):
        def make_hook(i):
            def hook(_module, _inputs, output):
                store[i] = output[0] if isinstance(output, tuple) else output
            return hook

        handles.append(layer.register_forward_hook(make_hook(index)))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def capture_layer_outputs(bundle, input_ids, attention_mask):
    """Tüm decoder katmanlarının çıktısını yakala.

    Dönüş: [n_layers, batch, seq, d_model] tensörü.
    """
    import torch

    store: dict[int, "torch.Tensor"] = {}
    with _layer_output_hooks(bundle.model, store), torch.no_grad():
        bundle.model(input_ids=input_ids, attention_mask=attention_mask)
    return torch.stack([store[i] for i in range(bundle.n_layers)], dim=0)


def mean_response_activations(
    bundle,
    items: Sequence[tuple[list[int], list[int]]],
    *,
    batch_size: int = 8,
) -> np.ndarray:
    """Her (prompt_ids, response_ids) çifti için katman başına ortalama residual.

    Dönüş: [n_texts, n_layers, d_model] float32.

    Ortalama float32'de birikir: bf16'da 160 token üzerinde toplama
    anlamlı hassasiyet kaybeder.
    """
    import torch

    pad_id = bundle.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = bundle.tokenizer.eos_token_id

    out = np.empty((len(items), bundle.n_layers, bundle.d_model), dtype=np.float32)

    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        seqs = [p + r for p, r in batch]
        max_len = max(len(s) for s in seqs)

        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        masks = []
        for row, (seq, (prompt_ids, _)) in enumerate(zip(seqs, batch)):
            input_ids[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            attention_mask[row, : len(seq)] = 1
            masks.append(
                response_token_mask(
                    prompt_len=len(prompt_ids),
                    total_len=len(seq),
                    pad_len=max_len - len(seq),
                )
            )

        device = bundle.model.device
        captured = capture_layer_outputs(
            bundle, input_ids.to(device), attention_mask.to(device)
        )  # [L, B, S, D]

        selector = torch.tensor(masks, dtype=torch.bool, device=device)  # [B, S]
        counts = selector.sum(dim=1).clamp(min=1).to(torch.float32)  # [B]
        weighted = captured.to(torch.float32) * selector.unsqueeze(0).unsqueeze(-1)
        summed = weighted.sum(dim=2)  # [L, B, D]
        means = summed / counts.view(1, -1, 1)  # [L, B, D]

        out[start : start + len(batch)] = means.permute(1, 0, 2).cpu().numpy()

    return out
```

**VRAM notu — batch boyutunu seçerken bilmen gereken hesap.** `capture_layer_outputs` tüm katmanları aynı anda tutar: `[L, B, S, D]`. Qwen3-1.7B'de L=28, D=2048; B=8 ve S≈400 için bf16 tensör ~366 MB, float32'ye çevrilince +733 MB. Model 3.4 GB, kullanılabilir VRAM ~7 GB — batch 8 sığar ama boşluk azdır. OOM görürsen `--batch-size 4` ilk çare; kalıcı çözüm gerekirse katman katman biriktirmek gerekir, ama bu planın kapsamı dışında.

- [ ] **Step 4: Saf testlerin geçtiğini doğrula**

Run: `cd ~/assistant-axis && uv run --extra dev pytest tests/test_activations.py -v`
Expected: PASS, 4 passed, 3 deselected

- [ ] **Step 5: GPU testlerini koş — bu planın en önemli doğrulaması**

Run: `cd ~/assistant-axis && uv run --extra dev --extra ml pytest tests/test_activations.py -v -m gpu`
Expected: PASS, 3 passed.

`test_hook_output_equals_hidden_states_from_forward` başarısız olursa **devam etme**: yanlış tensör yakalanıyor demektir ve bundan sonraki her sayı anlamsız olur.

- [ ] **Step 6: Commit**

```bash
git add src/aax/activations.py tests/test_activations.py
git commit -m "feat: hook tabanlı aktivasyon yakalama"
```

---

### Task 5: Aşama 1 — tam üretim ve aktivasyon

**Files:**
- Create: `src/aax/rollouts.py`
- Create: `scripts/04_generate_rollouts.py`
- Create: `scripts/05_capture_activations.py`
- Test: `tests/test_rollouts.py`

**Interfaces:**
- Consumes: `aax.prompts` (Task 2), `aax.model` (Task 1), `aax.activations` (Task 4)
- Produces:
  - `aax.rollouts.rollout_record(spec, answer) -> dict`
  - `aax.rollouts.write_rollouts(path, records) -> None` — atomik JSONL yazımı
  - `aax.rollouts.read_rollouts(path) -> list[dict]`
  - Artifact: `data/rollouts.jsonl` — 16.000 kayıt
  - Artifact: `data/activations.npy` — `[16000, L, d_model]` float32
  - Artifact: `data/activations_index.json` — satır sırası ve meta

- [ ] **Step 1: Failing test'leri yaz**

`tests/test_rollouts.py`:

```python
import json

import pytest

from aax.prompts import RolloutSpec
from aax.rollouts import read_rollouts, rollout_record, write_rollouts


def make_spec(role="pirate", question="soru?"):
    return RolloutSpec(
        kind="role",
        role=role,
        system_prompt="You are a pirate.",
        question=question,
        sample_index=0,
    )


def test_record_carries_every_field_needed_downstream():
    record = rollout_record(make_spec(), "Arrr!")
    assert record["kind"] == "role"
    assert record["role"] == "pirate"
    assert record["system_prompt"] == "You are a pirate."
    assert record["question"] == "soru?"
    assert record["answer"] == "Arrr!"
    assert record["sample_index"] == 0


def test_write_then_read_roundtrips(tmp_path):
    records = [rollout_record(make_spec(question=f"s{i}"), f"a{i}") for i in range(5)]
    path = tmp_path / "rollouts.jsonl"
    write_rollouts(path, records)
    assert read_rollouts(path) == records


def test_write_is_atomic_no_temp_left_behind(tmp_path):
    path = tmp_path / "rollouts.jsonl"
    write_rollouts(path, [rollout_record(make_spec(), "x")])
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != "rollouts.jsonl"]
    assert leftovers == []


def test_read_rejects_truncated_file(tmp_path):
    path = tmp_path / "rollouts.jsonl"
    path.write_text('{"kind": "role"}\n{"kind": "ro', encoding="utf-8")
    with pytest.raises(ValueError, match="satır"):
        read_rollouts(path)


def test_empty_answer_is_rejected():
    with pytest.raises(ValueError, match="boş"):
        rollout_record(make_spec(), "   ")
```

- [ ] **Step 2: Test'lerin başarısız olduğunu doğrula**

Run: `cd ~/assistant-axis && uv run --extra dev pytest tests/test_rollouts.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aax.rollouts'`

- [ ] **Step 3: `src/aax/rollouts.py` yaz**

```python
"""Rollout kayıtlarının şeması ve diske yazımı.

Yazım atomik: 16.000 kayıtlık bir dosyanın yarısı diskte kalırsa aşağı akış
sessizce eksik veriyle çalışır.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from aax.prompts import RolloutSpec


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
```

- [ ] **Step 4: Testlerin geçtiğini doğrula**

Run: `cd ~/assistant-axis && uv run --extra dev pytest tests/test_rollouts.py -v`
Expected: PASS, 5 passed

- [ ] **Step 5: `scripts/04_generate_rollouts.py` yaz**

```python
#!/usr/bin/env python3
"""Aşama 1 — 16.000 rollout'un vLLM ile üretimi.

vLLM burada kullanılıyor çünkü steering yok, sadece metin lazım. Steering'li
ve capping'li her koşu HF transformers kullanır (spec Bölüm 4.1) — makale
vLLM steering'inin tutarlı %2-3 daha kötü ölçtüğünü raporluyor.

VRAM: RTX 4060'ın 8188 MiB'ının ~1215'i masaüstünde. vLLM
gpu_memory_utilization'ı TOPLAM belleğe göre hesaplar, bu yüzden 0.9 gibi bir
değer OOM verir. Varsayılanı 0.70 tuttuk.

Kullanım:
    uv run --extra gen python scripts/04_generate_rollouts.py
    uv run --extra gen python scripts/04_generate_rollouts.py --limit 100  # duman testi
"""
from __future__ import annotations

import argparse
import json

from aax import config
from aax.prompts import build_default_specs, build_role_specs, load_role_catalog, to_chat_messages
from aax.rollouts import rollout_record, write_rollouts

OUT_PATH = config.DATA_DIR / "rollouts.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="ilk N spec (duman testi)")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--samples-per-default-prompt", type=int, default=10)
    args = parser.parse_args()

    catalog = load_role_catalog(config.DATA_DIR / "roles.json")
    questions = json.loads(
        (config.DATA_DIR / "questions.json").read_text(encoding="utf-8")
    )["shared_questions"]

    specs = build_role_specs(catalog, questions) + build_default_specs(
        questions, samples_per_prompt=args.samples_per_default_prompt
    )
    if args.limit is not None:
        specs = specs[: args.limit]
    print(f"{len(specs)} rollout üretilecek")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(config.TARGET_MODEL)
    prompts = [
        tokenizer.apply_chat_template(
            to_chat_messages(spec),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for spec in specs
    ]

    llm = LLM(
        model=config.TARGET_MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=2048,
    )
    sampling = SamplingParams(
        max_tokens=args.max_new_tokens, temperature=1.0, top_p=0.95
    )
    outputs = llm.generate(prompts, sampling)

    records = []
    empty = 0
    for spec, output in zip(specs, outputs):
        answer = output.outputs[0].text.strip()
        if not answer:
            empty += 1
            continue
        records.append(rollout_record(spec, answer))

    write_rollouts(OUT_PATH, records)
    print(f"Yazıldı: {OUT_PATH} ({len(records)} kayıt, {empty} boş yanıt atlandı)")
    if empty > len(specs) * 0.02:
        print(f"UYARI: boş yanıt oranı %{100*empty/len(specs):.1f} — max_model_len'i kontrol et")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: `scripts/05_capture_activations.py` yaz**

```python
#!/usr/bin/env python3
"""Aşama 1 — üretilmiş rollout'ların aktivasyonlarını yakala.

vLLM süreçten çıkmış olmalı: iki motor aynı anda VRAM'e sığmaz.

Kullanım:
    uv run --extra ml python scripts/05_capture_activations.py
    uv run --extra ml python scripts/05_capture_activations.py --batch-size 4  # OOM olursa
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from aax import config
from aax.activations import mean_response_activations
from aax.model import free_vram_mib, load_hf_model
from aax.prompts import RolloutSpec, to_chat_messages
from aax.rollouts import read_rollouts

ACTS_PATH = config.DATA_DIR / "activations.npy"
INDEX_PATH = config.DATA_DIR / "activations_index.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    records = read_rollouts(config.DATA_DIR / "rollouts.jsonl")
    print(f"{len(records)} rollout okundu")

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

    acts = mean_response_activations(bundle, items, batch_size=args.batch_size)

    np.save(ACTS_PATH, acts)
    INDEX_PATH.write_text(
        json.dumps(
            {
                "n_rows": int(acts.shape[0]),
                "n_layers": int(acts.shape[1]),
                "d_model": int(acts.shape[2]),
                "model": config.TARGET_MODEL,
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
    print(f"Yazıldı: {ACTS_PATH} {acts.shape} float32 (~{acts.nbytes/1e9:.2f} GB)")
    print(f"Yazıldı: {INDEX_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 7: Duman testi — 100 rollout**

Run: `cd ~/assistant-axis && uv run --extra gen python scripts/04_generate_rollouts.py --limit 100`
Expected: `Yazıldı: .../rollouts.jsonl (100 kayıt, 0 boş yanıt atlandı)` civarı. OOM alırsan `--gpu-memory-utilization 0.60` dene ve raporla.

Run: `cd ~/assistant-axis && uv run --extra ml python scripts/05_capture_activations.py`
Expected: `(100, L, D) float32`. OOM alırsan `--batch-size 4`.

- [ ] **Step 8: Commit**

```bash
git add src/aax/rollouts.py scripts/04_generate_rollouts.py scripts/05_capture_activations.py tests/test_rollouts.py
git commit -m "feat: Aşama 1 üretim ve aktivasyon yakalama"
```

- [ ] **Step 9: OPERATÖR ADIMI — tam koşu**

Duman testi geçtikten sonra, GPU'yu uzun süre meşgul edecek koşu:

```bash
cd ~/assistant-axis && uv run --extra gen python scripts/04_generate_rollouts.py && uv run --extra ml python scripts/05_capture_activations.py
```

Beklenen: ~16.000 rollout, `activations.npy` yaklaşık `16000 × L × d_model × 4` bayt. Qwen3-1.7B için bu ~3-4 GB — disk yeterli (170 GB boş).

---

### Task 6: Aşama 2 — rol ifadesi probe'u

Hakeme 16.000 rollout sormak batch'li bile olsa ~1600 çağrı eder ve aşama bütçesi 300. Çözüm: 2.000 rollout'u hakeme sor, o etiketlerle yerel bir sınıflandırıcı eğit, kalanı bedava etiketle.

**Files:**
- Create: `src/aax/probe.py`
- Create: `scripts/06_label_and_train_probe.py`
- Test: `tests/test_probe.py`

**Interfaces:**
- Consumes: `aax.judge.score_role_expression` (Plan 1), `aax.gateway` (Plan 1), `aax.rollouts.read_rollouts` (Task 5)
- Produces:
  - `aax.probe.stratified_sample(records, n, *, seed) -> list[int]` — rol başına dengeli indeks örneklemesi
  - `aax.probe.RoleExpressionProbe` — `fit(embeddings, labels)`, `predict(embeddings) -> list[str]`, `holdout_agreement: float`
  - `aax.probe.embed_answers(answers: list[str]) -> np.ndarray` — bge-m3
  - Artifact: `data/probe_labels.json` — hakem etiketleri
  - Artifact: `data/role_expression.json` — 16.000 rollout için kategori

- [ ] **Step 1: Failing test'leri yaz**

`tests/test_probe.py`:

```python
import numpy as np
import pytest

from aax.probe import RoleExpressionProbe, stratified_sample


def make_records(n_roles=4, per_role=50):
    return [
        {"kind": "role", "role": f"rol{r}", "answer": f"yanit {r}-{i}"}
        for r in range(n_roles)
        for i in range(per_role)
    ]


def test_stratified_sample_is_balanced_across_roles():
    records = make_records(n_roles=4, per_role=50)
    idx = stratified_sample(records, n=40, seed=1)
    roles = [records[i]["role"] for i in idx]
    counts = {r: roles.count(r) for r in set(roles)}
    assert len(idx) == 40
    assert max(counts.values()) - min(counts.values()) <= 1


def test_stratified_sample_is_deterministic():
    records = make_records()
    assert stratified_sample(records, n=20, seed=7) == stratified_sample(records, n=20, seed=7)


def test_stratified_sample_differs_with_seed():
    records = make_records()
    assert stratified_sample(records, n=20, seed=7) != stratified_sample(records, n=20, seed=8)


def test_stratified_sample_rejects_n_larger_than_population():
    records = make_records(n_roles=2, per_role=3)
    with pytest.raises(ValueError, match="örnek"):
        stratified_sample(records, n=100, seed=1)


def test_probe_learns_a_separable_signal():
    """Doğrusal ayrılabilir sentetik veride probe neredeyse mükemmel olmalı.

    Bu testin amacı probe'un ÇALIŞTIĞINI göstermek, gerçek veride
    başarılı olacağını değil."""
    rng = np.random.default_rng(0)
    n = 200
    fully = rng.normal(loc=+3.0, scale=1.0, size=(n, 8))
    no = rng.normal(loc=-3.0, scale=1.0, size=(n, 8))
    embeddings = np.vstack([fully, no])
    labels = ["fully"] * n + ["no"] * n

    probe = RoleExpressionProbe(seed=0)
    probe.fit(embeddings, labels)
    assert probe.holdout_agreement > 0.95


def test_probe_reports_low_agreement_on_pure_noise():
    """Sinyal yoksa probe bunu saklamamalı — geri çekilme kuralı buna bakar."""
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(400, 8))
    labels = ["fully" if i % 2 == 0 else "no" for i in range(400)]

    probe = RoleExpressionProbe(seed=0)
    probe.fit(embeddings, labels)
    assert probe.holdout_agreement < 0.75


def test_probe_predict_returns_one_label_per_row():
    rng = np.random.default_rng(0)
    embeddings = np.vstack([
        rng.normal(loc=+3.0, size=(50, 8)),
        rng.normal(loc=-3.0, size=(50, 8)),
    ])
    labels = ["fully"] * 50 + ["no"] * 50
    probe = RoleExpressionProbe(seed=0)
    probe.fit(embeddings, labels)
    out = probe.predict(rng.normal(size=(7, 8)))
    assert len(out) == 7
    assert set(out) <= {"fully", "somewhat", "no"}


def test_probe_refuses_to_predict_before_fit():
    probe = RoleExpressionProbe(seed=0)
    with pytest.raises(RuntimeError, match="eğitilmedi"):
        probe.predict(np.zeros((2, 8)))
```

- [ ] **Step 2: Test'lerin başarısız olduğunu doğrula**

Run: `cd ~/assistant-axis && uv run --extra dev pytest tests/test_probe.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aax.probe'`

- [ ] **Step 3: `src/aax/probe.py` yaz**

```python
"""Rol ifadesi probe'u — spec Bölüm 8, Sapma 2.

Makale her yanıtı LLM hakeme sorar. Bizim gateway bütçemiz buna yetmiyor
(16.000 rollout ≈ 1600 çağrı, aşama bütçesi 300). Bunun yerine 2.000 yanıtı
hakeme sorup etiketleriyle bge-m3 embedding'leri üzerine lojistik regresyon
oturtuyoruz.

Geri çekilme kuralı: held-out uyum %85'in altındaysa probe atılır ve rol
düzeyinde kaba bir tut/at filtresine dönülür. Bu karar raporlanır.
"""
from __future__ import annotations

import random
from collections import defaultdict

import numpy as np

HOLDOUT_FRACTION = 0.2
FALLBACK_THRESHOLD = 0.85


def stratified_sample(records: list[dict], n: int, *, seed: int) -> list[int]:
    """Rol başına mümkün olduğunca dengeli n indeks seç."""
    if n > len(records):
        raise ValueError(f"örnek sayısı popülasyondan büyük: {n} > {len(records)}")

    by_role: dict[str | None, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_role[record.get("role")].append(index)

    rng = random.Random(seed)
    for indices in by_role.values():
        rng.shuffle(indices)

    roles = sorted(by_role, key=lambda r: (r is None, r))
    chosen: list[int] = []
    cursor = 0
    while len(chosen) < n:
        progressed = False
        for role in roles:
            bucket = by_role[role]
            if cursor < len(bucket):
                chosen.append(bucket[cursor])
                progressed = True
                if len(chosen) == n:
                    break
        if not progressed:
            break
        cursor += 1
    return sorted(chosen)


def embed_answers(answers: list[str], *, model_id: str = "BAAI/bge-m3") -> np.ndarray:
    """bge-m3 ile yanıt embedding'leri. L2 normalize."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_id)
    return model.encode(
        answers, batch_size=16, normalize_embeddings=True, show_progress_bar=True
    )


class RoleExpressionProbe:
    """bge-m3 embedding'leri üzerine lojistik regresyon."""

    def __init__(self, *, seed: int = 0) -> None:
        self.seed = seed
        self._model = None
        self.holdout_agreement: float | None = None

    def fit(self, embeddings: np.ndarray, labels: list[str]) -> None:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split

        x_train, x_test, y_train, y_test = train_test_split(
            embeddings,
            labels,
            test_size=HOLDOUT_FRACTION,
            random_state=self.seed,
            stratify=labels if len(set(labels)) > 1 else None,
        )
        model = LogisticRegression(max_iter=2000, random_state=self.seed)
        model.fit(x_train, y_train)
        self._model = model
        self.holdout_agreement = float(model.score(x_test, y_test))

    def predict(self, embeddings: np.ndarray) -> list[str]:
        if self._model is None:
            raise RuntimeError("probe eğitilmedi — önce fit() çağır")
        return list(self._model.predict(embeddings))

    @property
    def is_trustworthy(self) -> bool:
        return (
            self.holdout_agreement is not None
            and self.holdout_agreement >= FALLBACK_THRESHOLD
        )
```

- [ ] **Step 4: Testlerin geçtiğini doğrula**

Run: `cd ~/assistant-axis && uv run --extra dev --extra ml pytest tests/test_probe.py -v`
Expected: PASS, 9 passed

- [ ] **Step 5: `scripts/06_label_and_train_probe.py` yaz**

```python
#!/usr/bin/env python3
"""Aşama 2 — hakem etiketleri topla, probe eğit, 16.000 rollout'u etiketle.

Kullanım:
    uv run python scripts/06_label_and_train_probe.py --dry-run
    uv run --extra ml python scripts/06_label_and_train_probe.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

import numpy as np

from aax import config
from aax.gateway import BudgetExceeded, CircuitOpen, GatewayError, build_default_client
from aax.judge import JudgeParseError, score_role_expression
from aax.probe import RoleExpressionProbe, embed_answers, stratified_sample
from aax.prompts import load_role_catalog
from aax.rollouts import read_rollouts

STAGE = "stage2_probe_labels"
SEED = 20260806
LABEL_SAMPLE_SIZE = 2000
LABELS_PATH = config.DATA_DIR / "probe_labels.json"
OUT_PATH = config.DATA_DIR / "role_expression.json"


def collapse(score: int) -> str:
    if score == 3:
        return "fully"
    if score == 2:
        return "somewhat"
    if score in (0, 1):
        return "no"
    raise ValueError(f"Puan 0-3 aralığı dışında: {score!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sample-size", type=int, default=LABEL_SAMPLE_SIZE)
    args = parser.parse_args()

    records = read_rollouts(config.DATA_DIR / "rollouts.jsonl")
    role_rows = [i for i, r in enumerate(records) if r["kind"] == "role"]
    role_records = [records[i] for i in role_rows]

    chosen_local = stratified_sample(role_records, n=args.sample_size, seed=SEED)
    chosen = [role_rows[i] for i in chosen_local]

    # load_role_catalog üzerinden: kısmi/pilot bir katalogla etiketleme yapmak,
    # yanlış rol kümesi üzerinde probe eğitmek demek olurdu.
    catalog = {r["role"]: r["description"] for r in load_role_catalog(config.DATA_DIR / "roles.json")}

    by_role: dict[str, list[int]] = defaultdict(list)
    for row in chosen:
        by_role[records[row]["role"]].append(row)

    client = build_default_client()

    if args.dry_run:
        planned = sum(
            (len(rows) + 9) // 10 for rows in by_role.values()
        )
        cap = config.STAGE_BUDGETS[STAGE]
        print(f"Planlanan çağrı (üst sınır, cache yok sayılarak): {planned}")
        print(f"Aşama bütçesi: {cap}")
        if planned > cap:
            print("HATA: plan aşama bütçesini aşıyor — --sample-size küçült.", file=sys.stderr)
            return 1
        return 0

    labels: dict[int, str] = {}
    try:
        for role, rows in by_role.items():
            items = [(records[i]["question"], records[i]["answer"]) for i in rows]
            scores = score_role_expression(
                client,
                role=role,
                description=catalog.get(role, f"the role of a {role}"),
                items=items,
                stage=STAGE,
            )
            for row, score in zip(rows, scores):
                labels[row] = collapse(score)
            print(f"\r{len(labels)}/{len(chosen)} etiketlendi", end="")
    except (BudgetExceeded, CircuitOpen) as exc:
        print(f"\nDURDURULDU: {exc}", file=sys.stderr)
        return 2
    except (GatewayError, JudgeParseError) as exc:
        print(f"\nBAŞARISIZ: {exc}", file=sys.stderr)
        return 2

    print()
    LABELS_PATH.write_text(
        json.dumps({"seed": SEED, "labels": {str(k): v for k, v in labels.items()}}, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Yazıldı: {LABELS_PATH} ({len(labels)} etiket), gönderilen istek: {client.sends_made}")

    rows = sorted(labels)
    embeddings = embed_answers([records[i]["answer"] for i in rows])
    probe = RoleExpressionProbe(seed=SEED)
    probe.fit(embeddings, [labels[i] for i in rows])
    print(f"Probe held-out uyumu: {probe.holdout_agreement:.1%} (eşik %85)")

    if not probe.is_trustworthy:
        print(
            "PROBE GÜVENİLİR DEĞİL — spec'in geri çekilme kuralı devreye giriyor.\n"
            "  Rol düzeyinde tut/at filtresine dön ve bunu sonuçlarda raporla.",
            file=sys.stderr,
        )
        return 1

    all_role_answers = [records[i]["answer"] for i in role_rows]
    all_embeddings = embed_answers(all_role_answers)
    predicted = probe.predict(all_embeddings)

    expression = {str(row): labels.get(row, pred) for row, pred in zip(role_rows, predicted)}
    OUT_PATH.write_text(
        json.dumps(
            {
                "holdout_agreement": probe.holdout_agreement,
                "n_judge_labels": len(labels),
                "n_probe_labels": len(role_rows) - len(labels),
                "expression": expression,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    counts = {c: list(expression.values()).count(c) for c in ("fully", "somewhat", "no")}
    print(f"Yazıldı: {OUT_PATH}")
    print(f"Dağılım: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Dry-run'ı doğrula**

Run: `cd ~/assistant-axis && uv run python scripts/06_label_and_train_probe.py --dry-run`
Expected: planlanan çağrı sayısı ve aşama bütçesi (300) basılır, çıkış kodu 0. Anahtar yoksa temiz tanı ve çıkış kodu 2 — bu da beklenen.

- [ ] **Step 7: Commit**

```bash
git add src/aax/probe.py scripts/06_label_and_train_probe.py tests/test_probe.py
git commit -m "feat: Aşama 2 rol ifadesi probe'u"
```

---

### Task 7: Aşama 3 — eksen çıkarımı ve A kriteri

Bu planın nihai çıktısı. `axis.py` tamamen saf numpy: model, GPU, ağ yok. Bu sayede ekilmiş bir yönle sentetik veride tam test edilebilir.

**Files:**
- Create: `src/aax/axis.py`
- Create: `scripts/07_extract_axis.py`
- Test: `tests/test_axis.py`

**Interfaces:**
- Consumes: `data/activations.npy`, `data/role_expression.json` (Task 6), `data/activations_index.json` (Task 5)
- Produces:
  - `aax.axis.role_vectors(activations, row_roles, row_categories, *, min_responses=10) -> tuple[np.ndarray, list[str]]`
  - `aax.axis.contrast_axis(default_mean, role_mean) -> np.ndarray` — L2 normalize
  - `aax.axis.pca_components(vectors, n_components) -> tuple[np.ndarray, np.ndarray]` — `(components, explained_variance_ratio)`
  - `aax.axis.cosine(a, b) -> float`
  - `aax.axis.projection_percentile(value, distribution) -> float`
  - `aax.axis.evaluate_criterion_a(cos_pc1_axis, default_percentile) -> dict`
  - Artifact: `results/axis/` — vektörler, PCA, figür, `criterion_a.json`

- [ ] **Step 1: Failing test'leri yaz**

`tests/test_axis.py`:

```python
import numpy as np
import pytest

from aax.axis import (
    contrast_axis,
    cosine,
    evaluate_criterion_a,
    pca_components,
    projection_percentile,
    role_vectors,
)


def test_cosine_of_identical_vectors_is_one():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine(v, v) == pytest.approx(1.0)


def test_cosine_of_opposite_vectors_is_minus_one():
    v = np.array([1.0, 0.0])
    assert cosine(v, -v) == pytest.approx(-1.0)


def test_cosine_rejects_zero_vector():
    with pytest.raises(ValueError, match="sıfır"):
        cosine(np.zeros(3), np.ones(3))


def test_contrast_axis_is_unit_norm():
    axis = contrast_axis(np.array([5.0, 0.0]), np.array([1.0, 0.0]))
    assert np.linalg.norm(axis) == pytest.approx(1.0)


def test_contrast_axis_points_from_roles_toward_default():
    axis = contrast_axis(np.array([10.0, 0.0]), np.array([2.0, 0.0]))
    assert axis[0] > 0


def test_pca_recovers_a_planted_direction():
    """Ekilmiş bir yön varsa PC1 onu bulmalı."""
    rng = np.random.default_rng(0)
    planted = np.array([1.0, 0.0, 0.0, 0.0])
    scores = rng.normal(scale=5.0, size=200)
    noise = rng.normal(scale=0.1, size=(200, 4))
    vectors = scores[:, None] * planted[None, :] + noise

    components, ratios = pca_components(vectors, n_components=2)
    assert abs(cosine(components[0], planted)) > 0.99
    assert ratios[0] > 0.9


def test_role_vectors_skip_categories_below_minimum():
    acts = np.ones((12, 2, 3), dtype=np.float32)
    roles = ["a"] * 12
    cats = ["fully"] * 9 + ["no"] * 3
    vectors, names = role_vectors(acts, roles, cats, min_responses=10)
    assert names == []


def test_role_vectors_averages_qualifying_rows():
    acts = np.zeros((10, 1, 2), dtype=np.float32)
    acts[:, 0, 0] = 4.0
    roles = ["pirate"] * 10
    cats = ["fully"] * 10
    vectors, names = role_vectors(acts, roles, cats, min_responses=10)
    assert names == ["pirate"]
    assert vectors.shape == (1, 1, 2)
    assert vectors[0, 0, 0] == pytest.approx(4.0)


def test_role_vectors_keeps_fully_and_somewhat_separate():
    acts = np.zeros((20, 1, 2), dtype=np.float32)
    acts[:10, 0, 0] = 1.0
    acts[10:, 0, 0] = 9.0
    roles = ["bard"] * 20
    cats = ["fully"] * 10 + ["somewhat"] * 10
    vectors, names = role_vectors(acts, roles, cats, min_responses=10)
    assert sorted(names) == ["bard::fully", "bard::somewhat"]


def test_projection_percentile_at_extremes():
    dist = np.arange(100.0)
    assert projection_percentile(-5.0, dist) == pytest.approx(0.0)
    assert projection_percentile(200.0, dist) == pytest.approx(1.0)


def test_criterion_a_passes_when_both_conditions_hold():
    result = evaluate_criterion_a(cos_pc1_axis=0.72, default_percentile=0.95)
    assert result["passed"] is True


def test_criterion_a_fails_on_low_cosine():
    result = evaluate_criterion_a(cos_pc1_axis=0.41, default_percentile=0.98)
    assert result["passed"] is False
    assert "cos" in result["reason"]


def test_criterion_a_fails_when_default_not_in_top_decile():
    result = evaluate_criterion_a(cos_pc1_axis=0.80, default_percentile=0.55)
    assert result["passed"] is False
    assert "desil" in result["reason"]


def test_criterion_a_accepts_negative_cosine_by_magnitude():
    """PC1'in işareti keyfîdir; önemli olan büyüklük."""
    result = evaluate_criterion_a(cos_pc1_axis=-0.75, default_percentile=0.02)
    assert result["passed"] is True
```

- [ ] **Step 2: Test'lerin başarısız olduğunu doğrula**

Run: `cd ~/assistant-axis && uv run --extra dev pytest tests/test_axis.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aax.axis'`

- [ ] **Step 3: `src/aax/axis.py` yaz**

```python
"""Persona uzayı analizi — saf numpy.

Bu modül model, GPU veya ağ bilmez. Girdi vektör matrisleri, çıktı eksen ve
PCA sonuçları. Bu sayede ekilmiş bir yönle sentetik veride tam test edilebilir
(spec Bölüm 4.3, Bölüm 10).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

COS_THRESHOLD = 0.6
TOP_DECILE = 0.9


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        raise ValueError("sıfır vektörün kosinüsü tanımsız")
    return float(np.dot(a, b) / (na * nb))


def contrast_axis(default_mean: np.ndarray, role_mean: np.ndarray) -> np.ndarray:
    """Assistant Axis = mean(default) − mean(rol vektörleri), L2 normalize.

    Makale bunu PC1'e tercih ediyor: PC1'in her modelde aynı anlamı taşıyacağı
    garanti değil (Ek G.5).
    """
    axis = np.asarray(default_mean, dtype=np.float64) - np.asarray(role_mean, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm == 0:
        raise ValueError("kontrast vektörü sıfır — default ve rol ortalamaları aynı")
    return axis / norm


def role_vectors(
    activations: np.ndarray,
    row_roles: list[str],
    row_categories: list[str],
    *,
    min_responses: int = 10,
) -> tuple[np.ndarray, list[str]]:
    """(rol, kategori) başına ortalama aktivasyon.

    Makalenin kuralı: bir kategori en az `min_responses` yanıt içermiyorsa
    o vektör hesaplanmaz. fully ve somewhat ayrı vektörler üretir.

    Dönüş: ([n_vectors, n_layers, d_model], isimler) — isim "rol::kategori",
    ya da kategori tekse sadece "rol".
    """
    buckets: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, (role, category) in enumerate(zip(row_roles, row_categories)):
        if category in ("fully", "somewhat"):
            buckets[(role, category)].append(index)

    kept = {k: rows for k, rows in buckets.items() if len(rows) >= min_responses}
    if not kept:
        return np.empty((0,) + activations.shape[1:], dtype=np.float32), []

    roles_with_both = defaultdict(set)
    for role, category in kept:
        roles_with_both[role].add(category)

    names: list[str] = []
    vectors: list[np.ndarray] = []
    for (role, category), rows in sorted(kept.items()):
        name = f"{role}::{category}" if len(roles_with_both[role]) > 1 else role
        names.append(name)
        vectors.append(activations[rows].astype(np.float64).mean(axis=0))

    return np.stack(vectors).astype(np.float32), names


def pca_components(
    vectors: np.ndarray, n_components: int
) -> tuple[np.ndarray, np.ndarray]:
    """Roller arası ortalamayı çıkarıp PCA koş.

    Dönüş: (bileşenler [k, d], açıklanan varyans oranı [k]).
    """
    centered = np.asarray(vectors, dtype=np.float64)
    centered = centered - centered.mean(axis=0, keepdims=True)
    _u, s, vt = np.linalg.svd(centered, full_matrices=False)
    variance = s**2
    ratios = variance / variance.sum()
    k = min(n_components, vt.shape[0])
    return vt[:k], ratios[:k]


def projection_percentile(value: float, distribution: np.ndarray) -> float:
    """`value`'nun dağılım içindeki konumu, 0-1 arası."""
    dist = np.asarray(distribution, dtype=np.float64)
    return float((dist <= value).sum() / len(dist))


def evaluate_criterion_a(cos_pc1_axis: float, default_percentile: float) -> dict:
    """Spec Bölüm 7, A kriteri.

    Geçer: orta katmanda |cos(PC1, kontrast vektörü)| > 0.6 VE default
    Assistant projeksiyonu PC1'in en üst (veya en alt) desilinde.

    PC1'in işareti SVD'nin keyfî bir seçimidir; hem kosinüs hem desil
    büyüklük üzerinden değerlendirilir.
    """
    magnitude = abs(cos_pc1_axis)
    in_extreme_decile = (
        default_percentile >= TOP_DECILE or default_percentile <= 1 - TOP_DECILE
    )

    reasons = []
    if magnitude <= COS_THRESHOLD:
        reasons.append(f"|cos| {magnitude:.3f} <= {COS_THRESHOLD}")
    if not in_extreme_decile:
        reasons.append(
            f"default projeksiyonu uç desilde değil (persentil {default_percentile:.3f})"
        )

    return {
        "cos_pc1_axis": cos_pc1_axis,
        "cos_magnitude": magnitude,
        "default_percentile": default_percentile,
        "passed": not reasons,
        "reason": "; ".join(reasons) if reasons else "her iki koşul da sağlandı",
    }
```

- [ ] **Step 4: Testlerin geçtiğini doğrula**

Run: `cd ~/assistant-axis && uv run --extra dev --extra ml pytest tests/test_axis.py -v`
Expected: PASS, 15 passed

- [ ] **Step 5: `scripts/07_extract_axis.py` yaz**

```python
#!/usr/bin/env python3
"""Aşama 3 — rol vektörleri, PCA, Assistant Axis, A kriteri.

Kullanım:
    uv run --extra ml python scripts/07_extract_axis.py
"""
from __future__ import annotations

import json

import numpy as np

from aax import config
from aax.axis import (
    contrast_axis,
    cosine,
    evaluate_criterion_a,
    pca_components,
    projection_percentile,
    role_vectors,
)

OUT_DIR = config.RESULTS_DIR / "axis"


def main() -> int:
    acts = np.load(config.DATA_DIR / "activations.npy")
    index = json.loads((config.DATA_DIR / "activations_index.json").read_text(encoding="utf-8"))
    expression = json.loads(
        (config.DATA_DIR / "role_expression.json").read_text(encoding="utf-8")
    )["expression"]

    rows = index["rows"]
    middle = index["middle_layer"]
    print(f"{acts.shape[0]} satır, {acts.shape[1]} katman, orta katman {middle}")

    role_idx = [i for i, r in enumerate(rows) if r["kind"] == "role"]
    default_idx = [i for i, r in enumerate(rows) if r["kind"] == "default"]

    categories = [expression.get(str(i), "no") for i in role_idx]
    vectors, names = role_vectors(
        acts[role_idx], [rows[i]["role"] for i in role_idx], categories
    )
    print(f"{len(names)} rol vektörü hesaplandı (>=10 yanıt kuralı sonrası)")
    if len(names) < 40:
        print(f"UYARI: yalnızca {len(names)} vektör — PCA kararsız olabilir.")

    default_mean_all = acts[default_idx].astype(np.float64).mean(axis=0)  # [L, D]

    fully_rows = [i for i, c in zip(role_idx, categories) if c == "fully"]
    role_mean_all = acts[fully_rows].astype(np.float64).mean(axis=0)  # [L, D]

    axis_per_layer = np.stack(
        [contrast_axis(default_mean_all[l], role_mean_all[l]) for l in range(acts.shape[1])]
    )

    cos_by_layer = []
    for layer in range(acts.shape[1]):
        components, _ = pca_components(vectors[:, layer, :], n_components=1)
        cos_by_layer.append(cosine(components[0], axis_per_layer[layer]))

    components_mid, ratios_mid = pca_components(vectors[:, middle, :], n_components=10)
    pc1 = components_mid[0]
    role_projections = vectors[:, middle, :] @ pc1
    default_projection = float(default_mean_all[middle] @ pc1)
    percentile = projection_percentile(default_projection, role_projections)

    verdict = evaluate_criterion_a(cos_by_layer[middle], percentile)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "assistant_axis.npy", axis_per_layer)
    np.save(OUT_DIR / "role_vectors.npy", vectors)
    (OUT_DIR / "role_names.json").write_text(json.dumps(names, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / "criterion_a.json").write_text(
        json.dumps(
            {
                **verdict,
                "middle_layer": middle,
                "n_role_vectors": len(names),
                "cos_by_layer": cos_by_layer,
                "explained_variance_ratio": ratios_mid.tolist(),
                "n_components_for_70pct": int(np.searchsorted(np.cumsum(ratios_mid), 0.70) + 1),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    order = np.argsort(role_projections)
    print()
    print(f"PC1 varyans oranı: {ratios_mid[0]:.1%}")
    print(f"cos(PC1, eksen) orta katmanda: {cos_by_layer[middle]:.3f}")
    print(f"default Assistant persentili: {percentile:.3f}")
    print()
    print("PC1'in bir ucu:", [names[i] for i in order[:6]])
    print("PC1'in diğer ucu:", [names[i] for i in order[-6:]])
    print()
    print("A KRİTERİ:", "GEÇTİ" if verdict["passed"] else "DÜŞTÜ")
    print(" ", verdict["reason"])
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Tam test paketinin yeşil olduğunu doğrula**

Run: `cd ~/assistant-axis && uv run --extra dev --extra ml pytest -q`
Expected: PASS. Plan 1'in 207 testi + bu planın ~47 testi.

- [ ] **Step 7: Commit**

```bash
git add src/aax/axis.py scripts/07_extract_axis.py tests/test_axis.py
git commit -m "feat: Aşama 3 eksen çıkarımı ve A kriteri"
```

- [ ] **Step 8: OPERATÖR ADIMI — A kriteri kararı**

```bash
cd ~/assistant-axis && uv run --extra ml python scripts/07_extract_axis.py
```

Çıktı `results/axis/criterion_a.json`'a yazılır ve commit edilir.

**GEÇTİ** → Plan 3 (Aşama 4-5: steering sweep ve persona drift) yazılabilir.
**DÜŞTÜ** → bu da bir sonuçtur: "Assistant Axis 1.7B ölçeğinde oluşmuyor". Eşik gevşetilmez. Bu durumda Llama-3.2-3B ile tekrarlamak veya çalışmayı burada sonlandırıp negatif bulguyu raporlamak arasında karar verilir.

---

## Plan 2 Tamamlanma Kriterleri

- [ ] `uv run --extra dev --extra ml pytest -q` yeşil; GPU testleri `-m gpu` ile ayrıca geçiyor
- [ ] `data/judge_gate.json` — `passed: true`, uyum ≥ %75
- [ ] `data/rollouts.jsonl` — ~16.000 kayıt
- [ ] `data/activations.npy` — `[~16000, L, d_model]` float32
- [ ] `data/role_expression.json` — held-out uyum raporlanmış
- [ ] `results/axis/criterion_a.json` — A kriteri kararı, commit edilmiş
- [ ] Gateway bütçesi: `stage05_judge_gate` ≤ 15, `stage2_probe_labels` ≤ 300
