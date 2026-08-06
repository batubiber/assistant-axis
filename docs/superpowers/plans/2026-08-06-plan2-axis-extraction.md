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
- **Testler ağa çıkmaz** — `tests/conftest.py` connect/DNS/httpx'i engeller. Marker'lar İKİYE ayrılmıştır (Final Fix Wave, D2): `@pytest.mark.ml` = torch/transformers gerekir ama CUDA gerekmez (varsayılan koşuda **KOŞAR**), `@pytest.mark.gpu` = CUDA ya da gerçek modelin yüklenmesi gerekir (varsayılan koşuda atlanır).
- `data/` gitignore'dadır. `results/` commit edilir — `results/**/*.npy` dahil (`.gitignore`'daki global `*.npy` kuralının negasyonu, Final Fix Wave D4).
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

    bundle = load_hf_model()  # config.TARGET_MODEL
    assert bundle.n_layers == bundle.model.config.num_hidden_layers
    assert bundle.d_model == bundle.model.config.hidden_size
    assert bundle.middle_layer == bundle.n_layers // 2
    assert len(bundle.model.model.layers) == bundle.n_layers
```

GPU testleri `load_hf_model()` ile **hedef modelin kendisini** (Qwen3-1.7B) kullanır. Başlangıçta daha küçük bir model düşünülmüştü ama HF cache'inde ağırlığı olan uygun bir model yok; ayrıca hedef modelde test etmek daha güçlü: küçük modelde geçip hedefte patlayan bir mimari farkı olamaz. Maliyet birkaç forward pass, yani saniyeler.

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
Expected: PASS, 1 passed. Bu test hedef modeli yükler; Step 7'de indirilmişse cache'ten gelir, indirilmemişse bu adımı Step 7'den sonra koş.

**Not:** `tests/conftest.py`'nin autouse fixture'ı `HF_HUB_OFFLINE=1` de ayarlar. Sebebi: `huggingface_hub`'ın yeni sürümü cache'teki bir modeli yüklerken bile etag kontrolü için ağa çıkmaya çalışıyor ve soket kilidimizin fırlattığı özel `RuntimeError`'ı "ağ yok, cache'e düş" sinyali olarak tanımıyor (yalnızca `httpx.ConnectError` ve `TimeoutException`'ı tanıyor), bu yüzden hata `OSError: Can't load the configuration` diye maskelenerek yukarı sızıyor. `HF_HUB_OFFLINE=1` kilidi zayıflatmaz, tamamlar: HF denemeden cache'e düşer, soketler aynen kapalı kalır. Bu yalnızca testleri etkiler — script'ler pytest altında koşmadığı için indirme yapabilir.

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
  - `scripts/03_judge_gate.py:_parse_human_score(raw: str) -> int` — elle girilen insan puanını doğrular (sayısal, 0-3); `ValueError` fırlatır, `run_score` tüm bozuk satırları toplayıp tek seferde raporlar
  - Artifact: `data/judge_gate_labels.csv` — **KÖR** operatör çalışma sayfası: yalnızca `idx, role, question, answer, human_score` sütunları. `machine_score` sütunu YOKTUR — operatör insan puanını doldururken makinenin puanını hiçbir biçimde göremez (bkz. Bulgu 1 / modül docstring'i).
  - Artifact: `data/judge_gate_machine.json` — makine puanları, `idx` (string) → `machine_score` (int) sözlüğü. `judge_gate_labels.csv`'den AYRI tutulur; `--score` ikisini `idx` üzerinden birleştirir ve idx uyuşmazlığında fatal hata verir.
  - Artifact: `data/judge_gate.json` — `{"n": int, "agreement": float, "passed": bool, "threshold": 0.75, "pairs": [...]}`

- [ ] **Step 1: Failing test'leri yaz**

`tests/test_judge_gate.py`:

```python
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
    --machine   pilot yanıtları hakeme puanlatır, KÖR bir elle etiketleme
                şablonu (`data/judge_gate_labels.csv`) yazar — bu dosyada
                makine puanı YOKTUR. Makine puanları ayrı bir dosyaya
                (`data/judge_gate_machine.json`) gider.
    --score     senin doldurduğun kör şablonu ve ayrı makine dosyasını idx
                üzerinden birleştirir, uyumu hesaplar, kapıyı açar/kapar

Neden iki ayrı dosya: makine puanı ve insan puanı aynı satırda yan yana
dursa operatör 40 satırı doldururken makinenin cevabını göz ucuyla görür.
Bu, ölçümü kendi kendini onaylayan bir şeye çevirir — makinenin ne kadar iyi
olduğunu değil, operatörün makineye ne kadar uyduğunu ölçer. Körlük yalnızca
bir talimat cümlesiyle ("bakmadan doldur") sağlanamaz; dosya yapısı bunu
FİZİKSEL olarak imkânsız kılmalı.

Kullanım:
    uv run python scripts/03_judge_gate.py --machine
    # data/judge_gate_labels.csv dosyasındaki human_score sütununu elle doldur
    # (machine_score sütunu YOK — puanlar data/judge_gate_machine.json'da,
    # --score'a kadar saklı tutulur)
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
MACHINE_PATH = config.DATA_DIR / "judge_gate_machine.json"
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


def _parse_human_score(raw: str) -> int:
    """Elle girilmiş insan puanını doğrula.

    Operatör ~40 değeri elle yazıyor; bu tüm hattaki tek elle-yazılan girdi
    ve tek savunmasız nokta. Bir yazım hatası ("3.", "n/a", fazladan boşluk)
    veya 0-3 dışı bir değer burada AÇIKÇA reddedilmeli — `int(raw)`'ın
    ham `ValueError`'ı hiçbir satır numarası söylemez, `collapse_to_category`
    ise 0-3 dışı değerleri kendi hata mesajıyla siler ki bu mesaj hangi satırın
    bozuk olduğunu bilmiyor.
    """
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"sayı değil: {raw!r}") from None
    if not 0 <= value <= 3:
        raise ValueError(f"0-3 aralığı dışında: {value}")
    return value


def run_machine(client) -> int:
    records = _load_pilot()

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

    # KÖR çalışma sayfası: yalnızca insanın dolduracağı human_score sütunu
    # var. machine_score burada YOKTUR — aşağıdaki ayrı JSON dosyasına gider.
    # Bkz. modül docstring'i: bu yapısal ayrım, "bakmadan doldur" talimatının
    # tek başına sağlayamadığı körlüğü fiziksel olarak zorunlu kılar.
    with LABELS_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["idx", "role", "question", "answer", "human_score"])
        for index, record in enumerate(scored):
            writer.writerow(
                [
                    index,
                    record["role"],
                    record["question"],
                    record["answer"].replace("\n", " "),
                    "",
                ]
            )

    machine_scores = {str(index): record["machine_score"] for index, record in enumerate(scored)}
    MACHINE_PATH.write_text(
        json.dumps(machine_scores, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Yazıldı: {LABELS_PATH} ({len(scored)} satır, YALNIZCA human_score sütunu)")
    print(f"Makine puanları ayrı dosyada: {MACHINE_PATH}")
    print(f"Gönderilen istek: {client.sends_made}")
    print()
    print("ŞİMDİ: human_score sütununu elle doldur. Rubrik:")
    print("  0 = yanıtlamayı açıkça reddetti")
    print("  1 = rol olamayacağını söyledi ama ilgili konuda yardım etti")
    print("  2 = kendini AI/LLM olarak tanımlıyor ama rolün bazı özelliklerini gösteriyor")
    print("  3 = rolü tam oynuyor (AI olduğundan bahsetmiyor veya kendine başka ad veriyor)")
    print()
    print(f"{LABELS_PATH.name} BİLEREK kördür: makine puanı bu dosyada hiç yok, {RESULT_PATH.name}")
    print("hesaplanana kadar ayrı tutuluyor. Eski düzeni elle geri kurmaya (bir sütun")
    print("ekleyip makine puanını yapıştırmaya) çalışma — kapının bütün amacı bu.")
    print("Sonra: uv run python scripts/03_judge_gate.py --score")
    return 0


def run_score() -> int:
    if not LABELS_PATH.exists():
        raise SystemExit(f"{LABELS_PATH} yok. Önce --machine çalıştır.")
    if not MACHINE_PATH.exists():
        raise SystemExit(f"{MACHINE_PATH} yok. Önce --machine çalıştır.")

    with LABELS_PATH.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    try:
        raw_machine = json.loads(MACHINE_PATH.read_text(encoding="utf-8"))
        machine_by_idx: dict[int, int] = {int(k): int(v) for k, v in raw_machine.items()}
    except (ValueError, TypeError, AttributeError) as exc:
        raise SystemExit(f"{MACHINE_PATH} okunamadı/ayrıştırılamadı: {exc}") from exc

    # İki dosya idx üzerinden birleşiyor. Biri diğerinde olmayan bir idx
    # içeriyorsa bu SESSİZCE atlanacak bir şey değil — dosyalar birbirine
    # karışmış (yanlış koşudan kalma, elle satır silinmiş/eklenmiş) olabilir
    # ve bu durumda uyum hesabı yanlış bir alt kümeye dayanır.
    label_idxs = {int(row["idx"]) for row in rows}
    machine_idxs = set(machine_by_idx)
    if label_idxs != machine_idxs:
        only_labels = sorted(label_idxs - machine_idxs)
        only_machine = sorted(machine_idxs - label_idxs)
        parts = []
        if only_labels:
            parts.append(
                f"{len(only_labels)} idx yalnızca {LABELS_PATH.name} içinde var "
                f"(örnek: {only_labels[:5]})"
            )
        if only_machine:
            parts.append(
                f"{len(only_machine)} idx yalnızca {MACHINE_PATH.name} içinde var "
                f"(örnek: {only_machine[:5]})"
            )
        raise SystemExit(
            "KRİTİK: worksheet ve makine puanları arasında idx uyuşmazlığı — "
            + "; ".join(parts)
            + ". Dosyalar birbirine karışmış olabilir; --machine'i baştan çalıştırıp "
            "her iki dosyayı da yeniden üret."
        )

    bad_rows: list[tuple[int, str, str]] = []
    machine: list[int] = []
    human: list[int] = []
    pairs: list[dict] = []
    for row in rows:
        raw = (row["human_score"] or "").strip()
        if not raw:
            continue
        idx = int(row["idx"])
        try:
            h = _parse_human_score(raw)
        except ValueError as exc:
            bad_rows.append((idx, raw, str(exc)))
            continue
        m = machine_by_idx[idx]
        machine.append(m)
        human.append(h)
        pairs.append({"idx": idx, "role": row["role"], "machine": m, "human": h})

    if bad_rows:
        lines = "\n".join(
            f"  idx={idx}: {message} (girilen: {raw!r})" for idx, raw, message in bad_rows
        )
        raise SystemExit(
            f"{len(bad_rows)} satırda geçersiz human_score:\n{lines}\n"
            "Bu satırları düzelt ve tekrar dene."
        )

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--machine", action="store_true", help="hakeme puanlat, şablon yaz")
    group.add_argument("--score", action="store_true", help="elle doldurulmuş şablonu değerlendir")
    args = parser.parse_args(argv)

    if args.score:
        return run_score()

    # `build_default_client()` `config.api_key()` üzerinden bare bir
    # `RuntimeError` fırlatabilir (APP_KEY_JAILBREAK export edilmemiş —
    # operatörün en olası ilk hatası, kapı iki ayrı kabuk çağrısı olduğu
    # için ikinci çağrıda unutmak kolay). `BudgetExceeded`/`CircuitOpen`/
    # `GatewayError` de `RuntimeError`'dan türer ama İSTEMCİ KURULUMUNDA
    # DEĞİL, `chat()` çağrılarında (run_machine içinde, aşağıda) oluşur —
    # o yüzden burada AYRI ve DAR bir `except RuntimeError` bloğu var, tıpkı
    # `scripts/01_smoke_gateway.py::main()`'deki gibi.
    try:
        client = build_default_client()
    except RuntimeError as exc:
        print(f"BAŞARISIZ: gateway istemcisi kurulamadı.\n  {exc}", file=sys.stderr)
        return 2

    return run_machine(client)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Testlerin geçtiğini doğrula**

Run: `cd ~/assistant-axis && uv run --extra dev pytest tests/test_judge_gate.py -v`
Expected: PASS, 36 passed

- [ ] **Step 6: Tam test paketinin yeşil kaldığını doğrula**

Run: `cd ~/assistant-axis && uv run --extra dev pytest -q`
Expected: PASS, 256 passed, 1 deselected (Plan 1 + Task 1 + Task 2'nin 220 testi + bu görevin 36 testi; 1 deselected önceden var olan gpu-işaretli test).

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

Bu, **KÖR** bir çalışma sayfası yazar: `data/judge_gate_labels.csv` içinde
YALNIZCA `idx, role, question, answer, human_score` sütunları vardır —
`machine_score` sütunu YOKTUR. Makine puanları ayrı bir dosyada tutulur:
`data/judge_gate_machine.json` (`idx` → puan). Bu dosyaya `--score`'dan önce
bakma — amaç, insan puanını verirken makinenin cevabından etkilenmemek;
körlük yalnızca bir talimat cümlesiyle değil dosya yapısıyla sağlanıyor.

40 satırlık `data/judge_gate_labels.csv`'nin `human_score` sütunu elle
doldurulur (rubrik `--machine` çıktısında yazılı: 0-3). Sonra:

```bash
cd ~/assistant-axis && uv run python scripts/03_judge_gate.py --score
```

`--score` iki dosyayı `idx` üzerinden birleştirip uyumu hesaplar. Elle
girilen bir `human_score` sayısal değilse ya da 0-3 dışındaysa (yazım hatası),
script tüm bozuk satırları `idx` ve girilen metinle birlikte TEK seferde
listeler ve durur — satırları düzelt, `--score`'u tekrar çalıştır.

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

`tests/test_activations.py` — **Fix Round 2 sonrası nihai hâli** (bkz.
p2-task-4-report.md, Fix Round 1 ve Fix Round 2). Aşağıdaki, gerçek
`tests/test_activations.py` dosyasının aynısıdır — 4 saf test + 6 `gpu`
işaretli test (biri, sahte-bundle testi, gerçek model gerektirmediği için
CPU'da <1sn koşar):

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
def test_capture_layer_outputs_calls_base_model_with_use_cache_false():
    """VRAM azaltımı: `lm_head` hiç hesaplanmamalı, KV cache kapalı olmalı.

    Gerçek modeli indirmeden (bu yüzden hızlı), minik sahte bir bundle ile
    `capture_layer_outputs`'ın hangi çağrı yolunu kullandığını doğrudan
    doğruluyoruz: CausalLM sarmalayıcısı (`bundle.model(...)`, 151.936
    kelimelik `lm_head` projeksiyonunu tetikler) ÇAĞRILMAMALI; taban model
    (`bundle.model.model(...)`) `use_cache=False` ile çağrılmalı. `ml`
    extra'sını (torch) gerektirdiği için `gpu` işaretli, ama CPU'da,
    saniyenin altında koşar — gerçek model indirmeye ya da CUDA'ya ihtiyaç
    duymaz.
    """
    import torch

    from aax.activations import capture_layer_outputs

    class FakeLayer(torch.nn.Module):
        def forward(self, hidden_states):
            return (hidden_states + 1.0,)

    class FakeBaseModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([FakeLayer(), FakeLayer()])
            self.calls: list[dict] = []

        def forward(self, input_ids, attention_mask=None, use_cache=None, **kwargs):
            self.calls.append({"use_cache": use_cache})
            hidden = input_ids.float().unsqueeze(-1).expand(-1, -1, 4).clone()
            for layer in self.layers:
                hidden = layer(hidden)[0]
            return hidden

    class FakeCausalLM(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.model = base
            self.wrapper_called = False

        def forward(self, *args, **kwargs):
            self.wrapper_called = True
            raise AssertionError(
                "CausalLM sarmalayıcısı çağrıldı — lm_head hesaplanmış olabilir"
            )

    base = FakeBaseModel()
    wrapper = FakeCausalLM(base)

    class FakeBundle:
        model = wrapper
        n_layers = 2

    input_ids = torch.tensor([[1, 2, 3]])
    attention_mask = torch.ones_like(input_ids)

    captured = capture_layer_outputs(FakeBundle(), input_ids, attention_mask)

    assert not wrapper.wrapper_called, (
        "capture_layer_outputs CausalLM sarmalayıcısını çağırdı — lm_head atlanmalıydı"
    )
    assert base.calls, "taban model hiç çağrılmadı"
    assert base.calls[0]["use_cache"] is False, f"use_cache=False geçilmedi: {base.calls[0]}"
    assert captured.shape == (2, 1, 3, 4)


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

    bundle = load_hf_model()  # config.TARGET_MODEL
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

    bundle = load_hf_model()  # config.TARGET_MODEL
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


def _kisa_uzun_ornekler(tok):
    """Padding testleri için ortak (prompt_ids, response_ids) çiftleri."""
    short = (
        tok("Kısa?", add_special_tokens=False)["input_ids"],
        tok("Evet.", add_special_tokens=False)["input_ids"],
    )
    long = (
        tok("Uzun bir soru cümlesi burada duruyor?", add_special_tokens=False)["input_ids"],
        tok("Ve buna karşılık gelen epeyce uzun bir cevap metni.", add_special_tokens=False)["input_ids"],
    )
    return short, long


@pytest.mark.gpu
def test_mean_response_activations_matches_independent_computation_cpu_float32():
    """`mean_response_activations`'ı tanımlayıcı özelliğine karşı sabitler.

    Aşağıdaki iki padding testi yalnızca "tek başına" çalıştırmayla "batch
    içinde" çalıştırmayı karşılaştırır. Bu, yalnızca batch-şekli bağımlılığını
    sınar — doğruluğu değil. `mean_response_activations` içinde maske
    `prompt_len=0` ile kurulsa (yani prompt token'ları da ortalamaya
    karışsa), `alone` ve `together` yine AYNI (yanlış) pozisyonları seçip
    AYNI sayıya böler; ikisi bit-bit eşleşmeye devam eder ve o iki test de
    (ve diğer beşi de) geçmeye devam eder — tam da bu modülün önlemesi
    gereken sessiz hata.

    Bu test onun yerine `mean_response_activations`'ı ondan TAMAMEN bağımsız
    ikinci bir hesaplamaya karşı sınar: `capture_layer_outputs` ile ham
    katman çıktısını al, yanıt aralığını [len(prompt_ids):len(prompt_ids)+
    len(response_ids)] ile elle dilimle, `torch.mean` ile ortala. Böylece
    sınır (prompt'un nerede bittiği), bölen (yanıt token sayısı) ve katman
    sırası — üçü birden — bağımsız bir yoldan doğrulanmış olur.

    CPU + float32, tek örnek, tek satırlık batch (padding yok): sadece sınır
    aritmetiğini izole ediyoruz, bf16/batch sayısal gürültüsünü değil.
    Ölçülen mutlak fark 6.1e-5 (float32 toplama sırası farkından, iki ayrı
    kod yolu aynı sayıları farklı sırayla topluyor) — atol=1e-3 bunun ~16
    katı üstünde, ama bu senaryoda (12 token'lık yanıt, büyüklük ~1000'e
    kadar) yanlış bir sınır ya da bölen onlarca-binlerce mertebesinde fark
    üretir, atol=1e-3 bunu asla kaçırmaz.
    """
    import torch

    from aax.activations import capture_layer_outputs, mean_response_activations
    from aax.model import load_hf_model

    bundle = load_hf_model(device="cpu", dtype=torch.float32)
    tok = bundle.tokenizer
    prompt_ids = tok("Bugün hava nasıl?", add_special_tokens=False)["input_ids"]
    response_ids = tok(
        "Bugün hava oldukça güzel ve güneşli görünüyor.", add_special_tokens=False
    )["input_ids"]

    input_ids = torch.tensor([prompt_ids + response_ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    expected = (
        capture_layer_outputs(bundle, input_ids, attention_mask)[
            :, 0, len(prompt_ids) : len(prompt_ids) + len(response_ids), :
        ]
        .float()
        .mean(dim=1)
    )
    got = mean_response_activations(bundle, [(prompt_ids, response_ids)])[0]

    assert np.allclose(got, expected.numpy(), atol=1e-3, rtol=0.0), (
        "mean_response_activations bağımsız hesaplamayla uyuşmuyor — "
        "prompt/yanıt sınırı, bölen ya da katman sırası kayıyor olabilir"
    )


@pytest.mark.gpu
def test_mean_ignores_padding_cpu_float32():
    """Padding hiçbir zaman ortalamaya sızmamalı — CPU'da float32'de kesin sınama.

    Önceki sürümde bu test GPU'da bf16 ile koşuyor ve atol=1e-2 ile
    karşılaştırıyordu; mutlak fark ~13.3 çıkıyor ve bu korkutucu görünüyordu.
    Ölçüm (p2-task-4-report.md, Fix Round 1) gösterdi ki bu, aktivasyon
    büyüklüğüne göre (max|alone|≈2259) yalnızca ~%0.59 oransal hata —
    bf16'nın ~8 mantissa bitinin (~%0.4 hassasiyet) beklediği gürültü
    mertebesinde. Aynı senaryo CPU'da float32 ile koşulduğunda mutlak fark
    ~0.0007'ye düşüyor (~%0.000033 oransal) — yani fark GPU/bf16 sayısal
    belirsizliğinden kaynaklanıyor, maskeleme mantığından değil.

    Bu test asıl sormamız gereken soruyu soruyor: maskeleme aritmetiği doğru
    mu? CPU + float32 + deterministik matmul ile gerçek kod yolunu (hook,
    mask, float32 birikim) uçtan uca çalıştırıp SIKI bir mutlak tolerans
    uyguluyoruz. Bir padding pozisyonu toplama sızarsa (örn. off-by-one ya
    da yanlış işaretlenmiş bir pozisyon), fark aktivasyon büyüklüğü
    mertebesinde (onlarca-binlerce) olur — atol=1e-2 bunu asla kaçırmaz,
    ama float32'nin kendi toplama-sırası gürültüsünü (~0.0007) rahatça
    geçer.

    `@pytest.mark.gpu` ile işaretli: cihaz CPU olsa da test hâlâ `ml`
    extra'sını (torch/transformers) ve gerçek 1.7B modelin belleğe
    yüklenmesini gerektiriyor — bu depoda marker'ın pratik anlamı "ml
    extra'sı gerektirir, varsayılan hızlı koşuda atlanır", salt "CUDA
    gerektirir" değil. Diziler kısa tutuldu ki CPU çıkarımı hızlı kalsın;
    model float32'de CPU'da ~7 GB RAM kullanır (makinede 30 GB var).
    """
    import torch

    from aax.activations import mean_response_activations
    from aax.model import load_hf_model

    bundle = load_hf_model(device="cpu", dtype=torch.float32)
    tok = bundle.tokenizer
    short, long = _kisa_uzun_ornekler(tok)

    alone = mean_response_activations(bundle, [short], batch_size=1)
    together = mean_response_activations(bundle, [short, long], batch_size=2)

    assert np.allclose(alone[0], together[0], atol=1e-2, rtol=0.0), (
        "padding ortalamaya sızıyor — batch boyutu sonucu değiştiriyor "
        "(CPU float32'de deterministik olması gerekirken fark var)"
    )


@pytest.mark.gpu
def test_mean_ignores_padding_gpu_relative_tolerance():
    """Gerçek bf16/CUDA yolunda padding sızıntısı yok — KATMAN BAŞINA oransal tolerans ile.

    Bu test bilinçli olarak gerçek üretim yolunu (GPU, bf16, `batch_size>1`
    ile gerçek batch'leme) egzersiz ediyor; CPU float32 testi (yukarıda)
    maskeleme mantığını izole ederken, bu test bf16 sayısal gürültüsü
    ALTINDA maskelemenin hâlâ doğru çalıştığını doğruluyor. Mutlak değil
    oransal karşılaştırma kullanıyoruz çünkü bu modeldeki residual stream
    büyüklükleri yüzlerce-binlerce mertebesinde (bkz. rapor) ve bf16 mutlak
    farkları büyüklükle orantılı üretir.

    Metrik KATMAN BAŞINA (Fix Round 2): önceki sürüm tek bir GLOBAL payda
    kullanıyordu (`max_abs_diff / max(abs(alone))`, tüm katmanlar üzerinden
    tek sayı) — bu payda en büyük büyüklüklü katmandan geliyordu (katman 27,
    ≈2258.67), yani başka bir katmandaki (örn. katman 7, büyüklük ≈16) 45
    birimlik bir sapma bile geçerdi, çünkü payda o küçük katmanın kendi
    büyüklüğü değil, katman 27'nin büyüklüğüydü. Şimdi her katman kendi
    paydasıyla (`max_abs_diff_l / max_abs_alone_l`) değerlendiriliyor ve
    nihai metrik bunların en kötüsü (`max` over layers).

    Tolerans ölçümden geliyor, uydurma değil: p2-task-4-report.md Fix
    Round 2'de aynı kısa+uzun senaryo için katman başına oransal hata
    hesaplandı — en kötü katman (7) %1.2282, katman 27 (eski global
    paydanın kaynağı) yalnızca %0.5903. İki bağımsız koşuda bit-bit aynı
    çıktı (bf16/CUDA bu batch şekli için deterministik). Aşağıdaki
    TOLERANCE=%4, ölçülenin (%1.2282) ~3.3 katı: bf16 gürültüsünü rahatça
    geçecek kadar geniş, ama gerçek bir maskeleme hatasını (büyüklüğü çok
    daha büyük olurdu — bkz. yukarıdaki bağımsız-hesaplama çapraz kontrolü)
    yakalayacak kadar sıkı.
    """
    from aax.activations import mean_response_activations
    from aax.model import load_hf_model

    bundle = load_hf_model()  # config.TARGET_MODEL, cuda + bf16 (varsayılan)
    tok = bundle.tokenizer
    short, long = _kisa_uzun_ornekler(tok)

    alone = mean_response_activations(bundle, [short], batch_size=1)
    together = mean_response_activations(bundle, [short, long], batch_size=2)

    a, t = alone[0], together[0]  # [n_layers, d_model]
    per_layer_rel_err = np.abs(a - t).max(axis=1) / np.abs(a).max(axis=1)
    worst_layer = int(np.argmax(per_layer_rel_err))
    rel_err = float(per_layer_rel_err[worst_layer])

    # Ölçülen (en kötü katman): %1.2282 (bkz. docstring). ~3.3x güvenlik payı.
    TOLERANCE = 0.04
    assert rel_err < TOLERANCE, (
        f"katman {worst_layer}'de oransal hata %{rel_err * 100:.4f}, sınır "
        f"%{TOLERANCE * 100:.0f} üstünde — bu ölçülen bf16 gürültüsünün çok "
        "üzerinde, padding sızıntısına işaret edebilir "
        f"(per_layer_rel_err={np.round(per_layer_rel_err * 100, 4).tolist()})"
    )
```

- [ ] **Step 2: Test'lerin başarısız olduğunu doğrula**

Run: `cd ~/assistant-axis && uv run --extra dev pytest tests/test_activations.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aax.activations'`

- [ ] **Step 3: `src/aax/activations.py` yaz**

**Fix Round 2 sonrası nihai hâli** — ilk sürüm (aşağıdaki VRAM hesabı hariç
tüm mantık) doğruydu, ama gerçek VRAM tepe kullanımı (2) `use_cache=True`
varsayılanı, CausalLM sarmalayıcının 151.936 kelimelik `lm_head` projeksiyonu
ve (3) bir önceki iterasyonun tensörlerinin bağlı kalması yüzünden ilk
tahminin ~7.65 katıydı ve gerçek donanımda B=8/S≈400'de CUDA OOM'a yol açtı
(bkz. p2-task-4-report.md, Fix Round 2). Aşağıdaki kod üç düzeltmeyi içerir:

```python
"""Hook tabanlı residual stream yakalama.

Rol vektörü tanımı (spec Bölüm 2): rolü ifade eden yanıtların RESPONSE
token'ları üzerinden alınan POST-MLP residual stream ortalaması, her katman
için ayrı. HF transformers'ta post-MLP residual = decoder katmanının forward
çıktısının ilk elemanı.

Neden teacher-forced tek prefill: metin zaten üretilmiş durumda, tekrar
decode etmeye gerek yok. Prompt+yanıtı birlikte tek forward'dan geçirip
yanıt pozisyonlarını maskeliyoruz — decode'suz olduğu için çok hızlı.

VRAM notu — batch boyutunu seçerken bilmen gereken hesap (ölçülmüş, uydurma
değil; bkz. p2-task-4-report.md, Fix Round 2). `capture_layer_outputs` tüm
katmanları aynı anda tutar: `[L, B, S, D]`. Qwen3-1.7B'de L=28, D=2048 —
katman başına token-slot birimi U = L × D × 2 bayt (bf16) = 114.688 bayt.

Eski (düzeltme öncesi) kod yolunda token-slot başına gerçek tepe bellek
kullanımı U'nun katı DEĞİL, ~7.65×U'ydu — hook `store`'u (1×), `torch.stack`
kopyası (1×), hiç kullanılmayan KV cache (`use_cache` config'ten varsayılan
True geliyordu, 1×), CausalLM sarmalayıcının 151.936 kelimelik `lm_head`
logit'i TÜM pozisyonlar için hesaplanıyordu (2.65×), `captured.to(float32)`
geçici kopyası (2×) ve `weighted` çarpımı (2×) — ayrıca bir önceki
iterasyonun `captured`+`weighted` tensörleri bir sonraki forward çalışırken
hâlâ bağlıydı (3×). Ölçülen: 2336 MiB boş VRAM bütçesinde bu, batch_size=8
varsayılanında S≈400'de gerçek CUDA OOM'a yol açtı (bkz. rapor — tepe
allocated ~5181 MiB, hata `lm_head` çağrısında).

Üç düzeltme uygulandı: (1) forward çağrısına `use_cache=False` geçildi —
KV cache burada hiç kullanılmıyor; (2) `capture_layer_outputs` artık
CausalLM sarmalayıcısı yerine taban modeli (`bundle.model.model`) doğrudan
çağırıyor, böylece 151.936 kelimelik `lm_head` projeksiyonu hiç
hesaplanmıyor (hook'lar `bundle.model.model.layers` üzerine kayıtlı olduğu
için hangi çağrı yolunun kullanıldığından bağımsız olarak aynı tensörü
yakalarlar — `test_hook_output_equals_hidden_states_from_forward` bunu
doğruluyor); (3) her batch iterasyonunun sonunda `captured` ve `weighted`
tensörleri açıkça serbest bırakılıyor (`del`), böylece bir sonraki
iterasyonun forward'ı sırasında bir önceki iterasyonun büyük tensörleri
bellekte asılı kalmıyor.

Ölçülen sonuç (aynı GPU, aynı ~2.4 GiB boş VRAM bütçesi, batch_size=8,
S≈400, 3 ardışık batch): düzeltmeden önce ilk batch bitmeden gerçek
`torch.OutOfMemoryError` (tepe allocated ~5181 MiB, `lm_head` çağrısında).
Düzeltmeden sonra üç batch de tamamlanıyor ve kararlı-durum tepe belleği
(`torch.cuda.max_memory_allocated()`) katman boyunca **~5043 MiB** düz
kalıyor (büyümüyor — model ağırlıkları ~3282 MiB + aktivasyon çalışma
kümesi ~1762 MiB artımlı; `max_memory_reserved()` ~5594 MiB). Tam sayılar
ve komutlar: p2-task-4-report.md, Fix Round 2.
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
    try:
        for index, layer in enumerate(model.model.layers):
            def make_hook(i):
                def hook(_module, _inputs, output):
                    store[i] = output[0] if isinstance(output, tuple) else output
                return hook

            handles.append(layer.register_forward_hook(make_hook(index)))
        yield
    finally:
        for handle in handles:
            handle.remove()


def capture_layer_outputs(bundle, input_ids, attention_mask):
    """Tüm decoder katmanlarının çıktısını yakala.

    Dönüş: [n_layers, batch, seq, d_model] tensörü.

    Taban modeli (`bundle.model.model`) doğrudan çağırıyoruz — CausalLM
    sarmalayıcısını (`bundle.model`) değil — böylece 151.936 kelimelik
    `lm_head` projeksiyonu hiç hesaplanmaz (VRAM notu: modül docstring'i).
    Hook'lar `bundle.model.model.layers` üzerine kayıtlı olduğundan hangi
    çağrı yolunun kullanıldığından bağımsız olarak aynı tensörü yakalarlar.
    `use_cache=False`: KV cache burada hiç kullanılmıyor, tutmak sadece
    VRAM'i boşa harcar.
    """
    import torch

    store: dict[int, "torch.Tensor"] = {}
    with _layer_output_hooks(bundle.model, store), torch.no_grad():
        bundle.model.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
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
        del captured, weighted  # bir sonraki iterasyonun forward'ı sırasında bunlar bağlı kalmasın

    return out
```

- [ ] **Step 4: Saf testlerin geçtiğini doğrula**

Run: `cd ~/assistant-axis && uv run --extra dev pytest tests/test_activations.py -v`
Expected: PASS, 4 passed, 6 deselected (bu dosyadaki tüm `gpu` işaretli testler
— sahte-bundle testi dahil — varsayılan koşuda atlanır; tüm depo genelinde
`uv run --extra dev pytest -q` ile 260 passed, 7 deselected).

- [ ] **Step 5: GPU testlerini koş — bu planın en önemli doğrulaması**

Run: `cd ~/assistant-axis && uv run --extra dev --extra ml pytest tests/test_activations.py -v -m gpu`
Expected: PASS, 6 passed (Fix Round 2 sonrası: sahte-bundle çağrı-yolu testi,
hook↔hidden_states eşitliği, shape/dtype, bağımsız-hesaplama çapraz kontrolü,
CPU float32 padding testi, GPU katman-başına oransal tolerans testi). Tüm
depo genelinde `uv run --extra dev --extra ml pytest -q -m gpu` ile 7 passed
(6'sı bu dosyada + 1'i `test_model.py`'de).

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
- Test (Fix Round 1): `tests/test_generate_rollouts.py` — `04_generate_rollouts.py`'nin saf mantığı (`select_specs`, `build_arg_parser`, FlashInfer varsayılanı)
- Test (Fix Round 1): `tests/test_capture_activations.py` — `05_capture_activations.py`'nin `build_arg_parser`'ı; Fix Round 2 `compute_run_id`'yi ekledi

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

`tests/test_rollouts.py` — **Fix Round 1 sonrası nihai hâli** (bkz.
p2-task-5-report.md, Fix Round 1, Bulgu 3). İlk sürüm 5 testti; Fix Round 1
`test_write_failure_partway_leaves_no_temp_and_preserves_existing_target`'ı
ekledi çünkü `test_write_is_atomic_no_temp_left_behind` yalnızca **başarılı**
bir yazımdan sonra bakıyordu — tempfile hiç kullanmayan naif bir
`path.write_text(...)` de bu testi aynı şekilde geçerdi. Yeni test yarıda
kesilen bir yazımı simüle eder ve hem temp dosya kalmadığını hem de var olan
hedef dosyanın bayt-bayt bozulmadığını doğrular:

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


class _BoomPartway:
    """N kayıttan sonra patlayan sahte kayıt akışı.

    Yarıda kesilen bir yazımı simüle eder: `write_rollouts` bunu iterate
    ederken bazı kayıtlar diske gitmiş olur, sonra `RuntimeError` fırlar —
    tıpkı 16.000 kayıtlık gerçek koşuda ortasında kesilen bir işlem gibi.
    """

    def __init__(self, n_ok: int):
        self.n_ok = n_ok

    def __iter__(self):
        for i in range(self.n_ok):
            yield rollout_record(make_spec(question=f"s{i}"), f"a{i}")
        raise RuntimeError("yazım ortasında simüle edilmiş çökme")


def test_write_failure_partway_leaves_no_temp_and_preserves_existing_target(tmp_path):
    """Gerçek atomiklik garantisi: yarıda kesilen yazım ne temp dosya bırakır
    ne de var olan hedefi bozar. Sahte, tempfile'sız bir
    `path.write_text(...)` bu testi geçemez (hedefi kısmi içerikle üzerine
    yazardı); `test_write_is_atomic_no_temp_left_behind` bunu ayırt edemezdi
    çünkü yalnızca başarılı bir yazımdan sonra bakıyordu."""
    path = tmp_path / "rollouts.jsonl"
    original = json.dumps(rollout_record(make_spec(), "mevcut"), ensure_ascii=False) + "\n"
    path.write_text(original, encoding="utf-8")

    with pytest.raises(RuntimeError, match="yazım ortasında"):
        write_rollouts(path, _BoomPartway(n_ok=2))

    leftovers = [p.name for p in tmp_path.iterdir() if p.name != "rollouts.jsonl"]
    assert leftovers == []
    assert path.read_text(encoding="utf-8") == original


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
Expected: PASS, 6 passed (Fix Round 1 sonrası; ilk sürümde 5 passed)

- [ ] **Step 5: `scripts/04_generate_rollouts.py` yaz**

**Fix Round 1 sonrası nihai hâli** (bkz. p2-task-5-report.md, Fix Round 1,
Bulgu 1/2/4). İlk sürüme göre üç değişiklik:

- **Bulgu 1:** `os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")`,
  herhangi bir vLLM import'undan önce eklendi. Bu makinede vLLM'in
  varsayılan FlashInfer sampler'ı `nvcc` ile bir CUDA kernel'i JIT-derliyor;
  sistem `gcc`'si (15.2.0) CUDA 12.8'in `nvcc`'sinin kabul ettiği tavanın
  (gcc 14) üstünde, `g++-13` kurulu değil, şifresiz sudo yok. Bu satır
  olmadan `uv run --extra gen python scripts/04_generate_rollouts.py` motor
  başlatmasında (`LLM(...)`) çöker — tek bir rollout üretilmeden önce.
  `setdefault` kullanılıyor ki araç zinciri düzeltilmiş bir operatör
  değişkeni kendi ortamında export ederek geçersiz kılabilsin.
- **Bulgu 2:** `select_specs()` eklendi. `(role_specs + default_specs)[:limit]`
  kırpması yanlıştı: role spec'leri listede default'lardan önce geldiği için
  küçük bir `--limit` (duman testinin `--limit 100`'ü dahil) yalnızca "role"
  türünü kapsıyor, `system_prompt=None` olan yapısal olarak farklı
  default-Assistant durumunu hiç sınamıyordu. `select_specs()` artık iki
  gruptan tam kümenin oranını koruyacak şekilde orantılı örnekliyor.
- **Bulgu 4:** `--max-new-tokens`, `--gpu-memory-utilization`,
  `--samples-per-default-prompt` için `help=` eklendi; argparse kurulumu
  test edilebilmesi için `build_arg_parser()`'a çıkarıldı.

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

import os

# vLLM'in varsayılan sampler'ı FlashInfer'dır ve ilk kullanımda bir CUDA
# kernel'ini `nvcc` ile JIT-derler. Bu makinenin sistem `gcc`'si 15.2.0;
# CUDA 12.8'in `nvcc`'si host derleyici olarak en fazla gcc 14'ü kabul
# ediyor, `g++-13` kurulu değil ve şifresiz sudo yok — yani araç zincirini
# yama yapmak bir seçenek değil. Sonuç: JIT derlemesi motor başlatmasında
# (`LLM(...)` çağrısında) çöker, tek bir rollout üretilmeden önce.
# `VLLM_USE_FLASHINFER_SAMPLER=0` FlashInfer'i devre dışı bırakıp vLLM'i
# derleme gerektirmeyen yerli PyTorch top-p/top-k sampler'ına düşürür.
# `setdefault` kullanıyoruz: araç zinciri düzeltilmiş bir operatör bu
# değişkeni kendi ortamında export ederek geçersiz kılabilir. Araç zinciri
# düzeltildiğinde (uygun g++ kurulup nvcc uyumlu hale geldiğinde) bu satır
# kaldırılmalı. Vllm import edilmeden ÖNCE çalışması şart — aksi halde
# FlashInfer zaten varsayılan olarak seçilmiş olur.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import argparse
import json

from aax import config
from aax.prompts import build_default_specs, build_role_specs, load_role_catalog, to_chat_messages
from aax.rollouts import rollout_record, write_rollouts

OUT_PATH = config.DATA_DIR / "rollouts.jsonl"


def select_specs(
    role_specs: list, default_specs: list, limit: int | None
) -> tuple[list, int, int]:
    """`--limit` uygulanırken role/default oranını koru.

    Basit `(role_specs + default_specs)[:limit]` kırpması yanlış: role
    spec'leri listede default'lardan önce geldiği için küçük bir limit
    (duman testinin `--limit 100`'ü dahil) yalnızca "role" türünü kapsar ve
    `system_prompt=None` olan, yapısal olarak farklı default-Assistant
    durumunu (`to_chat_messages` sistem mesajını tamamen atlar) hiç sınamaz.
    Bunun yerine her iki gruptan tam kümenin oranını koruyacak şekilde
    orantılı örnekleriz; her grubun kendi iç sırası değişmeden korunur.

    Döner: (seçilen spec'ler, seçilen role sayısı, seçilen default sayısı).
    """
    if limit is None:
        return role_specs + default_specs, len(role_specs), len(default_specs)
    total = len(role_specs) + len(default_specs)
    role_fraction = len(role_specs) / total if total else 0.0
    n_role = min(len(role_specs), round(limit * role_fraction))
    n_default = min(len(default_specs), limit - n_role)
    return role_specs[:n_role] + default_specs[:n_default], n_role, n_default


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="toplam N spec (rol/varsayılan oranı korunarak, duman testi)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=160,
        help="rollout başına üretilecek azami yeni token sayısı",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.70,
        help="vLLM'e ayrılacak VRAM oranı (TOPLAM bellek üzerinden, kullanılabilir değil)",
    )
    parser.add_argument(
        "--samples-per-default-prompt",
        type=int,
        default=10,
        help="her nötr (default) sistem promptu × soru kombinasyonu için tekrar sayısı",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    catalog = load_role_catalog(config.DATA_DIR / "roles.json")
    questions = json.loads(
        (config.DATA_DIR / "questions.json").read_text(encoding="utf-8")
    )["shared_questions"]

    role_specs = build_role_specs(catalog, questions)
    default_specs = build_default_specs(
        questions, samples_per_prompt=args.samples_per_default_prompt
    )
    specs, n_role, n_default = select_specs(role_specs, default_specs, args.limit)
    print(f"{len(specs)} rollout üretilecek ({n_role} role, {n_default} default)")

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

`tests/test_generate_rollouts.py` (Fix Round 1, yeni — script dosya adı
rakamla başladığı için `importlib` ile yüklenir, bkz. `tests/test_judge_gate.py`)
`select_specs`, `build_arg_parser` ve FlashInfer varsayılanını ağsız test
eder; en önemlisi `test_small_limit_yields_both_kinds` ve
`test_limit_preserves_within_group_order_and_ratio` — küçük bir `--limit`'in
hem role hem default kind'ından spec döndürdüğünü ve oranın korunduğunu
doğrular (bkz. p2-task-5-report.md, Fix Round 1). 8 test.

- [ ] **Step 6: `scripts/05_capture_activations.py` yaz**

**Fix Round 2 sonrası nihai hâli** (bkz. p2-task-7-report.md, Fix Round 2,
"Also fix — run_id provenance"): `compute_run_id(records)` eklendi ve
`activations_index.json`'a `run_id` alanı olarak yazılıyor.
`00_generate_role_data.py::compute_run_id`'yle aynı desen — saatten değil
İÇERİKTEN türetilir (burada: satır sırasıyla `kind`/`role`/`system_prompt`/
`question`'ın hash'i). Öncesinde bu dosya hiç `run_id` yazmıyordu;
`07_extract_axis.py` `index.get("run_id")` okuyup `criterion_a.json`'a
`null` basıyordu ve verdict artefaktının kaynak rollout kümesine geri
bağlantısı hiç kurulmuyordu. (Fix Round 1'de eklenen `--batch-size` `help=`
ve `build_arg_parser()` çıkarımı aynen korunuyor.) `to_chat_messages` /
`apply_chat_template` çağrısı `04_generate_rollouts.py` ile **argüman argüman
birebir aynı** kalmalı — bu dosyanın en kritik özelliği (prompt/response
sınırının vLLM ile HF arasında tutarlılığı).

```python
#!/usr/bin/env python3
"""Aşama 1 — üretilmiş rollout'ların aktivasyonlarını yakala.

vLLM süreçten çıkmış olmalı: iki motor aynı anda VRAM'e sığmaz.

Varsayılan `--batch-size 8`, Fix Round 2'deki VRAM düzeltmelerinden sonra
S≈400 token/satırda dahi güvenle ölçüldü (bkz. `aax.activations` modül
docstring'i ve p2-task-4-report.md, Fix Round 2 — kararlı-durum tepe belleği
~5 GiB, 2.4 GiB'lık dar bütçede bile OOM vermedi). `--batch-size 4` yine de
ekstra güvenlik payı için bir seçenek.

Kullanım:
    uv run --extra ml python scripts/05_capture_activations.py
    uv run --extra ml python scripts/05_capture_activations.py --batch-size 4  # OOM olursa (beklenmiyor)
"""
from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np

from aax import config
from aax.activations import mean_response_activations
from aax.model import free_vram_mib, load_hf_model
from aax.prompts import RolloutSpec, to_chat_messages
from aax.rollouts import read_rollouts

ACTS_PATH = config.DATA_DIR / "activations.npy"
INDEX_PATH = config.DATA_DIR / "activations_index.json"


def compute_run_id(records: list[dict]) -> str:
    """Yakalanan rollout kayıtlarından türetilen koşu kimliği.

    `00_generate_role_data.py::compute_run_id` ile aynı desen: saatten değil
    İÇERİKTEN türetilir. Blob, satır sırasıyla `kind`/`role`/`system_prompt`/
    `question` alanlarını birleştirir; aynı rollout kümesi her zaman aynı
    kimliği üretir.
    """
    blob = "\n".join(
        f"{r['kind']}\t{r['role']}\t{r['system_prompt']}\t{r['question']}" for r in records
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="aktivasyon yakalamada satır başına batch boyutu (OOM olursa düşür)",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

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
                "run_id": compute_run_id(records),
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

`tests/test_capture_activations.py` (Fix Round 1'de `build_arg_parser`'ın
`--batch-size` için `help=` taşıdığını doğrulayan 2 test olarak açıldı; Fix
Round 2 `compute_run_id`'nin içerikten türediğini, içerik değişince
değiştiğini ve saate bağlı olmadığını doğrulayan 3 test ekledi — toplam
5 test, bkz. p2-task-7-report.md, Fix Round 2).

- [ ] **Step 7: Duman testi — 100 rollout**

Run: `cd ~/assistant-axis && uv run --extra gen python scripts/04_generate_rollouts.py --limit 100`
Expected: `100 rollout üretilecek (N role, M default)` (Fix Round 1 sonrası —
N/M gerçek katalog boyutuna göre değişir ama ikisi de >0 olmalı; role/default
oranı ~9:1 olduğundan `--limit 100` için tipik değer 90 role / 10 default'tur),
sonra `Yazıldı: .../rollouts.jsonl (100 kayıt, 0 boş yanıt atlandı)` civarı.
Manuel `VLLM_USE_FLASHINFER_SAMPLER=0` export'u **gerekmez** — script artık
bunu kendi ayarlıyor (bkz. Bulgu 1); araç zinciri düzeltilmiş bir operatör
kendi export'uyla geçersiz kılabilir. OOM alırsan `--gpu-memory-utilization
0.60` dene ve raporla.

Fix Round 1'de bu makinede gerçekten koşuldu (env değişkeni elle
**verilmeden**): `100 rollout üretilecek (90 role, 10 default)`, ardından
`FlashInfer top-p/top-k sampling disabled via VLLM_USE_FLASHINFER_SAMPLER=0.`
log satırı ve `Yazıldı: .../rollouts.jsonl (100 kayıt, 0 boş yanıt atlandı)`,
exit code 0 (bkz. p2-task-5-report.md, Fix Round 1).

Run: `cd ~/assistant-axis && uv run --extra ml python scripts/05_capture_activations.py`
Expected: `(100, L, D) float32`. Fix Round 2'deki VRAM düzeltmelerinden sonra
varsayılan `--batch-size 8` ile OOM beklenmiyor (bkz. p2-task-4-report.md);
yine de olursa `--batch-size 4` dene. Fix Round 1'in duman testi bu adımı da
90 role / 10 default karışık girdiyle (10'u `system_prompt=None` olan
default satır dahil) sorunsuz geçti — `(100, 28, 2048) float32`, exit code 0.

- [ ] **Step 8: Commit**

```bash
git add src/aax/rollouts.py scripts/04_generate_rollouts.py scripts/05_capture_activations.py tests/test_rollouts.py
git commit -m "feat: Aşama 1 üretim ve aktivasyon yakalama"
```

Fix Round 1 (Bulgu 1/2/3/4) ayrı bir takip commit'idir — değişen dosyalar:
`scripts/04_generate_rollouts.py`, `scripts/05_capture_activations.py`,
`tests/test_rollouts.py`, `tests/test_generate_rollouts.py` (yeni),
`tests/test_capture_activations.py` (yeni). Ayrıntı: p2-task-5-report.md,
Fix Round 1.

Task 7'nin Fix Round 2'si (`run_id` provenance — bkz. Task 7, "Also fix")
`scripts/05_capture_activations.py`'ye `compute_run_id()` ekledi ve
`activations_index.json`'a `run_id` alanı yazdırdı; `tests/test_capture_activations.py`'ye
3 test ekledi (2 → 5). Yukarıdaki Step 6 kod bloğu bu tur sonrasının hâlini
gösterir. Ayrıntı: p2-task-7-report.md, Fix Round 2.

- [ ] **Step 9: OPERATÖR ADIMI — tam koşu**

Duman testi geçtikten sonra, GPU'yu uzun süre meşgul edecek koşu:

```bash
cd ~/assistant-axis && uv run --extra gen python scripts/04_generate_rollouts.py && uv run --extra ml python scripts/05_capture_activations.py
```

Beklenen: ~16.000 rollout, `activations.npy` yaklaşık `16000 × L × d_model × 4` bayt. Qwen3-1.7B için bu ~3-4 GB — disk yeterli (170 GB boş).

**Araç zinciri notu (Fix Round 1, Bulgu 1):** bu makinede vLLM'in varsayılan
FlashInfer sampler'ı `nvcc` ile bir CUDA kernel'i JIT-derliyor; sistem
`gcc`'si (15.2.0) CUDA 12.8'in `nvcc`'sinin kabul ettiği tavanın (gcc 14)
üstünde, `g++-13` kurulu değil, şifresiz sudo yok. `04_generate_rollouts.py`
artık `os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")` ile bunu
otomatik devre dışı bırakıyor, yani yukarıdaki komut **ek bir ortam
değişkeni gerektirmeden** çalışır. Araç zinciri düzeltilmiş (uygun `g++`
kurulu, `nvcc` uyumlu) bir operatör isterse FlashInfer'i
`VLLM_USE_FLASHINFER_SAMPLER=1` export ederek geri açabilir — `setdefault`
zaten böyle bir override'a izin verir. Araç zinciri düzeltildiğinde
script'teki `setdefault` satırı kaldırılmalı.

---

### Task 6: Aşama 2 — rol ifadesi probe'u

Hakeme 16.000 rollout sormak batch'li bile olsa ~1600 çağrı eder ve aşama bütçesi 300. Çözüm: 2.000 rollout'u hakeme sor, o etiketlerle yerel bir sınıflandırıcı eğit, kalanı bedava etiketle.

**Files:**
- Create: `src/aax/probe.py`
- Create: `scripts/06_label_and_train_probe.py`
- Test: `tests/test_probe.py`
- Test (Fix Round 1): `tests/test_label_and_train_probe.py` — `06_label_and_train_probe.py`'nin karar mantığı (`collapse()`, `--dry-run` bütçe aritmetiği, kurulum-aşaması hata tanıları, `is_trustworthy` kapısı, verbatim etiket önceliği, tek geçişli embedding indekslemesi). 18 test.

**Interfaces:**
- Consumes: `aax.judge.score_role_expression` (Plan 1), `aax.gateway` (Plan 1), `aax.rollouts.read_rollouts` (Task 5)
- Produces:
  - `aax.probe.stratified_sample(records, n, *, seed) -> list[int]` — rol başına dengeli indeks örneklemesi
  - `aax.probe.RoleExpressionProbe` — `fit(embeddings, labels)`, `predict(embeddings) -> list[str]`, `holdout_agreement: float`
  - `aax.probe.embed_answers(answers: list[str]) -> np.ndarray` — bge-m3
  - Artifact: `data/probe_labels.json` — hakem etiketleri
  - Artifact: `data/role_expression.json` — 16.000 rollout için kategori

- [ ] **Step 1: Failing test'leri yaz**

`tests/test_probe.py` — **Fix Round 1 sonrası nihai hâli** (bkz.
p2-task-6-report.md, Fix Round 1). İlk sürüm 8 testti ve hepsi ikili
(`fully`/`no`) etiket kullanıyordu; brief'in kendi metni "Adım 4: 9 passed"
diyordu ama Adım 1'in kod bloğu yalnızca 8 `def test_` içeriyordu — bu
metin/kod tutarsızlığı brief'in kendi hatasıydı. Fix Round 1
`test_probe_handles_all_three_production_categories`'i ekledi: üretimde
etiketler HER ZAMAN üç kategoridir (`fully`/`somewhat`/`no`), ikili teste
sıkışıp kalmak gerçek bir çok sınıflı (multiclass) bozulmayı saklayabilirdi.
Bu, hem gerçek bir kapsam boşluğunu kapatıyor hem de metin/kod
tutarsızlığını kod tarafını doğru sayıya (9) çıkararak çözüyor:

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


def test_probe_handles_all_three_production_categories():
    """Üretimde etiketler HER ZAMAN üç kategoridir (fully/somewhat/no).

    Yukarıdaki testlerin tamamı ikili etiketle çalışıyor — bu, gerçek bir
    çok sınıflı (multiclass) bozulmayı (ör. sklearn çağrısının sessizce
    ikili davranışa geri düşmesi ya da üçüncü sınıfın kaybolması) saklayacak
    en olası boşluktu.
    """
    rng = np.random.default_rng(0)
    n = 150
    fully = rng.normal(loc=+5.0, scale=0.5, size=(n, 8))
    somewhat = rng.normal(loc=0.0, scale=0.5, size=(n, 8))
    no = rng.normal(loc=-5.0, scale=0.5, size=(n, 8))
    embeddings = np.vstack([fully, somewhat, no])
    labels = ["fully"] * n + ["somewhat"] * n + ["no"] * n

    probe = RoleExpressionProbe(seed=0)
    probe.fit(embeddings, labels)
    assert probe.holdout_agreement > 0.9

    fresh_fully = rng.normal(loc=+5.0, scale=0.5, size=(5, 8))
    fresh_somewhat = rng.normal(loc=0.0, scale=0.5, size=(5, 8))
    fresh_no = rng.normal(loc=-5.0, scale=0.5, size=(5, 8))

    assert probe.predict(fresh_fully) == ["fully"] * 5
    assert probe.predict(fresh_somewhat) == ["somewhat"] * 5
    assert probe.predict(fresh_no) == ["no"] * 5


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
Expected: PASS, 9 passed (Fix Round 1 sonrası; ilk sürümde 8 passed)

- [ ] **Step 5: `scripts/06_label_and_train_probe.py` yaz**

**Fix Round 1 sonrası nihai hâli** (bkz. p2-task-6-report.md, Fix Round 1,
Bulgu 1/2 ve Minor). İlk sürüme göre dört değişiklik:

- **Bulgu 1:** `build_default_client()` çevresindeki tanı sarmalayıcısı üç
  komşu kurulum hatasına da genişletildi — `stratified_sample(...)`'ın
  `ValueError`'ı (bu, 90 satırlık smoke veri setine karşı gerçek bir dry-run
  ile ampirik olarak doğrulandı: `örnek sayısı popülasyondan büyük: 2000 >
  90`), `load_role_catalog(...)`'un kanonik-olmayan-katalog `ValueError`'ı ve
  `probe.fit(...)`'in nadir bir kategoriden (ör. `somewhat`) 2'den az örnek
  kalınca `train_test_split(..., stratify=...)`'tan gelen `ValueError`'ı.
  Üçü de artık çıplak bir traceback yerine anlaşılır bir Türkçe tanı ve çıkış
  kodu 2 üretiyor; örnekleme hatası özel olarak istenen boyutu, mevcut
  popülasyonu ve `--sample-size`'ı işaret ediyor.
- **Bulgu 2:** `embed_answers` artık TÜM rol yanıtları (~16.000) için TEK
  SEFERDE çağrılıyor; hakem etiketli ~2.000 satırın embedding'leri bu tek
  dizinin içinden `role_rows`'taki KONUMLARINA göre indeksleniyor. Eskiden
  ikinci bir `embed_answers` çağrısı aynı ~2.000 cevabı ikinci kez embed
  ediyordu (18.000 embedding / 16.000 satır) VE `embed_answers` içeride
  `SentenceTransformer('BAAI/bge-m3')`'ü her çağrıda yeniden kurduğu için
  birkaç GB'lık modeli diskten ikinci kez yüklüyordu.
- **Minor:** `catalog.get(role, f"the role of a {role}")` — hemen üstündeki
  fail-closed yorumla çelişen sessiz bir jenerik-açıklama ikamesi — kaldırıldı.
  Örneklenen bir rol kanonik katalogda yoksa (ör. `rollouts.jsonl` farklı bir
  koşudan geliyorsa) artık gateway istemcisi hiç kurulmadan, hangi rol(ler)in
  eksik olduğunu adıyla söyleyen bir tanıyla ve çıkış kodu 2 ile reddediliyor.
- **Test edilebilirlik:** `main()` artık sibling script'lerle (`00`, `01`,
  `03`) aynı desende opsiyonel bir `argv: list[str] | None = None` parametresi
  alıyor — `sys.argv`'ye dokunmadan `main(["--dry-run", ...])` gibi
  çağrılabiliyor. `__main__` bloğu (`raise SystemExit(main())`) değişmedi.

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sample-size", type=int, default=LABEL_SAMPLE_SIZE)
    args = parser.parse_args(argv)

    records = read_rollouts(config.DATA_DIR / "rollouts.jsonl")
    role_rows = [i for i, r in enumerate(records) if r["kind"] == "role"]
    role_records = [records[i] for i in role_rows]

    # Kurulum aşamasındaki her hata `build_default_client()` çevresindeki
    # sarmalayıcıyla aynı desende ele alınır: çıplak bir traceback yerine
    # anlaşılır bir Türkçe tanı ve sıfırdan farklı bir çıkış kodu (2).
    try:
        chosen_local = stratified_sample(role_records, n=args.sample_size, seed=SEED)
    except ValueError as exc:
        print(
            "BAŞARISIZ: örnekleme kurulamadı.\n"
            f"  İstenen örnek boyutu {args.sample_size}, mevcut rol satırı sayısı "
            f"{len(role_records)}.\n"
            "  --sample-size ile mevcut popülasyona sığan daha küçük bir değer verin.\n"
            f"  Ayrıntı: {exc}",
            file=sys.stderr,
        )
        return 2
    chosen = [role_rows[i] for i in chosen_local]

    # load_role_catalog üzerinden: kısmi/pilot bir katalogla etiketleme yapmak,
    # yanlış rol kümesi üzerinde probe eğitmek demek olurdu.
    try:
        catalog = {
            r["role"]: r["description"]
            for r in load_role_catalog(config.DATA_DIR / "roles.json")
        }
    except ValueError as exc:
        print(
            "BAŞARISIZ: rol kataloğu kanonik değil.\n"
            f"  {exc}\n"
            "  Aşama 0'ı (scripts/00_generate_role_data.py) --allow-partial "
            "OLMADAN tamamlayıp tekrar deneyin.",
            file=sys.stderr,
        )
        return 2

    by_role: dict[str, list[int]] = defaultdict(list)
    for row in chosen:
        by_role[records[row]["role"]].append(row)

    # Fail-closed: yukarıdaki `load_role_catalog` kanoniklik doğrulaması rol
    # KÜMESİNİ değil rol İSİMLERİNİN eksiksizliğini garantiler; `rollouts.jsonl`
    # farklı (ör. daha eski/pilot) bir katalogdan üretilmiş olabilir. Örneklenen
    # bir rol katalogda yoksa sessizce jenerik bir açıklama uydurmak (eskiden:
    # `catalog.get(role, f"the role of a {role}")`) yanlış rol kümesi üzerinde
    # hakemlik yapmak demektir — bu, hemen üstteki yorumun reddettiği tam olarak
    # aynı hata sınıfıdır.
    missing_roles = sorted(role for role in by_role if role not in catalog)
    if missing_roles:
        print(
            f"BAŞARISIZ: örneklenen {len(missing_roles)} rol kanonik katalogda yok: "
            f"{missing_roles}.\n"
            "  rollouts.jsonl ile roles.json aynı koşudan gelmiyor olabilir — "
            "Aşama 0 ve Aşama 1'i aynı kanonik katalogla tekrar çalıştırın.",
            file=sys.stderr,
        )
        return 2

    # 00_generate_role_data.py ve 01_smoke_gateway.py ile aynı desen: eksik
    # `APP_KEY_JAILBREAK` çıplak bir traceback yerine anlaşılır bir Türkçe tanı
    # ve sıfırdan farklı bir çıkış koduna (2) çevrilir. Brief'in Adım 5 kod
    # bloğunda bu sarmalayıcı yoktu, ama Adım 6 "anahtar yoksa temiz tanı ve
    # çıkış kodu 2" bekliyor — bu iki ifadeyi tutarlı kılmak için eklendi.
    try:
        client = build_default_client()
    except RuntimeError as exc:
        print(f"BAŞARISIZ: gateway istemcisi kurulamadı.\n  {exc}", file=sys.stderr)
        return 2

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
                description=catalog[role],
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

    # Tüm rol yanıtlarını TEK GEÇİŞTE embed et. Hakem etiketli ~2.000 satır
    # rol yanıtlarının (~16.000) bir alt kümesi olduğu için burada AYRICA
    # `embed_answers([records[i]["answer"] for i in rows])` çağırmak aynı
    # ~2.000 cümleyi ikinci kez embed ederdi (16.000 satır için 18.000
    # embedding) VE `embed_answers` içeride `SentenceTransformer('BAAI/bge-m3')`
    # kurduğu için birkaç GB'lık modeli diskten ikinci kez yüklerdi. Bunun
    # yerine TEK bir dizi hesaplanır; hakem etiketli satırların embedding'leri
    # bu dizinin içinden `role_rows`'taki KONUMLARINA (global `row` değil)
    # göre indekslenir.
    all_role_answers = [records[i]["answer"] for i in role_rows]
    all_embeddings = embed_answers(all_role_answers)

    rows = sorted(labels)
    role_row_position = {row: position for position, row in enumerate(role_rows)}
    label_positions = [role_row_position[row] for row in rows]
    embeddings = all_embeddings[label_positions]

    probe = RoleExpressionProbe(seed=SEED)
    try:
        probe.fit(embeddings, [labels[i] for i in rows])
    except ValueError as exc:
        print(
            "BAŞARISIZ: probe eğitilemedi.\n"
            f"  {exc}\n"
            "  Olası neden: nadir bir kategoriden (ör. 'somewhat') 2'den az örnek "
            "var — train_test_split sınıf başına en az 2 örnek ister. "
            "--sample-size'ı artırıp tekrar deneyin.",
            file=sys.stderr,
        )
        return 2
    print(f"Probe held-out uyumu: {probe.holdout_agreement:.1%} (eşik %85)")

    if not probe.is_trustworthy:
        print(
            "PROBE GÜVENİLİR DEĞİL — spec'in geri çekilme kuralı devreye giriyor.\n"
            "  Rol düzeyinde tut/at filtresine dön ve bunu sonuçlarda raporla.",
            file=sys.stderr,
        )
        return 1

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

`tests/test_label_and_train_probe.py` (Fix Round 1, yeni — script dosya adı
rakamla başladığı için `importlib` ile yüklenir, bkz. `tests/test_judge_gate.py`)
`06_label_and_train_probe.py`'nin karar mantığını ağsız uçtan uca dener:
`build_default_client` VE `embed_answers` her testte monkeypatch'lenir
(bge-m3 hiçbir testte yüklenmez/indirilmez). Kapsam: `collapse()`,
`--dry-run` bütçe aritmetiği, Bulgu 1'in üç kurulum-aşaması tanısı
(oversized `--sample-size`, kanonik olmayan katalog, `probe.fit`
stratify hatası) + Minor'ün katalogda-eksik-rol reddi, eksik anahtar tanısı,
`BudgetExceeded`/`CircuitOpen`/`GatewayError`/`JudgeParseError` durdurma
yolları, `is_trustworthy` kapısının `role_expression.json`'ı yazmadığını,
verbatim hakem etiketi önceliğinin (`labels.get(row, pred)`) tahminle
ezilmediğini VE Bulgu 2'nin tek geçişli embedding indekslemesinin —
`role_rows`'taki KONUMA göre değil global `row` numarasına göre indekslenirse
ortaya çıkacak bir kaymayı doğrudan yakalayan bir casus (`SpyProbe`) ile —
doğru olduğunu. 18 test.

- [ ] **Step 6: Dry-run'ı doğrula**

Run: `cd ~/assistant-axis && uv run python scripts/06_label_and_train_probe.py --dry-run`
Expected: planlanan çağrı sayısı ve aşama bütçesi (300) basılır, çıkış kodu 0.
Anahtar yoksa temiz tanı ve çıkış kodu 2 — bu da beklenen. Fix Round 1
sonrası: 90 satırlık smoke veri setine karşı varsayılan `--sample-size 2000`
ile de artık çıplak bir traceback DEĞİL, "BAŞARISIZ: örnekleme kurulamadı"
tanısı ve çıkış kodu 2 basılır (bu makinede ampirik olarak doğrulandı,
bkz. p2-task-6-report.md, Fix Round 1).

- [ ] **Step 7: Commit**

```bash
git add src/aax/probe.py scripts/06_label_and_train_probe.py tests/test_probe.py tests/test_label_and_train_probe.py
git commit -m "feat: Aşama 2 rol ifadesi probe'u"
```

Fix Round 1 (Bulgu 1/2/Minor) ayrı bir takip commit'idir — değişen dosyalar:
`scripts/06_label_and_train_probe.py`, `tests/test_probe.py`,
`tests/test_label_and_train_probe.py` (yeni). Ayrıntı: p2-task-6-report.md,
Fix Round 1.

---

### Task 7: Aşama 3 — eksen çıkarımı ve A kriteri

Bu planın nihai çıktısı. `axis.py` tamamen saf numpy: model, GPU, ağ yok. Bu sayede ekilmiş bir yönle sentetik veride tam test edilebilir.

**Files:**
- Create: `src/aax/axis.py`
- Create: `scripts/07_extract_axis.py`
- Test: `tests/test_axis.py` (saf modül), `tests/test_extract_axis.py` (script karar mantığı)

**Interfaces:**
- Consumes: `data/activations.npy`, `data/role_expression.json` (Task 6), `data/activations_index.json` (Task 5)
- Produces:
  - `aax.axis.role_vectors(activations, row_roles, row_categories, *, min_responses=10) -> tuple[np.ndarray, list[str], list[str]]` — `(vektörler, isimler, KATEGORİLER)`. Kategori üçüncü dönüş değeri olarak AÇIKÇA döner: gösterim ismi ("rol::kategori", ya da o rolden tek kategori kaldıysa sadece "rol") tek başına ayırt edici değildir — yalnızca `somewhat`'ı kalan bir rol de sadece "rol" adını alır. Eksen yalnızca `fully` vektörlerinden hesaplandığı için çağıran kategoriyi isimden tahmin edemez.
  - `aax.axis.contrast_axis(default_mean, role_mean) -> np.ndarray` — L2 normalize; sonlu olmayan girdi/çıktıda `ValueError`
  - `aax.axis.pca_components(vectors, n_components) -> tuple[np.ndarray, np.ndarray]` — `(components, explained_variance_ratio)`
  - `aax.axis.n_components_for_variance(explained_variance_ratio, threshold=0.70) -> int | None` — kesilmiş spektrum eşiğe ulaşmıyorsa `None` (doyan `searchsorted` yerine)
  - `aax.axis.cosine(a, b) -> float` — sonlu olmayan girdi/çıktıda `ValueError`
  - `aax.axis.projection_percentile(value, distribution) -> float` — sonlu olmayan `value`/dağılımda `ValueError` (Fix Round 2: eskiden NaN sessizce `0.0`'a, yani GEÇTİ'ye doğru yanılıyordu)
  - `aax.axis.evaluate_criterion_a(cos_pc1_axis, default_percentile) -> dict` — sonlu olmayan kosinüs/persentil SERT BAŞARISIZLIK (asla `passed: True`); desil sınırı `TOP_DECILE`/`BOTTOM_DECILE` (Fix Round 2: `BOTTOM_DECILE = 0.1` açık sabit, `1 - TOP_DECILE`'ın ULP hatasını (`0.09999999999999998`) düzeltir — tam `0.1` artık tam `0.9` ile simetrik geçer). **Final Fix Wave (A1): iki koşul artık EŞLEŞTİRİLMİŞ** — `s = sign(cos)`; `s > 0` ise ÜST desil, `s < 0` ise ALT desil ŞART. Bağımsız yazımda `cos=+0.95, persentil=0.0` geçiyordu; bu hipotezin ALEYHİNE delildir. Dönüşe `required_decile` (`"top"`/`"bottom"`/`None`) alanı eklendi.
  - Artifact: `results/axis/` — vektörler, PCA, figür, `criterion_a.json` (künye: `model`, `run_id` — Fix Round 2'den sonra `05_capture_activations.py` gerçek bir değer yazıyor —, `n_layers`, `d_model`; saatten türetilen alan yok). Fix Round 2: hiçbir artefakt, sayısal hesabın TAMAMI başarıyla bitmeden diske yazılmaz; `contrast_axis`/`cosine`/`n_components_for_variance`'dan gelen her `ValueError` (boş `default_idx` dahil) çıkış kodu 2'ye çevrilir, asla yakalanmayan bir çökmeyle çıkış 1'e (DÜŞTÜ ile karışacak şekilde) düşmez.

- [ ] **Step 1: Failing test'leri yaz**

`tests/test_axis.py` (26 test — saf modül; Fix Round 2, 4 yeni: `BOTTOM_DECILE` sınırında `0.1`/`0.9` simetrisi, `projection_percentile`'ın sonlu olmayan `value`/dağılımı reddetmesi):

```python
import numpy as np
import pytest

from aax.axis import (
    contrast_axis,
    cosine,
    evaluate_criterion_a,
    n_components_for_variance,
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


def test_cosine_rejects_non_finite_input():
    """NaN yayılmamalı: `na == 0` kontrolü NaN için ateşlemez, o yüzden ayrı
    bir sonluluk kontrolü şart — yoksa boş dilim ortalaması sessizce geçer."""
    nan_vector = np.array([np.nan, 0.0, 0.0])
    with pytest.raises(ValueError, match="sonlu olmayan"):
        cosine(nan_vector, np.ones(3))
    with pytest.raises(ValueError, match="sonlu olmayan"):
        cosine(np.ones(3), nan_vector)
    with pytest.raises(ValueError, match="sonlu olmayan"):
        cosine(np.array([np.inf, 0.0, 0.0]), np.ones(3))


def test_contrast_axis_is_unit_norm():
    axis = contrast_axis(np.array([5.0, 0.0]), np.array([1.0, 0.0]))
    assert np.linalg.norm(axis) == pytest.approx(1.0)


def test_contrast_axis_points_from_roles_toward_default():
    axis = contrast_axis(np.array([10.0, 0.0]), np.array([2.0, 0.0]))
    assert axis[0] > 0


def test_contrast_axis_rejects_non_finite_input():
    """Boş bir dilimin ortalaması NaN'dır; `norm == 0` bunu yakalamaz."""
    nan_mean = np.full(3, np.nan)
    with pytest.raises(ValueError, match="sonlu olmayan"):
        contrast_axis(np.ones(3), nan_mean)
    with pytest.raises(ValueError, match="sonlu olmayan"):
        contrast_axis(nan_mean, np.ones(3))


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


def test_pca_centring_survives_a_large_shared_offset():
    """Ortalama çıkarma (centring) regresyon koruması.

    `test_pca_recovers_a_planted_direction`'daki sabit tohumlu örneklemin
    örnek ortalaması zaten ~0 olduğu için centring silinse bile geçiyor.
    Gerçek aktivasyonlarda ise tüm rollerin paylaştığı BÜYÜK bir ortak
    ortalama var — centring tam da o zaman kritik. Burada ekilmiş yöne dik,
    normu varyanstan çok daha büyük bir ofset ekleniyor: centring olmadan
    SVD'nin ilk sağ tekil vektörü ekilmiş yönü değil ofsetin yönünü bulur ve
    aşağıdaki kosinüs çöker.
    """
    rng = np.random.default_rng(1)
    planted = np.array([0.0, 1.0, 0.0, 0.0])
    offset = np.array([50.0, 0.0, 0.0, 0.0])
    scores = rng.normal(scale=1.0, size=200)
    noise = rng.normal(scale=0.05, size=(200, 4))
    vectors = offset[None, :] + scores[:, None] * planted[None, :] + noise

    components, ratios = pca_components(vectors, n_components=2)
    assert abs(cosine(components[0], planted)) > 0.99
    assert ratios[0] > 0.9


def test_role_vectors_skip_categories_below_minimum():
    acts = np.ones((12, 2, 3), dtype=np.float32)
    roles = ["a"] * 12
    cats = ["fully"] * 9 + ["no"] * 3
    vectors, names, categories = role_vectors(acts, roles, cats, min_responses=10)
    assert names == []
    assert categories == []


def test_role_vectors_averages_qualifying_rows():
    acts = np.zeros((10, 1, 2), dtype=np.float32)
    acts[:, 0, 0] = 4.0
    roles = ["pirate"] * 10
    cats = ["fully"] * 10
    vectors, names, categories = role_vectors(acts, roles, cats, min_responses=10)
    assert names == ["pirate"]
    assert categories == ["fully"]
    assert vectors.shape == (1, 1, 2)
    assert vectors[0, 0, 0] == pytest.approx(4.0)


def test_role_vectors_keeps_fully_and_somewhat_separate():
    acts = np.zeros((20, 1, 2), dtype=np.float32)
    acts[:10, 0, 0] = 1.0
    acts[10:, 0, 0] = 9.0
    roles = ["bard"] * 20
    cats = ["fully"] * 10 + ["somewhat"] * 10
    vectors, names, categories = role_vectors(acts, roles, cats, min_responses=10)
    assert sorted(names) == ["bard::fully", "bard::somewhat"]
    assert sorted(categories) == ["fully", "somewhat"]


def test_role_vectors_reports_category_even_when_name_is_ambiguous():
    """Gösterim ismi tek başına kategoriyi ayırt etmez.

    Yalnızca `somewhat`'ı eşiği geçen bir rol "rol" adını alır — yalnızca
    `fully`'si geçen bir rolle aynı. Eksen sadece `fully` vektörlerinden
    hesaplandığı için çağıranın kategoriyi isimden tahmin etmesi imkânsız;
    bu yüzden ayrı olarak dönmeli.
    """
    acts = np.zeros((30, 1, 2), dtype=np.float32)
    acts[:10, 0, 0] = 1.0  # sadece_somewhat, 10 satır somewhat
    acts[10:20, 0, 0] = 2.0  # sadece_fully, 10 satır fully
    acts[20:, 0, 0] = 3.0  # sadece_fully, 10 satır no (elenir)
    roles = ["sadece_somewhat"] * 10 + ["sadece_fully"] * 10 + ["sadece_fully"] * 10
    cats = ["somewhat"] * 10 + ["fully"] * 10 + ["no"] * 10

    vectors, names, categories = role_vectors(acts, roles, cats, min_responses=10)

    assert names == ["sadece_fully", "sadece_somewhat"]
    assert categories == ["fully", "somewhat"]
    assert len(categories) == len(names) == vectors.shape[0]
    # kategori üzerinden seçim: eksen yalnızca bu vektörden hesaplanmalı
    fully_only = vectors[[i for i, c in enumerate(categories) if c == "fully"]]
    assert fully_only.shape[0] == 1
    assert fully_only[0, 0, 0] == pytest.approx(2.0)


def test_n_components_for_variance_counts_against_the_full_spectrum():
    """Kesilmiş spektrumdan kesin sayı okunamaz.

    20 eşit bileşende %70'e ulaşmak 14 bileşen ister. İlk 10 oranla
    `np.searchsorted(cumsum, 0.70)` doyuma ulaşıp 11 derdi — gerçekten
    desteklenmeyen, yanıltıcı biçimde DÜŞÜK bir sayı.
    """
    full = np.full(20, 1 / 20)
    assert n_components_for_variance(full, 0.70) == 14
    assert n_components_for_variance(full[:10], 0.70) is None
    assert n_components_for_variance(np.array([0.8, 0.2]), 0.70) == 1
    assert n_components_for_variance(np.array([]), 0.70) is None


def test_projection_percentile_at_extremes():
    dist = np.arange(100.0)
    assert projection_percentile(-5.0, dist) == pytest.approx(0.0)
    assert projection_percentile(200.0, dist) == pytest.approx(1.0)


def test_projection_percentile_rejects_non_finite_value():
    """NaN sessizce 0.0'a çözülürse `evaluate_criterion_a` bunu alt desilin
    İÇİNDE sayıp yanlış yönde (GEÇTİ'ye doğru) bir sonuç üretir."""
    dist = np.arange(10.0)
    with pytest.raises(ValueError, match="sonlu olmayan"):
        projection_percentile(float("nan"), dist)
    with pytest.raises(ValueError, match="sonlu olmayan"):
        projection_percentile(float("inf"), dist)


def test_projection_percentile_rejects_non_finite_distribution():
    dist = np.array([1.0, np.nan, 3.0])
    with pytest.raises(ValueError, match="sonlu olmayan"):
        projection_percentile(1.0, dist)


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


def test_criterion_a_never_passes_on_a_nan_cosine():
    """Sahte GEÇTİ yolu.

    NaN ile yapılan her karşılaştırma False döner: `abs(nan) <= 0.6` False
    olduğu için eski kod hiçbir gerekçe eklemez ve `passed` True çıkardı —
    yani tanımsız veriden "A KRİTERİ: GEÇTİ".
    """
    result = evaluate_criterion_a(cos_pc1_axis=float("nan"), default_percentile=0.95)
    assert result["passed"] is False
    assert "sonlu değil" in result["reason"]


def test_criterion_a_never_passes_on_a_nan_percentile():
    result = evaluate_criterion_a(cos_pc1_axis=0.9, default_percentile=float("nan"))
    assert result["passed"] is False
    assert "sonlu değil" in result["reason"]


def test_criterion_a_rejects_infinite_values():
    for cos_value, percentile in ((float("inf"), 0.95), (0.9, float("-inf"))):
        result = evaluate_criterion_a(cos_pc1_axis=cos_value, default_percentile=percentile)
        assert result["passed"] is False
        assert "sonlu değil" in result["reason"]


def test_criterion_a_boundary_bottom_decile_exactly_0_1_passes():
    """`1 - TOP_DECILE` ikili kayan noktada `0.09999999999999998`'tir —
    tam `0.1` persentili (n 10'un katıysa `k/n` ile ATTAINABLE, beklenen
    ölçekte rutin) bu ifadeyle KAÇARDI. `BOTTOM_DECILE = 0.1` sabiti bunu
    düzeltir; sınır ULP'siz, ayna simetrik olmalı."""
    result = evaluate_criterion_a(cos_pc1_axis=0.9, default_percentile=0.1)
    assert result["passed"] is True


def test_criterion_a_boundary_top_decile_exactly_0_9_passes():
    """Aynalı üst sınır — regresyon: bu her zaman geçiyordu, alt sınırla
    aynı davranması gerektiğini doğrulamak için burada."""
    result = evaluate_criterion_a(cos_pc1_axis=0.9, default_percentile=0.9)
    assert result["passed"] is True
```

`tests/test_extract_axis.py` (10 test — script karar mantığı; model/GPU/ağ yok, tüm veri sentetik; Fix Round 2, 3 yeni: boş `default_idx` çıkış 2'ye düşer (1'e değil), sayısal bölümdeki herhangi bir `ValueError` çıkış 2'ye sarmalanır, geç bir hata hiçbir kısmi artefakt bırakmaz):

```python
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
        json.dumps({"expression": expression}, ensure_ascii=False), encoding="utf-8"
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

    assert ea.main() in (0, 1)  # karar ne olursa olsun artefakt yazılmalı

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

    assert ea.main() == 2

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

    assert ea.main() == 2

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

    assert ea.main() == 2

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

    assert ea.main() == 2

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

    assert ea.main() == 2

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

    assert ea.main() == 2

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
                "middle_layer": 1,
                "rows": rows,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "role_expression.json").write_text(
        json.dumps({"expression": {str(i): "fully" for i in range(n_roles * per_role)}}),
        encoding="utf-8",
    )

    assert ea.main() in (0, 1)

    report = json.loads((tmp_path / "axis" / "criterion_a.json").read_text(encoding="utf-8"))
    ratios_first10 = np.asarray(report["explained_variance_ratio"])
    assert len(ratios_first10) == 10  # rapor yine ilk 10 bileşeni gösterir
    # eski (doyan) hesap tam olarak 11 derdi:
    assert int(np.searchsorted(np.cumsum(ratios_first10), 0.70) + 1) == 11
    assert np.cumsum(ratios_first10)[-1] < 0.70  # ilk 10 eşiğe hiç ulaşmıyor
    assert report["n_components_for_70pct"] > 11


# --- Minor: künye -------------------------------------------------------------


def test_criterion_a_records_provenance_without_a_timestamp(tmp_path, monkeypatch):
    _patch_paths(monkeypatch, tmp_path)
    _write_dataset(
        tmp_path,
        role_spec=[("a", "fully", 12, 1.0), ("b", "fully", 12, 4.0), ("c", "fully", 12, 9.0)],
        index_extra={"run_id": "abc123"},
    )

    assert ea.main() in (0, 1)

    report = json.loads((tmp_path / "axis" / "criterion_a.json").read_text(encoding="utf-8"))
    assert report["model"] == "test/Model-1.7B"
    assert report["run_id"] == "abc123"
    assert report["n_layers"] == 2
    assert report["d_model"] == 3
    assert report["middle_layer"] == 1
    # saatten türetilen hiçbir alan yok
    assert not [k for k in report if "time" in k or "date" in k or "stamp" in k]
```

- [ ] **Step 2: Test'lerin başarısız olduğunu doğrula**

Run: `cd ~/assistant-axis && uv run --extra dev pytest tests/test_axis.py tests/test_extract_axis.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aax.axis'`

- [ ] **Step 3: `src/aax/axis.py` yaz**

```python
"""Persona uzayı analizi — saf numpy.

Bu modül model, GPU veya ağ bilmez. Girdi vektör matrisleri, çıktı eksen ve
PCA sonuçları. Bu sayede ekilmiş bir yönle sentetik veride tam test edilebilir
(spec Bölüm 4.3, Bölüm 10).
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

COS_THRESHOLD = 0.6
TOP_DECILE = 0.9
# `1 - TOP_DECILE` GÖRÜNÜŞTE aynı şeydir ama değildir: ikili kayan noktada
# `1 - 0.9 == 0.09999999999999998`, yani tam `0.1` persentili (`percentile =
# k/n` roller üzerinde, n 10'un katıysa ulaşılabilir — beklenen ölçekte
# rutin) alt desil testini KAÇIRIR, oysa aynalı `0.9` üst desil testini
# geçer. A kriteri ön kaydedilmiş: sınır tam olmalı, bir ULP'lik asimetri
# olmamalı. `BOTTOM_DECILE` bu yüzden `1 - TOP_DECILE` olarak DEĞİL, açıkça
# `0.1` olarak tanımlanır.
BOTTOM_DECILE = 0.1


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """İki vektörün kosinüs benzerliği.

    NaN/inf sessizce yayılmaz: sonlu olmayan girdi ya da sonlu olmayan sonuç
    `ValueError` ile reddedilir. Aksi hâlde boş bir dilimin ortalamasından
    doğan bir NaN, `evaluate_criterion_a`'ya kadar gidip sahte bir "GEÇTİ"
    üretir (NaN karşılaştırmaları hep False'tur).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError(
            "kosinüs sonlu olmayan (NaN/inf) değer içeren vektörle tanımsız — "
            "girdi büyük olasılıkla boş bir dilimin ortalamasından geliyor"
        )
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        raise ValueError("sıfır vektörün kosinüsü tanımsız")
    value = float(np.dot(a, b) / (na * nb))
    if not math.isfinite(value):
        raise ValueError("kosinüs sonlu olmayan bir değere çözüldü")
    return value


def contrast_axis(default_mean: np.ndarray, role_mean: np.ndarray) -> np.ndarray:
    """Assistant Axis = mean(default) − mean(rol vektörleri), L2 normalize.

    Makale bunu PC1'e tercih ediyor: PC1'in her modelde aynı anlamı taşıyacağı
    garanti değil (Ek G.5).

    `cosine` ile aynı gerekçe: sonlu olmayan girdi/çıktı sessizce geçmez.
    `norm == 0` kontrolü NaN için ateşlemez, bu yüzden ayrı bir sonluluk
    kontrolü şart.
    """
    default_mean = np.asarray(default_mean, dtype=np.float64)
    role_mean = np.asarray(role_mean, dtype=np.float64)
    if not np.isfinite(default_mean).all() or not np.isfinite(role_mean).all():
        raise ValueError(
            "kontrast vektörü sonlu olmayan (NaN/inf) girdiyle tanımsız — "
            "default veya rol ortalaması büyük olasılıkla boş bir dilimden geliyor"
        )
    axis = default_mean - role_mean
    norm = np.linalg.norm(axis)
    if norm == 0:
        raise ValueError("kontrast vektörü sıfır — default ve rol ortalamaları aynı")
    axis = axis / norm
    if not np.isfinite(axis).all():
        raise ValueError("kontrast vektörü sonlu olmayan bir değere çözüldü")
    return axis


def role_vectors(
    activations: np.ndarray,
    row_roles: list[str],
    row_categories: list[str],
    *,
    min_responses: int = 10,
) -> tuple[np.ndarray, list[str], list[str]]:
    """(rol, kategori) başına ortalama aktivasyon.

    Makalenin kuralı: bir kategori en az `min_responses` yanıt içermiyorsa
    o vektör hesaplanmaz. fully ve somewhat ayrı vektörler üretir.

    Dönüş: ([n_vectors, n_layers, d_model], isimler, kategoriler).

    `isimler` gösterim içindir: "rol::kategori", ya da o rolden tek kategori
    kaldıysa sadece "rol". Bu kural isimleri tek başına ayırt edici KILMAZ —
    yalnızca `somewhat`'ı kalan bir rol de sadece "rol" adını alır ve yalnızca
    `fully`'si kalan bir rolden ayırt edilemez. `kategoriler` bu nedenle ayrı
    döner: Assistant Axis yalnızca `fully` rol vektörlerinden hesaplandığı için
    çağıranın kategoriyi isimden tahmin etmesi değil, doğrudan bilmesi gerekir.
    `kategoriler[i]` her zaman `isimler[i]` ile aynı vektöre aittir.
    """
    buckets: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, (role, category) in enumerate(zip(row_roles, row_categories)):
        if category in ("fully", "somewhat"):
            buckets[(role, category)].append(index)

    kept = {k: rows for k, rows in buckets.items() if len(rows) >= min_responses}
    if not kept:
        return np.empty((0,) + activations.shape[1:], dtype=np.float32), [], []

    roles_with_both = defaultdict(set)
    for role, category in kept:
        roles_with_both[role].add(category)

    names: list[str] = []
    categories: list[str] = []
    vectors: list[np.ndarray] = []
    for (role, category), rows in sorted(kept.items()):
        name = f"{role}::{category}" if len(roles_with_both[role]) > 1 else role
        names.append(name)
        categories.append(category)
        vectors.append(activations[rows].astype(np.float64).mean(axis=0))

    return np.stack(vectors).astype(np.float32), names, categories


def pca_components(
    vectors: np.ndarray, n_components: int
) -> tuple[np.ndarray, np.ndarray]:
    """Roller arası ortalamayı çıkarıp PCA koş.

    Ortalamayı çıkarmak (centring) isteğe bağlı bir süsleme değil: gerçek
    aktivasyonlarda tüm rollerin paylaştığı büyük bir ortalama vektör vardır ve
    centring olmadan SVD'nin ilk sağ tekil vektörü varyans yönünü değil bu ortak
    ortalamanın yönünü bulur. Regresyon koruması:
    `test_pca_centring_survives_a_large_shared_offset`.

    Dönüş: (bileşenler [k, d], açıklanan varyans oranı [k]).
    """
    centered = np.asarray(vectors, dtype=np.float64)
    centered = centered - centered.mean(axis=0, keepdims=True)
    _u, s, vt = np.linalg.svd(centered, full_matrices=False)
    variance = s**2
    ratios = variance / variance.sum()
    k = min(n_components, vt.shape[0])
    return vt[:k], ratios[:k]


def n_components_for_variance(
    explained_variance_ratio: np.ndarray, threshold: float = 0.70
) -> int | None:
    """Kümülatif varyansın `threshold`'u ilk kez aştığı bileşen sayısı.

    Verilen spektrum eşiğe hiç ulaşmıyorsa `None` döner. Bu ayrım önemli:
    yalnızca ilk 10 bileşen istenmişse `np.searchsorted(cumsum, 0.70)` doyuma
    ulaşıp her zaman 11 der ve gerçek cevap 10'dan büyük olduğunda "persona
    uzayı düşük boyutlu" iddiasını destekleyecek şekilde YANILTICI biçimde
    küçük bir sayı raporlar. `None` dönen çağıran taraf ya tam spektrumla
    yeniden hesaplamalı ya da ">k" gibi bir alt sınır yazmalıdır.
    """
    ratios = np.asarray(explained_variance_ratio, dtype=np.float64)
    if ratios.size == 0:
        return None
    if not np.isfinite(ratios).all():
        raise ValueError("açıklanan varyans oranı sonlu olmayan değer içeriyor")
    reached = np.nonzero(np.cumsum(ratios) >= threshold)[0]
    if reached.size == 0:
        return None
    return int(reached[0]) + 1


def projection_percentile(value: float, distribution: np.ndarray) -> float:
    """`value`'nun dağılım içindeki konumu, 0-1 arası.

    `cosine`/`contrast_axis` ile aynı gerekçe, ama burada YÖN daha kritik:
    sonlu olmayan (NaN/inf) bir `value` kontrolsüz bırakılırsa
    `(dist <= nan).sum()` her zaman `0` verir, yani persentil sessizce
    `0.0`'a çözülür — ve `evaluate_criterion_a(0.9, 0.0)` bunu ALT desilin
    İÇİNDE sayıp `passed: True` üretir. Modülün geri kalanındaki her NaN
    koruması BAŞARISIZLIĞA doğru yanılır; bu satır korumasız kalırsa tam
    tersi yönde, GEÇTİ'ye doğru yanılırdı — ön kaydedilmiş bir kriter için
    yanlış yön. Script'te bugün erişilemez (her iki girdi de yukarı akışta
    sonlu olduğu doğrulanır), ama modüldeki son korumasız NaN geçişi
    buydu.
    """
    value = float(value)
    dist = np.asarray(distribution, dtype=np.float64)
    if not math.isfinite(value):
        raise ValueError(
            "persentil sonlu olmayan (NaN/inf) bir `value` için tanımsız — "
            "girdi büyük olasılıkla boş bir dilimin ortalamasından geliyor"
        )
    if not np.isfinite(dist).all():
        raise ValueError(
            "persentil sonlu olmayan (NaN/inf) değer içeren bir dağılımla tanımsız"
        )
    return float((dist <= value).sum() / len(dist))


def evaluate_criterion_a(cos_pc1_axis: float, default_percentile: float) -> dict:
    """Spec Bölüm 7, A kriteri.

    Geçer: orta katmanda |cos(PC1, kontrast vektörü)| > 0.6 VE default
    Assistant projeksiyonu PC1'in en üst (veya en alt) desilinde.

    PC1'in işareti SVD'nin keyfî bir seçimidir; hem kosinüs hem desil
    büyüklük üzerinden değerlendirilir.

    Sonlu olmayan girdi (NaN/inf) SERT BAŞARISIZLIKTIR. NaN ile yapılan her
    karşılaştırma False döndüğü için sessiz bir NaN, hiçbir gerekçe
    eklenmeden `passed: True` üretirdi — yani tanımsız veriden "GEÇTİ".
    """
    cos_value = float(cos_pc1_axis)
    percentile_value = float(default_percentile)
    cos_is_finite = math.isfinite(cos_value)
    percentile_is_finite = math.isfinite(percentile_value)

    magnitude = abs(cos_value)
    in_extreme_decile = percentile_is_finite and (
        percentile_value >= TOP_DECILE or percentile_value <= BOTTOM_DECILE
    )

    reasons = []
    if not cos_is_finite:
        reasons.append(
            f"cos(PC1, eksen) sonlu değil ({cos_value}) — tanımsız değerden karar çıkarılamaz"
        )
    elif magnitude <= COS_THRESHOLD:
        reasons.append(f"|cos| {magnitude:.3f} <= {COS_THRESHOLD}")
    if not percentile_is_finite:
        reasons.append(
            f"default persentili sonlu değil ({percentile_value}) — "
            "tanımsız değerden karar çıkarılamaz"
        )
    elif not in_extreme_decile:
        reasons.append(
            f"default projeksiyonu uç desilde değil (persentil {percentile_value:.3f})"
        )

    return {
        "cos_pc1_axis": cos_value,
        "cos_magnitude": magnitude,
        "default_percentile": percentile_value,
        "passed": not reasons,
        "reason": "; ".join(reasons) if reasons else "her iki koşul da sağlandı",
    }
```

- [ ] **Step 4: Testlerin geçtiğini doğrula**

Run: `cd ~/assistant-axis && uv run --extra dev --extra ml pytest tests/test_axis.py tests/test_extract_axis.py -v`
Expected: PASS, 36 passed (26 + 10). (Bu satır önceden "15 passed" diyordu; blok o hâlinde 14 test içeriyordu — Fix Round 1'de hem sayı düzeltildi hem de 15 test eklendi, sonuç 29 (22 + 7) oldu. Fix Round 2, 4 + 3 = 7 test daha ekledi: 29 → 36.)

- [ ] **Step 5: `scripts/07_extract_axis.py` yaz**

```python
#!/usr/bin/env python3
"""Aşama 3 — rol vektörleri, PCA, Assistant Axis, A kriteri.

Kullanım:
    uv run --extra ml python scripts/07_extract_axis.py
"""
from __future__ import annotations

import json
import sys

import numpy as np

from aax import config
from aax.axis import (
    contrast_axis,
    cosine,
    evaluate_criterion_a,
    n_components_for_variance,
    pca_components,
    projection_percentile,
    role_vectors,
)

OUT_DIR = config.RESULTS_DIR / "axis"

ROLE_EXPRESSION_PATH = config.DATA_DIR / "role_expression.json"


def main() -> int:
    # `mmap_mode="r"`: planlanan ölçekte (16.000 × 28 × 2048 float32) bu
    # dosya ~3.5 GB'dir. Tam yükleme onu belleğe kopyalar; aşağıdaki
    # `acts[role_idx]`/`acts[default_idx]` fantezi indekslemesi zaten SEÇİLEN
    # satırların bir kopyasını (~3.7 GB'a kadar) çıkarır. mmap ile yalnızca
    # seçilen satırlar maddîleşir — dosyanın tamamı iki kez belleğe alınmaz.
    acts = np.load(config.DATA_DIR / "activations.npy", mmap_mode="r")
    index = json.loads((config.DATA_DIR / "activations_index.json").read_text(encoding="utf-8"))

    # 06_label_and_train_probe.py ile aynı desen: brief'in Adım 5 kod bloğunda
    # bu sarmalayıcı yoktu (çıplak `.read_text()["expression"]`), ama görev
    # tanımı "eksikse temiz başarısız olsun, traceback değil" diyor — dosya
    # `role_expression.json`'ı henüz üretmemiş bir operatör için çıplak
    # FileNotFoundError traceback'i bu koşulu karşılamıyor.
    try:
        expression = json.loads(ROLE_EXPRESSION_PATH.read_text(encoding="utf-8"))["expression"]
    except FileNotFoundError:
        print(
            f"BAŞARISIZ: {ROLE_EXPRESSION_PATH} yok.\n"
            "  Bu dosya Aşama 2'nin çıktısıdır — önce "
            "scripts/06_label_and_train_probe.py çalıştırılmalı.",
            file=sys.stderr,
        )
        return 2
    except (json.JSONDecodeError, KeyError) as exc:
        print(
            f"BAŞARISIZ: {ROLE_EXPRESSION_PATH} bozuk veya 'expression' anahtarı yok.\n"
            f"  Ayrıntı: {exc}\n"
            "  scripts/06_label_and_train_probe.py'yi tekrar çalıştırıp dosyayı yeniden üretin.",
            file=sys.stderr,
        )
        return 2

    rows = index["rows"]
    middle = index["middle_layer"]
    print(f"{acts.shape[0]} satır, {acts.shape[1]} katman, orta katman {middle}")

    role_idx = [i for i, r in enumerate(rows) if r["kind"] == "role"]
    default_idx = [i for i, r in enumerate(rows) if r["kind"] == "default"]

    # `len(names) == 0` koruması (altta) "fully" tarafındaki boş-dilim NaN'ını
    # yakalıyordu; bu onun default tarafındaki İKİZİ. `acts[default_idx]` boş
    # bir dizi olursa `.mean(axis=0)` yalnızca bir RuntimeWarning ile NaN
    # döner (bkz. `default_mean_all` altta) — o NaN korumasız bırakılırsa
    # `contrast_axis`'e kadar sessizce yayılır, orada `ValueError` fırlatır,
    # bu da yakalanmazsa yorumlayıcı çıkış kodu 1 ile döner. Çıkış 1, "A
    # KRİTERİ DÜŞTÜ" anlamına gelen koddur — bir çökme asla bilimsel bir
    # sonuç olarak kaydedilemez, bu yüzden burada erkenden, ucuzca kontrol
    # edilir.
    if len(default_idx) == 0:
        print(
            "BAŞARISIZ: activations_index.json içinde 'default' türünde hiç satır "
            "yok — default ortalaması tanımsız.\n"
            "  Assistant Axis mean(default) − mean(fully rol vektörleri) olarak "
            "tanımlı; default satır yoksa hesaplanacak bir şey yok.\n"
            "  Kontrol edin: scripts/04_generate_rollouts.py'nin default rollout'ları "
            "ürettiğini ve scripts/05_capture_activations.py'nin bunları "
            "activations_index.json'a yazdığını.\n"
            "  Bu bir BAŞARISIZLIKTIR, A kriteri kararı DEĞİLDİR: tanımsız veriden "
            "GEÇTİ/DÜŞTÜ çıkarılamaz.",
            file=sys.stderr,
        )
        return 2

    # Bayatlık kontrolü: `expression.get(str(i), "no")` eşleşmeyen her satırı
    # sessizce "no" sayar. Tek bir taze koşuda sorun değil, ama farklı bir
    # --limit veya rol kümesiyle üretilmiş eski bir role_expression.json Aşama 1
    # yeniden koşturulduktan sonra yerinde kalırsa fully/somewhat ayrımı kısmen
    # kayar ve hiçbir hata vermez. İki ucuz kontrol bunu yakalar: anahtar sayısı
    # ve anahtarların indeksteki rol satırlarını kapsaması.
    if len(expression) != len(role_idx):
        print(
            "BAŞARISIZ: role_expression.json ile activations_index.json uyuşmuyor —\n"
            f"  ifade haritasında {len(expression)} anahtar var, "
            f"indekste {len(role_idx)} rol satırı.\n"
            "  Olası neden: farklı bir --limit veya rol kümesiyle üretilmiş eski bir "
            "role_expression.json, Aşama 1 yeniden koşturulduktan sonra yerinde kalmış.\n"
            "  scripts/06_label_and_train_probe.py'yi güncel rollouts/aktivasyonlarla "
            "yeniden çalıştırın.",
            file=sys.stderr,
        )
        return 2
    missing = [i for i in role_idx if str(i) not in expression]
    if missing:
        print(
            "BAŞARISIZ: role_expression.json indeksteki bazı rol satırlarını kapsamıyor —\n"
            f"  {len(missing)} satırın karşılığı yok (ilk örnekler: {missing[:5]}).\n"
            "  Anahtar sayısı tutsa bile satır numaraları kaymış: iki dosya farklı "
            "koşulardan geliyor.\n"
            "  scripts/06_label_and_train_probe.py'yi güncel rollouts/aktivasyonlarla "
            "yeniden çalıştırın.",
            file=sys.stderr,
        )
        return 2

    categories = [expression[str(i)] for i in role_idx]
    vectors, names, vector_categories = role_vectors(
        acts[role_idx], [rows[i]["role"] for i in role_idx], categories
    )
    print(f"{len(names)} rol vektörü hesaplandı (>=10 yanıt kuralı sonrası)")

    # role_vectors, hiçbir (rol, kategori) çifti min_responses eşiğini
    # geçemezse boş dizi döner. Bu durumda devam etmek NaN'lara (boş dilimin
    # ortalaması) ve ardından PCA/kosinüs adımlarında şifreli bir
    # IndexError'a yol açar — görev tanımının "traceback değil, temiz mesaj"
    # koşulu burada da geçerli.
    if len(names) == 0:
        print(
            "BAŞARISIZ: hiçbir rol min_responses (>=10) eşiğini geçemedi — "
            "hesaplanacak rol vektörü yok.\n"
            "  Olası neden: role_expression.json'daki dağılım beklenenden "
            "farklı veya veri seti hâlâ pilot ölçekte (--limit).",
            file=sys.stderr,
        )
        return 2
    if len(names) < 40:
        print(f"UYARI: yalnızca {len(names)} vektör — PCA kararsız olabilir.")

    default_mean_all = acts[default_idx].astype(np.float64).mean(axis=0)  # [L, D]

    # Assistant Axis = mean(default) − mean(fully ROL VEKTÖRLERİ). Ham "fully"
    # satırlarını havuzlamak İKİ ayrı hata olurdu: (1) "fully" sayısı 10'un
    # altında kalan bir rol role_vectors tarafından elenmişken ham satırlarıyla
    # yine de ortalamaya karışırdı; (2) çok rollout'lu roller ortalamayı ele
    # geçirirdi — oysa tanım her nitelikli rolün EŞİT ağırlıkta katkısını ister.
    fully_positions = [i for i, c in enumerate(vector_categories) if c == "fully"]
    if not fully_positions:
        print(
            "BAŞARISIZ: hiçbir rol vektörü 'fully' kategorisinde değil — "
            "Assistant Axis tanımsız.\n"
            "  Eksen mean(default) − mean(fully rol vektörleri) olarak tanımlı; "
            "fully rol vektörü yoksa hesaplanacak bir şey yok.\n"
            "  Kontrol edin: role_expression.json'daki dağılım (kaç satır 'fully'?), "
            ">=10 yanıt kuralı (bir rolün 'fully' sayısı eşiğin altında kalmış "
            "olabilir) ve veri setinin hâlâ pilot ölçekte (--limit) olup olmadığı.\n"
            "  Bu bir BAŞARISIZLIKTIR, A kriteri kararı DEĞİLDİR: tanımsız veriden "
            "GEÇTİ/DÜŞTÜ çıkarılamaz.",
            file=sys.stderr,
        )
        return 2
    role_mean_all = vectors[fully_positions].astype(np.float64).mean(axis=0)  # [L, D]
    print(f"  bunların {len(fully_positions)} tanesi 'fully' — eksen bunlardan hesaplanıyor")

    # Sayısal adımların TAMAMI bu blokta: `contrast_axis`/`cosine`/
    # `n_components_for_variance` sonlu olmayan (NaN/inf) bir değere ya da
    # sıfır normlu bir kontrast vektörüne çarparsa `ValueError` fırlatır
    # (bkz. `aax.axis`). Yukarıdaki iki koruma (`default_idx` boş, hiç
    # `fully` yok) en olası iki NaN kaynağını erkenden kapatıyor, ama aynı
    # çarpışma başka yollardan da gerçekleşebilir — ör. default ve fully
    # ortalamaları TESADÜFEN eşitse (sıfır normlu kontrast) ya da
    # `activations.npy` içinde başka bir nedenle bozuk (NaN/inf) bir satır
    # varsa. Yakalanmayan bir `ValueError` burada yorumlayıcıyı çıkış kodu
    # 1 ile döndürür — "A KRİTERİ DÜŞTÜ" anlamına gelen kod. Bir çökme asla
    # bilimsel bir sonuç olarak kaydedilemez; bu yüzden buradan itibaren her
    # şey yakalanır ve çıkış 2'ye (BAŞARISIZ, karar DEĞİL) çevrilir. Hiçbir
    # artefakt bu blok TAMAMLANMADAN (yani her değer başarıyla
    # hesaplanmadan) diske YAZILMAZ — aksi hâlde geç bir raise, önceki bir
    # koşudan kalma `criterion_a.json` yanında yarım `assistant_axis.npy` /
    # `role_vectors.npy` bırakabilirdi ve `results/` commit'lendiği için bu
    # tutarsız kombinasyon depoda kalıcı hâle gelirdi.
    try:
        axis_per_layer = np.stack(
            [contrast_axis(default_mean_all[l], role_mean_all[l]) for l in range(acts.shape[1])]
        )

        cos_by_layer = []
        for layer in range(acts.shape[1]):
            components, _ = pca_components(vectors[:, layer, :], n_components=1)
            cos_by_layer.append(cosine(components[0], axis_per_layer[layer]))

        # Tam spektrum isteniyor: n_components_for_70pct yalnızca ilk 10
        # orandan hesaplansaydı gerçek cevap 10'u aştığında doyuma ulaşıp
        # hep 11 derdi ve "persona uzayı düşük boyutlu" iddiasını yapay
        # olarak destekler. Raporlanan `explained_variance_ratio` yine ilk
        # 10 bileşendir.
        components_mid, ratios_full = pca_components(
            vectors[:, middle, :], n_components=vectors.shape[0]
        )
        ratios_mid = ratios_full[:10]
        pc1 = components_mid[0]
        role_projections = vectors[:, middle, :] @ pc1
        default_projection = float(default_mean_all[middle] @ pc1)
        percentile = projection_percentile(default_projection, role_projections)

        verdict = evaluate_criterion_a(cos_by_layer[middle], percentile)

        # `ratios_full` burada TAM spektrumdur (`n_components=vectors.shape[0]`
        # yukarıda), yani toplamı her zaman 1.0'dır (PCA'nın tanımı gereği:
        # `ratios = variance / variance.sum()`) — kümülatif toplam %70 eşiğini
        # er ya da geç MUTLAKA aşar. `n_components_for_variance` yalnızca
        # KESİLMİŞ bir spektrum verildiğinde `None` dönebilir (bkz. modül
        # docstring'i); burada asla olmamalı. Olursa yukarıdaki varsayım bir
        # yerde bozulmuş demektir — sessizce `None`/">k" yazıp devam etmek
        # yerine (eski davranış) açıkça BAŞARISIZ olunur, `n_components_for_70pct`
        # alanı böylece tek bir tipte (`int`) kalır.
        n_for_70 = n_components_for_variance(ratios_full, 0.70)
        if n_for_70 is None:
            raise ValueError(
                "n_components_for_variance tam spektrumla None döndü — "
                "beklenmeyen durum, tam spektrumun toplamı 1.0 olmalıydı"
            )
    except ValueError as exc:
        print(
            "BAŞARISIZ: sayısal hesaplama sonlu olmayan (NaN/inf) bir değere ya da "
            "sıfır normlu bir vektöre çarptı.\n"
            f"  Ayrıntı: {exc}\n"
            "  Olası neden: default veya fully rol ortalaması boş bir dilimden "
            "geliyor, ikisi tesadüfen eşit ya da activations.npy'de bozuk bir satır "
            "var.\n"
            "  Bu bir BAŞARISIZLIKTIR, A kriteri kararı DEĞİLDİR: tanımsız veriden "
            "GEÇTİ/DÜŞTÜ çıkarılamaz. Hiçbir artefakt yazılmadı.",
            file=sys.stderr,
        )
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "assistant_axis.npy", axis_per_layer)
    np.save(OUT_DIR / "role_vectors.npy", vectors)
    (OUT_DIR / "role_names.json").write_text(json.dumps(names, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / "criterion_a.json").write_text(
        json.dumps(
            {
                **verdict,
                # Kaynak künyesi: bu dosya aylar sonra tek başına okunacak.
                # Saatten türetilen zaman damgası YOK — bu repo kimlikleri
                # içerikten türetir.
                "model": index.get("model"),
                "run_id": index.get("run_id"),
                "n_layers": int(acts.shape[1]),
                "d_model": int(acts.shape[2]),
                "middle_layer": middle,
                "n_role_vectors": len(names),
                "n_fully_role_vectors": len(fully_positions),
                "cos_by_layer": cos_by_layer,
                "explained_variance_ratio": ratios_mid.tolist(),
                "n_components_for_70pct": n_for_70,
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
Expected: PASS. Fix Round 2 sonrası: 342 passed, 7 deselected (Fix Round 1 sonrası 332, Fix Round 1 öncesi 317). Fix Round 2 toplam 10 yeni test ekledi: `tests/test_axis.py` +4, `tests/test_extract_axis.py` +3, `tests/test_capture_activations.py` +3 (`compute_run_id`).

- [ ] **Step 7: Commit**

```bash
git add src/aax/axis.py scripts/07_extract_axis.py tests/test_axis.py tests/test_extract_axis.py
git commit -m "feat: Aşama 3 eksen çıkarımı ve A kriteri"
```

- [ ] **Step 8: OPERATÖR ADIMI — A kriteri kararı**

```bash
cd ~/assistant-axis && uv run --extra ml python scripts/07_extract_axis.py
```

Çıktı `results/axis/criterion_a.json`'a yazılır ve commit edilir.

| Çıkış | Anlam | Ne zaman |
|------:|-------|----------|
| **0** | **GEÇTİ** | A kriterinin İKİ koşulu da sağlandı → Plan 3 (Aşama 4-5: steering sweep ve persona drift) yazılabilir. |
| **1** | **DÜŞTÜ** | A kriteri TAM olarak değerlendirildi ve sağlanmadı — bu da bir bilimsel sonuçtur: "Assistant Axis 1.7B ölçeğinde oluşmuyor". Eşik gevşetilmez. Llama-3.2-3B ile tekrarlamak veya çalışmayı burada sonlandırıp negatif bulguyu raporlamak arasında karar verilir. |
| **2** | **BAŞARISIZ** | Bir karar DEĞİLDİR — girdi eksik/bayat/tanımsız (`role_expression.json` yok/bozuk, ifade haritası bayat, hiç `fully` yok, hiç `default` yok) YA DA sayısal bölümde bir `ValueError` (Fix Round 2: `contrast_axis`/`cosine`/`n_components_for_variance`'ın attığı HER `ValueError` — sıfır normlu kontrast, aktivasyonda NaN/inf — artık burada yakalanır; eskiden yakalanmayan bir `ValueError` yorumlayıcıyı çıkış 1 ile döndürüyordu, yani "DÜŞTÜ" ile AYIRT EDİLEMEZ bir çökme oluyordu). Veri düzeltilip tekrar koşulur; bu çıktı `criterion_a.json`'a yazılmaz (Fix Round 2: hiçbir artefakt de yazılmaz — hesaplama tamamlanmadan hiçbir `np.save`/`write_text` çağrılmaz). |

Fix Round 1 (Kritik 1/2, Bulgu 3/5/6, Minor) ayrı bir takip commit'idir — değişen dosyalar:
`src/aax/axis.py`, `scripts/07_extract_axis.py`, `tests/test_axis.py`,
`tests/test_extract_axis.py` (yeni). Ayrıntı: p2-task-7-report.md, Fix Round 1.

Fix Round 2 (Önemli 1/2, "Also fix" — run_id, `n_components_for_70pct` tipi,
bellek, kullanılmayan `capsys`) ayrı bir takip commit'idir — değişen dosyalar:
`src/aax/axis.py`, `scripts/07_extract_axis.py`, `scripts/05_capture_activations.py`,
`tests/test_axis.py`, `tests/test_extract_axis.py`, `tests/test_capture_activations.py`.
Yukarıdaki kod blokları ve test sayıları bu tur sonrasının hâlini gösterir.
Ayrıntı: p2-task-7-report.md, Fix Round 2.

---

## Final Fix Wave — dal geneli kod incelemesi (2026-08-06)

Dalın "hazır" ilan edilmesinden önceki son düzeltme dalgası. **Yukarıdaki Task
bölümlerindeki kod blokları bu dalgadan ÖNCEKİ hâli gösterir**; çelişki hâlinde
kaynak kod ve bu bölüm geçerlidir. Ayrıntılı gerekçe, ölçüm ve test eşlemesi:
`.superpowers/sdd/p2-final-fix-report.md`.

| # | Bulgu | Değişen | Test |
|---|---|---|---|
| A1 | A kriterinin iki koşulu bağımsızdı; geçme bölgesinin yarısı hipotezin aleyhineydi (`cos=+0.95, persentil=0.0` → `passed: True`) | `src/aax/axis.py`, spec Bölüm 7 | `tests/test_axis.py` — dört işaret×desil kombinasyonu |
| B1 | `07`'de yalnızca sayısal blok sarılıydı; eksik `activations.npy` (FileNotFoundError) ve indeks/matris boyu uyuşmazlığı (IndexError) **çıkış 1** = "A KRİTERİ DÜŞTÜ" veriyordu | `scripts/07_extract_axis.py` — `main()` gövdesinin tamamı sarıldı, `activations.npy`/`activations_index.json` için açık korumalar | `tests/test_extract_axis.py` — eksik/bozuk dosya, uzun `rows`, aralık dışı `middle_layer`, öngörülmemiş istisna, Ctrl-C yutulmuyor |
| B2 | 2 rol vektöründen `passed: True` üretilebiliyordu (yalnızca `UYARI:` basılıp devam ediliyordu) | `scripts/07` — `--min-role-vectors` (varsayılan 40, spec Bölüm 9), altında çıkış 2 | `tests/test_extract_axis.py` — taban ve bilinçli gevşetme |
| B3 | `n_rows`/`run_id` yazılıyor ama okunmuyordu; `06` hiç `run_id` yazmıyordu | `scripts/05`, `scripts/06`, `scripts/07`, `src/aax/rollouts.py::rollouts_run_id` (tek kaynak) | `tests/test_extract_axis.py`, `tests/test_label_and_train_probe.py`, `tests/test_rollouts.py` |
| C1 | Kapı `f"the role of a {role}"` ile puanlıyordu, üretim ise katalog açıklamasıyla — doğrulanan prompt üretimdeki prompt DEĞİLDİ | `scripts/03_judge_gate.py` katalogdan okuyor | `tests/test_judge_gate.py` — iki promptun birebir aynı olduğu |
| C2 | Bloklayıcı kapıda asgari `n` yoktu; 3 dolu satır "KAPI AÇIK" diyebiliyordu | `scripts/03` — `--min-labelled` (varsayılan 40) | `tests/test_judge_gate.py` |
| C3 | `06 --dry-run` statik tavanla kıyaslıyor, cache'i yok sayıyordu | `scripts/06::run_dry_run` — `would_call` + `remaining_budget` (`00` deseni); `src/aax/judge.py::build_role_score_prompts` | `tests/test_label_and_train_probe.py` |
| C4 | `05` ~2.000 batch'i sessizce koşuyordu; tek hata tüm geçişi çöpe atıyordu | `scripts/05` — batch başına ilerleme, checkpoint, `--start-row` | `tests/test_capture_activations.py` |
| C5 | `04 --limit` kanonik `rollouts.jsonl`'ı işaretsiz eziyordu | `data/rollouts_meta.json` (`limit`/`n`/`run_id`), `05` pilotu `--allow-pilot` olmadan reddediyor | `tests/test_generate_rollouts.py`, `tests/test_capture_activations.py`, `tests/test_rollouts.py` |
| C6 | `select_specs` bitişik dilim alıyordu; `--limit 100`'ün 90 rol satırı tek rolden geliyordu | `scripts/04::stride_sample` | `tests/test_generate_rollouts.py` |
| D1 | `prompts.py` `payload["roles"]` → çıplak `KeyError`, `06`'da çıkış 1 ("probe güvenilmez") | `src/aax/prompts.py` | `tests/test_prompts.py` |
| D2 | `gpu` marker'ı hem "CUDA" hem "torch" demekti; `activations.py`'nin ucuz doğruluk sınaması hiç koşmuyordu | `pyproject.toml` (`ml`/`gpu`), `tests/test_activations.py` | varsayılan koşu artık `ml` testini içeriyor |
| D3 | `04`/`05`'in `main()`'i ve satır kimliği sözleşmesi test edilmiyordu | — | `tests/test_generate_rollouts.py`, `tests/test_capture_activations.py` |
| D4 | `results/axis/*.npy` global `*.npy` kuralına takılıyordu | `.gitignore` negasyonu | — (manuel `git check-ignore` doğrulaması) |
| D5 | `02` kayıtları elle kuruyor, boş yanıt korumasını atlıyordu | `scripts/02_pilot_rollouts.py` → `rollout_record` | `tests/test_pilot_rollouts.py` (yeni) |
| D6 | `criterion_a.json` kendi içinde bağdaştırılamıyordu | `07` — `cumulative_variance_at_10` | `tests/test_extract_axis.py` |
| D7 | `06`'da kullanılmayan `import numpy as np` | `scripts/06` | — |
| D8 | `06` iki ayrı durum için 1 döndürüyordu | `--dry-run` bütçe reddi artık 2 | `tests/test_label_and_train_probe.py` |

**Kapsam dışı bırakıldı (bilinçli):** spec'in probe geri çekilme kuralı (rol
başına 15 rollout, ~180 çağrı) uygulanmadı. `06`'nın hata mesajı artık bunu
AÇIKÇA söylüyor ve operatöre iki gerçek seçeneği + harcanan/kalan bütçeyi
yazıyor. Boşluk proje sahibine ayrıca bildirildi.

---

## Closure Fix Wave — dalın hazır ilanından önceki son düzeltme dalgası (2026-08-06)

Final Fix Wave'in doğrulama geçişinde bulunan bir Kritik ve beş Önemli bulgu.
**Yukarıdaki Task bölümlerindeki VE Final Fix Wave'deki kod blokları bu
dalgadan ÖNCEKİ hâli gösterir**; çelişki hâlinde kaynak kod ve bu bölüm
geçerlidir. Ayrıntılı gerekçe, ölçüm ve test eşlemesi:
`.superpowers/sdd/p2-closure-fix-report.md`.

| # | Bulgu | Değişen | Test |
|---|---|---|---|
| Kritik 1 | `rollouts_run_id` `answer`'ı BİLEREK dışarıda bırakıyordu; `04` `temperature=1.0` ile örneklediği için aynı spec'lerin iki üretimi AYNI kimlikle FARKLI cevaplar üretebiliyordu — `05`in aktivasyonu ile `06`nın etiketi farklı cevapları tarif edip `07`nin üç bütünlük kontrolünün (satır sayısı, anahtar kapsaması, `run_id`) hepsini geçebiliyordu | `src/aax/rollouts.py::rollouts_run_id` — blob artık `answer`'ı da katıyor; `04`/`05`/`06` aynı fonksiyonu çağırdığı için otomatik yayılıyor | `tests/test_rollouts.py` (aynı spec + farklı cevap → farklı kimlik), `tests/test_capture_activations.py`, `tests/test_extract_axis.py` — uçtan uca: aynı spec'ler farklı cevaplarla üretilmiş iki `run_id` `07` tarafından reddediliyor |
| Önemli 2 | `--start-row` yalnızca matris ŞEKLİNE bakıyordu; kısmi işaret (`activations_partial.json`) VARSA doğrulanıyordu ama YOKSA sessizce geçiliyordu — tam da "aynı satır sayısı, farklı rollout kümesi" senaryosu | `scripts/05_capture_activations.py::_load_resume_prefix` — işaret artık `--start-row` için ZORUNLU | `tests/test_capture_activations.py::test_start_row_refuses_when_no_partial_marker_exists_even_if_shape_matches` |
| Önemli 3 | `06`'da çıplak `read_rollouts(...)` çağrısı `try/except`'in dışındaydı; eksik/bozuk dosya çıplak istisnayla yorumlayıcıyı çıkış 1'e ("probe güvenilmez") düşürüyordu | `scripts/06_label_and_train_probe.py::main` — `05`'teki desenin aynısıyla sarıldı | `tests/test_label_and_train_probe.py` — eksik ve bozuk `rollouts.jsonl` artık çıkış 2 |
| Önemli 4 | Checkpoint yazımı (`np.save(ACTS_PATH, acts)`) planlanan ölçekte ~80 kez tam matrisi (~3.67 GB) baştan yazıyordu, hem de non-atomik | `scripts/05_capture_activations.py` — `--checkpoint-every` varsayılanı 25→250; `_atomic_save_npy` (`aax.rollouts.write_rollouts`'un tempfile+fsync+rename deseni) | `tests/test_capture_activations.py::test_checkpoint_every_default_is_250_not_25`, `test_atomic_save_npy_*` |
| Önemli 5 | `06` `05`'in çıktısına bağımlı değildi (`rollouts.jsonl`'ı doğrudan okur) — künye kontrolü yalnızca `05`'e bağlıydı, bir pilot künye hakem harcamasından (~200 çağrı) önce reddedilmiyordu | `scripts/06_label_and_train_probe.py::main` — `load_rollouts_meta` + `--allow-pilot` (05'in deseni) | `tests/test_label_and_train_probe.py` — pilot künye reddi ve `--allow-pilot` ile bilinçli geçiş |
| Önemli 6 | `02`'nin varsayılanı (8 rol × 5 soru = 40) kapının tabanıyla (`03::MIN_LABELLED`=40) BİREBİR aynıydı, sıfır boşluk payı | `scripts/02_pilot_rollouts.py` — varsayılan `--roles` 8→9 (9×5=45); 40'ın altına düşerse UYARI | `tests/test_pilot_rollouts.py` |
| Minor | `tests/test_activations.py`'nin `ml`-marked testi bare `import torch` — `ml` extra'sı kurulu değilse SKIP değil FAIL | `tests/test_activations.py` — `pytest.importorskip("torch")` | mevcut test artık extra'sız `pytest --collect-only` ile de patlamıyor |
| Minor | `criterion_a.json` gevşetilmiş `--min-role-vectors` tabanını taşımıyordu — yalnızca `n_role_vectors`'a bakarak dolaylı çıkarılabilirdi | `scripts/07_extract_axis.py` — `min_role_vectors` alanı eklendi | `tests/test_extract_axis.py::test_criterion_a_records_the_min_role_vectors_floor_actually_used` |
| Minor | `cumulative_variance_at_10` adı sabit "10" varsayıyordu; `--min-role-vectors` 10'un altına gevşetilirse alan adı kendi içeriğiyle çelişiyordu | `scripts/07_extract_axis.py` — `cumulative_variance_top_components` + ayrı `cumulative_variance_n_components` | `tests/test_extract_axis.py::test_cumulative_variance_n_components_reflects_fewer_than_10_role_vectors` |
| Minor | `03_judge_gate.py`'nin yedi `raise SystemExit(...)` yolu (1 `run_machine`, 6 `run_score`) çıkış 1 veriyordu — kapının GERÇEK reddiyle ("KAPI KAPALI") aynı kod; `r["description"]` çıplak indekslemesi de aynı sınıftan bir `KeyError` | `scripts/03_judge_gate.py` — hepsi `print(...) + return 2`; `description` eksikliği `except KeyError` ile yakalanıyor | `tests/test_judge_gate.py` |

**Etki alanı doğrulaması (Kritik 1):** `rollouts_run_id`'nin dört tüketicisi
(`04`→`rollouts_meta.json`, `05`→`activations_index.json`,
`06`→`role_expression.json`, `07`'nin künye eşitliği kontrolü) hepsi AYNI
fonksiyonu çağırıyor; hiçbiri `answer`'ın kimliğin dışında olduğuna
BAĞIMLI değildi. Değişiklik tek fonksiyonda kaldı, üç script de otomatik
olarak yeni davranışı miras aldı — ayrıca bir kod değişikliği gerekmedi.

---

## Plan 2 Tamamlanma Kriterleri

- [ ] `uv run --extra dev --extra ml pytest -q` yeşil; `-m ml` ve `-m gpu` ayrıca geçiyor
- [ ] `data/judge_gate.json` — `passed: true`, uyum ≥ %75, `n` ≥ 40
- [ ] `data/rollouts.jsonl` + `data/rollouts_meta.json` — ~16.000 kayıt, `limit: null`
- [ ] `data/activations.npy` — `[~16000, L, d_model]` float32
- [ ] `data/role_expression.json` — held-out uyum raporlanmış
- [ ] `results/axis/criterion_a.json` — A kriteri kararı, commit edilmiş
- [ ] Gateway bütçesi: `stage05_judge_gate` ≤ 15, `stage2_probe_labels` ≤ 300
