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
