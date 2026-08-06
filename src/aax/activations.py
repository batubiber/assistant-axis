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
