"""Assistant Axis üzerinde steering.

`activations.py`'nin aynadaki hâli: o, decoder katmanının çıktısını OKUR;
bu, aynı tensöre forward hook ile sabit bir vektör EKLER. Aynı tensör
olması şart — yakaladığımız yerden başka bir yere yazarsak ölçtüğümüz
eksende steering yapmamış oluruz.

Güç her zaman o katmanın KENDİ ortalama residual normunun oranıdır.
Ölçülen normlar Qwen3-1.7B'de L14=137.0, L19=435.6 — üç kat fark. Mutlak
ölçek kullanmak iki katmanı karşılaştırılamaz hâle getirirdi.

Steering her token pozisyonuna uygulanır (prompt dahil), makalenin
Bölüm 3.2.1'deki kurulumu.
"""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np


def mean_residual_norm(activations: np.ndarray, layer: int) -> float:
    """Bir katmandaki ortalama L2 residual normu.

    `activations`: [n_rows, n_layers, d_model]. Steering ölçeği buradan gelir.
    """
    if not 0 <= layer < activations.shape[1]:
        raise ValueError(
            f"katman aralık dışı: {layer} (0-{activations.shape[1] - 1})"
        )
    rows = np.asarray(activations[:, layer, :], dtype=np.float64)
    return float(np.linalg.norm(rows, axis=1).mean())


def steering_delta(
    direction: np.ndarray, strength: float, layer_norm: float
) -> np.ndarray:
    """Katmanın çıktısına eklenecek sabit vektör: `strength · layer_norm · v̂`."""
    d = np.asarray(direction, dtype=np.float64)
    if not np.isfinite(d).all():
        raise ValueError("steering yönü sonlu olmayan (NaN/inf) değer içeriyor")
    if not np.isfinite(strength) or not np.isfinite(layer_norm):
        raise ValueError("strength ve layer_norm sonlu olmalı")
    norm = np.linalg.norm(d)
    if norm == 0:
        raise ValueError("steering yönü sıfır vektör — yön tanımsız")
    return (d / norm) * (strength * layer_norm)


@contextmanager
def steer(bundle, *, layer: int, direction, strength: float, layer_norm: float):
    """Verilen katmanın çıktısına steering vektörünü ekleyen hook'u takar.

    Çıkışta — istisna hâlinde de — hook kaldırılır. Sızan bir hook sonraki
    her üretimi sessizce bozar.
    """
    import torch

    if not 0 <= layer < bundle.n_layers:
        raise ValueError(f"katman aralık dışı: {layer} (0-{bundle.n_layers - 1})")

    delta_np = steering_delta(direction, strength, layer_norm)
    handle = None
    try:
        target = bundle.model.model.layers[layer]
        delta = torch.tensor(
            delta_np, dtype=torch.float32, device=bundle.model.device
        )

        def hook(_module, _inputs, output):
            is_tuple = isinstance(output, tuple)
            hidden = output[0] if is_tuple else output
            shifted = hidden + delta.to(hidden.dtype)
            return (shifted, *output[1:]) if is_tuple else shifted

        # prepend=True — NEDEN GEREKLİ, sonradan silinmesin diye:
        #
        # PyTorch forward hook'ları KAYIT SIRASINA göre çalışır, ve bir hook
        # None-olmayan bir değer döndürürse SONRAKİ hook'lar ve çağıranın
        # kendisi o değeri görür. Yani katmanın etkin çıktısı — bir sonraki
        # katmana giden, sonunda logit'lere ulaşan tensör — hook sırasından
        # BAĞIMSIZ olarak zaten steering'lidir; sıra hesaplamayı değiştirmez.
        #
        # Değiştirdiği şey, bizden SONRA kayıtlı bir GÖZLEMCİ hook'un ne
        # gördüğü. transformers, `output_hidden_states=True` istendiğinde
        # her decoder katmanına kendi okuma-amaçlı forward hook'unu tembelce
        # ve KALICI olarak takıyor (bkz. transformers/utils/output_capturing.py,
        # `install_output_capuring_hook`). O hook biz `steer()`'e girmeden
        # ÖNCE (örn. steering'siz bir taban forward'da) takılmışsa, varsayılan
        # `register_forward_hook` sırasında BİZDEN ÖNCE çalışır ve delta
        # eklenmeden önceki değeri yakalar — hesaplama doğru olsa da
        # `hidden_states` anlık görüntüsü steering'i GÖRMEZ. Bu tam olarak
        # `test_hook_shifts_the_target_layer_output_by_exactly_the_delta`
        # testinin başta düşmesinin sebebiydi: logit'ler 0.75'e kadar
        # değişiyordu ama `hidden_states[l+1]` bit-bir-bit aynı kalıyordu.
        #
        # `prepend=True`, hook listesinin BAŞINA ekler: biz her zaman ilk
        # çalışırız, delta'yı ekleriz, ve bizden sonra kayıtlı her gözlemci
        # (transformers'ınki dahil) zaten steering'li tensörü görür. Bu aynı
        # zamanda anlamlı olan seçim: bir katmanın "çıktısı" onu gözlemleyen
        # her şeye göre steering uygulanmış olmalı, kaydın zamanlamasına göre
        # değil — ileride steering AÇIKKEN aktivasyon yakalayan bir plan
        # bunu varsayacaktır.
        handle = target.register_forward_hook(hook, prepend=True)
        yield
    finally:
        if handle is not None:
            handle.remove()


def generate_steered(
    bundle,
    messages: list[dict],
    *,
    layer: int,
    direction,
    strength: float,
    layer_norm: float,
    max_new_tokens: int = 120,
) -> str:
    """Steering açıkken tek bir yanıt üret.

    HF transformers kullanılır, vLLM değil: makale vLLM steering'inin
    tutarlı %2-3 daha kötü ölçtüğünü raporluyor (Ek G.5).
    """
    import torch

    tok = bundle.tokenizer
    text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tok(text, return_tensors="pt").to(bundle.model.device)

    with steer(
        bundle,
        layer=layer,
        direction=direction,
        strength=strength,
        layer_norm=layer_norm,
    ), torch.no_grad():
        out = bundle.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(
        out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()
