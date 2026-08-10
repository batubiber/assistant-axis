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
    activations = np.asarray(activations)
    if activations.ndim != 3:
        # 2 veya 4 boyutlu bir girişte shape[1] katman ekseni DEĞİLDİR —
        # aralık kontrolü yanlış ekseni doğrular ve dilim sessizce yanlış
        # şeyi döndürür. 3 boyut (n_rows, n_layers, d_model) zorunlu.
        raise ValueError(
            "activations 3 boyutlu olmalı [n_rows, n_layers, d_model], "
            f"ndim={activations.ndim}"
        )
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
    if d.ndim != 1:
        # Skaler (0-d) ya da matris (2-d+) bir `direction` burada sessizce
        # yanlış şekilde broadcast edilebilirdi. Uzunluk-1 bir yönün TÜM
        # d_model boyutlarına düz bir kayma yaymasına karşı koruma ise
        # burada değil `steer()`'de: orada `bundle.d_model` bilgisi var.
        raise ValueError(f"steering yönü 1 boyutlu olmalı, ndim={d.ndim}")
    if not np.isfinite(d).all():
        raise ValueError("steering yönü sonlu olmayan (NaN/inf) değer içeriyor")
    if not np.isfinite(strength) or not np.isfinite(layer_norm):
        raise ValueError("strength ve layer_norm sonlu olmalı")
    if layer_norm < 0:
        raise ValueError("layer_norm negatif olamaz — steering yönünü ters çevirir")
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

    direction_arr = np.asarray(direction)
    if direction_arr.ndim != 1:
        # `shape[-1]` erişimi (hemen aşağıdaki d_model kontrolü) ndim=0'da
        # IndexError atardı — bu kontrol ONDAN ÖNCE gelmeli. Mesaj
        # `steering_delta`'nın ürettiğiyle birebir aynı: 0-d/2-d+ bir yön
        # için okuyucu hangi yoldan geçtiğinden bağımsız aynı temiz Türkçe
        # hatayı görmeli.
        raise ValueError(f"steering yönü 1 boyutlu olmalı, ndim={direction_arr.ndim}")
    if direction_arr.shape[-1] != bundle.d_model:
        raise ValueError(
            "steering yönü d_model ile uyuşmuyor: "
            f"{direction_arr.shape[-1]} != {bundle.d_model}"
        )

    delta_np = steering_delta(direction, strength, layer_norm)
    handle = None
    try:
        target = bundle.model.model.layers[layer]
        # Delta'yı BİR KEZ, hook takılırken, modelin kendi dtype'ında kur —
        # forward başına yeniden cast'lemek yerine (sweep boyunca ~420k
        # çağrı). Hook içinde `.to(hidden.dtype)` yerine `.to(hidden)`
        # kullanıyoruz: `Tensor.to`, hedef zaten aynı dtype/device'taysa
        # kopyasız kendini döndürür, yani tek-cihazlı bugünkü kurulumda
        # pratik ek maliyet yok. Ayrıca device'ı da hedefe göre kurar — çok
        # cihazlı bir device_map altında bu katman `bundle.model.device`'tan
        # FARKLI bir cihazda olabilir; salt `.to(dtype)` kullansaydık delta
        # yanlış cihazda kalır ve toplama SESSİZCE yanlış katmanı izlemek
        # yerine device-uyuşmazlığı hatasıyla AÇIKÇA patlardı.
        delta = torch.tensor(
            delta_np, dtype=bundle.model.dtype, device=bundle.model.device
        )

        def hook(_module, _inputs, output):
            is_tuple = isinstance(output, tuple)
            hidden = output[0] if is_tuple else output
            shifted = hidden + delta.to(hidden)
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
        # `prepend=True`, hook listesinin BAŞINA ekler: biz bu KATMANIN
        # kendi modül hook listesinde her zaman ilk çalışırız, delta'yı
        # ekleriz, ve bizden sonra kayıtlı her gözlemci (transformers'ınki
        # dahil) zaten steering'li tensörü görür. NİTELEYİCİ NOT: bu garanti
        # yalnızca bu MODÜLE özgü hook listesi için geçerlidir —
        # `register_module_forward_hook` ile kaydedilen GLOBAL hook'lar
        # `prepend=True`'dan bağımsız olarak yine her modül-özel hook'tan
        # (bizimki dahil) ÖNCE çalışır. Bu projede böyle bir global hook
        # kullanılmıyor; biri eklenirse bu satırdaki garanti geçersiz kalır.
        # Bu aynı zamanda anlamlı olan seçim: bir katmanın "çıktısı" onu
        # gözlemleyen her şeye göre steering uygulanmış olmalı, kaydın
        # zamanlamasına göre değil — ileride steering AÇIKKEN aktivasyon
        # yakalayan bir plan bunu varsayacaktır.
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
