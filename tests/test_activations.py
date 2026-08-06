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
    """Gerçek bf16/CUDA yolunda padding sızıntısı yok — ORANSAL tolerans ile.

    Bu test bilinçli olarak gerçek üretim yolunu (GPU, bf16, `batch_size>1`
    ile gerçek batch'leme) egzersiz ediyor; CPU float32 testi (yukarıda)
    maskeleme mantığını izole ederken, bu test bf16 sayısal gürültüsü
    ALTINDA maskelemenin hâlâ doğru çalıştığını doğruluyor. Mutlak değil
    oransal karşılaştırma kullanıyoruz çünkü bu modeldeki residual stream
    büyüklükleri yüzlerce-binlerce mertebesinde (bkz. rapor) ve bf16 mutlak
    farkları büyüklükle orantılı üretir.

    Tolerans ölçümden geliyor, uydurma değil: p2-task-4-report.md Fix
    Round 1'de aynı kısa+uzun senaryo için max_abs_diff=13.333,
    max(abs(alone))=2258.667 → oransal hata = %0.590. CPU float32
    kontrolünde aynı oran %0.000033'e düşüyor, yani bu %0.59 gerçekten
    bf16'dan kaynaklanıyor. Aşağıdaki TOLERANCE=%2, ölçülenin ~3.4 katı:
    bf16 gürültüsünü rahatça geçecek kadar geniş, ama gerçek bir maskeleme
    hatasını (büyüklüğü çok daha büyük olurdu) yakalayacak kadar sıkı.
    """
    from aax.activations import mean_response_activations
    from aax.model import load_hf_model

    bundle = load_hf_model()  # config.TARGET_MODEL, cuda + bf16 (varsayılan)
    tok = bundle.tokenizer
    short, long = _kisa_uzun_ornekler(tok)

    alone = mean_response_activations(bundle, [short], batch_size=1)
    together = mean_response_activations(bundle, [short, long], batch_size=2)

    max_abs_diff = float(np.abs(alone[0] - together[0]).max())
    max_abs_alone = float(np.abs(alone[0]).max())
    rel_err = max_abs_diff / max_abs_alone

    # Ölçülen: %0.590 (bkz. docstring). Kasıtlı olarak ~3.4x güvenlik payı.
    TOLERANCE = 0.02
    assert rel_err < TOLERANCE, (
        f"oransal hata %{rel_err * 100:.4f}, sınır %{TOLERANCE * 100:.0f} üstünde "
        "— bu ölçülen bf16 gürültüsünün çok üzerinde, padding sızıntısına işaret edebilir "
        f"(max_abs_diff={max_abs_diff:.4f}, max(abs(alone))={max_abs_alone:.4f})"
    )
