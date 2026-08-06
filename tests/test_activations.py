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
