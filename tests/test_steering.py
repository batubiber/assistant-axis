import numpy as np
import pytest

from aax.steering import mean_residual_norm, steering_delta


def test_mean_residual_norm_is_the_mean_l2_over_rows():
    acts = np.zeros((3, 2, 4), dtype=np.float32)
    acts[0, 1, :] = [3.0, 4.0, 0.0, 0.0]   # norm 5
    acts[1, 1, :] = [0.0, 0.0, 6.0, 8.0]   # norm 10
    acts[2, 1, :] = [0.0, 0.0, 0.0, 0.0]   # norm 0
    assert mean_residual_norm(acts, layer=1) == pytest.approx(5.0)


def test_mean_residual_norm_rejects_out_of_range_layer():
    acts = np.zeros((2, 3, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="katman"):
        mean_residual_norm(acts, layer=3)


def test_steering_delta_scales_unit_direction_by_strength_and_norm():
    d = np.array([1.0, 0.0, 0.0])
    out = steering_delta(d, strength=0.25, layer_norm=200.0)
    assert out == pytest.approx([50.0, 0.0, 0.0])


def test_steering_delta_normalises_a_non_unit_direction():
    d = np.array([0.0, 3.0, 4.0])   # norm 5
    out = steering_delta(d, strength=1.0, layer_norm=10.0)
    assert np.linalg.norm(out) == pytest.approx(10.0)


def test_steering_delta_zero_strength_is_zero_vector():
    out = steering_delta(np.array([1.0, 1.0]), strength=0.0, layer_norm=99.0)
    assert np.allclose(out, 0.0)


def test_steering_delta_rejects_non_finite():
    with pytest.raises(ValueError, match="sonlu"):
        steering_delta(np.array([np.nan, 1.0]), strength=0.5, layer_norm=10.0)


def test_steering_delta_rejects_zero_direction():
    with pytest.raises(ValueError, match="sıfır"):
        steering_delta(np.zeros(3), strength=0.5, layer_norm=10.0)


def test_mean_residual_norm_rejects_wrong_ndim():
    """2 boyutlu girişte shape[1] katman ekseni DEĞİLDİR — aralık kontrolü
    yanlış ekseni doğrular ve dilim sessizce yanlış şeyi döndürür."""
    with pytest.raises(ValueError, match="boyutlu"):
        mean_residual_norm(np.zeros((3, 4)), layer=0)
    with pytest.raises(ValueError, match="boyutlu"):
        mean_residual_norm(np.zeros((2, 3, 4, 5)), layer=0)


def test_steering_delta_rejects_non_1d_direction():
    with pytest.raises(ValueError, match="boyutlu"):
        steering_delta(np.zeros((2, 2)), strength=0.5, layer_norm=10.0)
    with pytest.raises(ValueError, match="boyutlu"):
        steering_delta(np.array(1.0), strength=0.5, layer_norm=10.0)


def test_steering_delta_rejects_negative_layer_norm():
    """Negatif layer_norm kabul edilirse steering yönünü sessizce ters çevirir."""
    with pytest.raises(ValueError, match="negatif"):
        steering_delta(np.array([1.0, 0.0]), strength=0.5, layer_norm=-1.0)


@pytest.mark.ml
def test_steering_hook_runs_before_a_later_registered_observer_hook():
    """`steer()` `prepend=True` kullanmalı, yoksa steering görünmez olur.

    PyTorch forward hook'ları kayıt sırasına göre çalışır ve bir hook
    None-olmayan bir değer döndürürse SONRAKİ hook'lar (ve çağıranın
    kendisi) onu görür — katmanın etkin çıktısı hook sırasından BAĞIMSIZ
    olarak zaten steering'lidir. Sıra yalnızca bizden SONRA kayıtlı bir
    GÖZLEMCİ hook'un ne gördüğünü değiştirir.

    Bu, sahte transformers'ın `output_hidden_states` yakalayıcısı gibi:
    steer()'den ÖNCE kayıtlı, sadece okuyor, hiçbir şey döndürmüyor. Bu
    testte `steer()`'in hook'u BEKLENMEDİK ŞEKİLDE varsayılan (prepend
    olmayan) sırayla kaydolursa, gözlemci delta eklenmeden ÖNCEKİ değeri
    görür ve bu test düşer — tam olarak GPU testinin ilk koşuda düştüğü
    sebep (`hidden_states[l+1]` steering'siz görünüyordu, oysa logit'ler
    0.75'e kadar değişmişti). Bu yüzden GPU'suz, saniyenin altında koşan
    bu test artık davranışı sabitliyor.
    """
    torch = pytest.importorskip("torch")

    from aax.steering import steer

    class FakeLayer(torch.nn.Module):
        def forward(self, hidden_states):
            return hidden_states  # Qwen3DecoderLayer gibi düz tensör döner

    class FakeBaseModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([FakeLayer()])

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = FakeBaseModel()
            self.device = torch.device("cpu")
            self.dtype = torch.float32

    class FakeBundle:
        def __init__(self):
            self.model = FakeModel()
            self.n_layers = 1
            self.d_model = 4

    bundle = FakeBundle()
    observed = []
    # transformers'ın `output_hidden_states` yakalayıcısı gibi: steer()'den
    # ÖNCE kayıtlı, hiçbir şey döndürmeyen salt-okunur bir gözlemci.
    bundle.model.model.layers[0].register_forward_hook(
        lambda _m, _i, output: observed.append(output)
    )

    direction = np.zeros(4, dtype=np.float32)
    direction[0] = 1.0
    hidden = torch.zeros(1, 1, 4)
    with steer(bundle, layer=0, direction=direction, strength=1.0, layer_norm=10.0):
        bundle.model.model.layers[0](hidden)

    expected = hidden + torch.tensor([10.0, 0.0, 0.0, 0.0])
    assert torch.equal(observed[0], expected), (
        "önceden kayıtlı gözlemci steering'den ÖNCEKİ değeri gördü — "
        "steer() prepend=True kullanmalı"
    )


@pytest.mark.ml
def test_steer_rejects_direction_that_does_not_match_d_model():
    """`d_model` uyuşmazlığı erken reddedilmeli — aksi hâlde uzunluk-1 (veya
    başka yanlış uzunlukta) bir yön tüm boyutlara sessizce broadcast edilir.

    Hata `bundle.model`'e hiç dokunmadan, katman aralığı kontrolünden hemen
    sonra fırlatılmalı — bu yüzden sahte bundle'ın `model`'i `None`.
    """
    pytest.importorskip("torch")

    from aax.steering import steer

    class FakeBundle:
        n_layers = 2
        d_model = 4
        model = None

    bundle = FakeBundle()
    with pytest.raises(ValueError, match="d_model"):
        with steer(bundle, layer=0, direction=np.zeros(3), strength=0.5, layer_norm=10.0):
            pass  # pragma: no cover - hataya kadar erişilmemeli


@pytest.mark.ml
def test_steer_rejects_0d_direction_with_clean_value_error_not_index_error():
    """0-d (skaler) bir `direction`, `direction_arr.shape[-1]` erişiminde
    IndexError DEĞİL, `steering_delta`'nın ürettiğiyle aynı temiz Türkçe
    ValueError'ı vermeli.

    `steering_delta` tek başına bu girişi doğru ele alıyor
    (`test_steering_delta_rejects_non_1d_direction`), ama `steer()` d_model
    kontrolü ndim kontrolünden ÖNCE çalışırsa oraya hiç ulaşılmıyordu —
    bkz. p3-task-1-fix2-brief.md F1. Sahte bundle'ın `model`'i `None`:
    hata `bundle.model`'e hiç dokunmadan fırlatılmalı.
    """
    pytest.importorskip("torch")

    from aax.steering import steer

    class FakeBundle:
        n_layers = 2
        d_model = 4
        model = None

    bundle = FakeBundle()
    # `pytest.raises(ValueError, ...)` IndexError'ı KABUL ETMEZ — regresyon
    # (guard'ın yanlış sırası) bu testi ValueError yerine IndexError ile
    # düşürerek işaret eder.
    with pytest.raises(ValueError, match="boyutlu"):
        with steer(bundle, layer=0, direction=np.array(1.0), strength=0.5, layer_norm=10.0):
            pass  # pragma: no cover - hataya kadar erişilmemeli


@pytest.mark.ml
def test_steering_delta_lands_on_every_position_of_every_row_in_a_batch():
    """[B, S, D] girişte delta HER satırın HER token pozisyonuna eklenmeli —
    makalenin Bölüm 3.2.1'deki "her token pozisyonu" kurulumunun dayandığı
    broadcast özelliği. Önceki testler yalnızca `torch.zeros(1, 1, 4)`
    (B=S=1) kullanıyordu ve bu özelliği hiç sınamıyordu.
    """
    torch = pytest.importorskip("torch")

    from aax.steering import steer

    class FakeLayer(torch.nn.Module):
        def forward(self, hidden_states):
            return hidden_states

    class FakeBaseModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([FakeLayer()])

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = FakeBaseModel()
            self.device = torch.device("cpu")
            self.dtype = torch.float32

    class FakeBundle:
        def __init__(self):
            self.model = FakeModel()
            self.n_layers = 1
            self.d_model = 4

    bundle = FakeBundle()
    direction = np.zeros(4, dtype=np.float32)
    direction[0] = 1.0
    B, S, D = 3, 5, 4
    hidden = torch.zeros(B, S, D)

    with steer(bundle, layer=0, direction=direction, strength=1.0, layer_norm=10.0):
        out = bundle.model.model.layers[0](hidden)

    expected = torch.zeros(B, S, D)
    expected[..., 0] = 10.0
    assert torch.equal(out, expected), "delta her satırın her pozisyonuna eklenmemiş"


@pytest.mark.ml
def test_generate_steered_strips_prompt_disables_thinking_and_keeps_hook_active_every_step():
    """`generate_steered`'in üç sözleşme parçası — hiçbiri şimdiye kadar
    test edilmiyordu:

    1. `out[0][inputs["input_ids"].shape[1]:]` dilimi prompt'u dışlıyor mu?
    2. `enable_thinking=False` gerçekten chat template'e ulaşıyor mu?
    3. Steering hook'u `generate()`'in TÜM decode adımlarında aktif mi
       kalıyor, yoksa yalnızca prefill'de mi?

    Sahte `generate`, 3 "adım" simüle eder (prefill dâhil) ve her adımda
    hedef katmanı `nn.Module.__call__` üzerinden GERÇEKTEN çağırır — bu
    yüzden `steer()`'in kaydettiği forward hook her adımda da tetiklenir.
    Hook'un yalnızca ilk adımda değil SON adımda da aktif olduğunu
    doğrulayarak, `with steer(...):`'in `generate()` çağrısının tamamını
    sarmaladığı (ve prefill'den sonra erken kaldırılmadığı) sabitlenir.
    """
    torch = pytest.importorskip("torch")

    from aax.steering import generate_steered

    class FakeLayer(torch.nn.Module):
        def forward(self, hidden_states):
            return hidden_states

    class FakeBaseModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([FakeLayer()])

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = FakeBaseModel()
            self.device = torch.device("cpu")
            self.dtype = torch.float32
            self.layer_call_count = 0
            self.last_layer_output = None

        def generate(self, input_ids, max_new_tokens=120, **_kwargs):
            tokens = input_ids
            for _ in range(3):  # prefill + 2 decode adımı simülasyonu
                hidden = torch.zeros(1, tokens.shape[1], 4)
                out = self.model.layers[0](hidden)  # steer() hook'unu tetikler
                self.layer_call_count += 1
                self.last_layer_output = out
                next_token = torch.full((1, 1), 99, dtype=tokens.dtype)
                tokens = torch.cat([tokens, next_token], dim=1)
            return tokens

    class FakeEncoding(dict):
        def to(self, _device):
            return self

    class FakeTokenizer:
        def __init__(self):
            self.eos_token_id = 0
            self.last_template_kwargs = None
            self.last_decoded_ids = None

        def apply_chat_template(self, messages, tokenize=False,
                                 add_generation_prompt=True, enable_thinking=None):
            self.last_template_kwargs = {
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "enable_thinking": enable_thinking,
            }
            return "PROMPT"

        def __call__(self, _text, return_tensors="pt"):
            return FakeEncoding(input_ids=torch.tensor([[1, 2, 3]]))

        def decode(self, ids, skip_special_tokens=True):
            self.last_decoded_ids = ids.tolist()
            return "yanıt metni"

    class FakeBundle:
        def __init__(self):
            self.model = FakeModel()
            self.tokenizer = FakeTokenizer()
            self.n_layers = 1
            self.d_model = 4

    bundle = FakeBundle()
    direction = np.zeros(4, dtype=np.float32)
    direction[0] = 1.0

    result = generate_steered(
        bundle,
        [{"role": "user", "content": "merhaba"}],
        layer=0,
        direction=direction,
        strength=1.0,
        layer_norm=10.0,
    )

    # (1) hook, üç decode adımının HEPSİNDE tetiklendi — sadece prefill'de değil
    assert bundle.model.layer_call_count == 3
    expected_shift = torch.tensor([10.0, 0.0, 0.0, 0.0])
    assert torch.allclose(bundle.model.last_layer_output[0, 0], expected_shift), (
        "steering hook'u SON decode adımında aktif değildi — erken kaldırılmış olabilir"
    )

    # (2) döndürülen dizi prompt'u dışlıyor — yalnızca 3 yeni üretilen token
    assert bundle.tokenizer.last_decoded_ids == [99, 99, 99]
    assert result == "yanıt metni"

    # (3) enable_thinking=False chat template'e ulaştı
    assert bundle.tokenizer.last_template_kwargs["enable_thinking"] is False


@pytest.mark.gpu
def test_zero_strength_leaves_logits_bit_identical():
    """α=0 steering hiçbir şeyi değiştirmemeli — hook'un kendisi çıktıyı bozmuyor."""
    import torch

    from aax.model import load_hf_model
    from aax.steering import steer

    bundle = load_hf_model()
    tok = bundle.tokenizer
    enc = tok("Merhaba, bu bir testtir.", return_tensors="pt").to(bundle.model.device)
    direction = np.zeros(bundle.d_model, dtype=np.float32)
    direction[0] = 1.0

    with torch.no_grad():
        base = bundle.model(**enc).logits
    with steer(bundle, layer=bundle.middle_layer, direction=direction,
               strength=0.0, layer_norm=137.0), torch.no_grad():
        steered = bundle.model(**enc).logits

    assert torch.equal(base, steered), "α=0 iken çıktı birebir aynı olmalı"


@pytest.mark.gpu
def test_hook_shifts_the_target_layer_output_by_exactly_the_delta():
    """Steering, hedef katmanın çıktısını TAM OLARAK delta kadar kaydırmalı.

    Aktivasyon yakalamayla aynı tensöre yazdığımızı doğrular: kaydırma
    hidden_states[l+1]'de görünmeli, hidden_states[l]'de görünmemeli.

    Bu, planın "dur sinyali" testidir: düşerse yanlış tensöre yazıyoruz
    demektir ve bundan sonraki her ölçüm anlamsızdır. Bu yüzden tolerans
    keyfi DEĞİL, gerçek bf16 aritmetiğinden türetilir:

    - Beklenen değer, `steer()`'in GERÇEKTE kurduğu bf16-kuantalı delta'dır
      (`steering_delta`'nın döndürdüğü float64 idealize değer değil).
      `delta.to(bfloat16)` tek başına ~13.7 -> 13.6875 kuantalıyor —
      float64'e karşı karşılaştırmak, eski sabit toleransın (atol=1e-1)
      %12.5'ini hesaplamanın kendisiyle hiç ilgisi olmayan bir yuvarlamaya
      harcıyordu.
    - Fark (`diff_bf16`), erken `float()`'e cast'lenmeden ÖNCE, iki bf16
      tensörünün bf16 çıkarması olarak hesaplanır — ölçmek istediğimiz
      TAM OLARAK bu.
    - Tolerans `base[L+1].abs().max()` (TÜM tensörün — tüm pozisyon ve tüm
      2048 boyutun — global maksimumu) ile DEĞİL, yalnızca `direction`'ın
      sıfır-olmayan olduğu boyut(lar)daki (burada: dim 3) değerlerin
      maksimumuyla ölçeklenir. Ölçülmüş NEDEN: bu modelde
      `base[L+1].abs().max()` **12480** — konum 0 (BOS benzeri "dikkat
      çukuru" token'ı), boyut 1793'teki, steering'le hiç ilgisi olmayan
      bilinen bir "kütlesel aktivasyon" aykırı değeri. O değerle
      ölçeklenen bir tolerans (`eps·12480≈97.5`) beklenen delta'nın
      (~13.7) TAMAMINDAN büyük olur — no-op bir hook'u bile geçirir (aşağı
      bakınız, ampirik olarak doğrulandı). Doğru ölçek, TOPLAMANIN
      GERÇEKTEN gerçekleştiği boyuttaki büyüklüktür: `base[L+1] +
      beklenen_delta`, yalnızca `direction`'ın sıfır-olmadığı boyutlarda —
      bu, hook DOĞRU çalışsaydı `got`'un o boyutlarda alacağı değerdir,
      dolayısıyla hook'un GERÇEKTE ürettiği (belki hatalı) `got`'a değil,
      yalnızca `base` (steering'den etkilenmez) ve `steering_delta`'nın
      matematiksel çıktısına bağlıdır — `got`'u ölçek için kullanmak,
      `got`'u büyüten bir hook hatasının kendi toleransını da büyütmesine
      (dairesellik) yol açardı.
      `torch.finfo(bfloat16).eps` (2^-7, bf16'nın makine epsilonu) çarpı bu
      yerel ölçek, o mertebede BİR ULP genişliğine karşılık gelir;
      `hidden + delta` toplamının bf16'ya yuvarlanması en fazla yarım ULP
      hata katar. Ölçülen: yerel ölçek ≈20.6, atol≈0.161, gerçek hata
      ≈0.0625 (yaklaşık 2.6× pay) — bkz. bu testin no-op hook deneyi
      (rapor: p3-task-1-report.md, Fix Round 1).
      SONRAKİ OKUYUCU: bunu yuvarlak bir sayıyla DEĞİŞTİRME — bu türetim
      (bf16 eps × yönün dokunduğu boyutlara maskelenmiş referans ölçek)
      no-op hook'u AYIRT EDİYOR, ama bu ampirik doğrulamanın kapsamı
      DAR: yalnızca burada kullanılan tek yön indeksi (`direction[3] =
      1.0`, aşağıda) ve tek prompt ("Kısa bir cümle.", aşağıda) ile
      koşuldu — BOYUT ekseninde genellenmedi. `mask = delta_bf16 != 0`
      (aşağıda) yalnızca yönün DOKUNMADIĞI aykırı boyutları dışarıda
      bırakır; yönün KENDİ boyutu (burada dim 3) bir "kütlesel
      aktivasyon" boyutu olsaydı bu maskeleme hiçbir koruma sağlamazdı.
      UYARI: testi başka bir yön indeksi üzerinden parametrize eden
      okuyucu, o boyutun kendisi kütlesel-aktivasyon boyutu OLABİLECEĞİ
      için ayırt etme özelliğini YENİDEN doğrulamak zorundadır —
      `atol`'un beklenen delta'ya (~13.7) yaklaşması bu başarısızlığın
      işaretidir.
    """
    import torch

    from aax.model import load_hf_model
    from aax.steering import steer, steering_delta

    bundle = load_hf_model()
    tok = bundle.tokenizer
    enc = tok("Kısa bir cümle.", return_tensors="pt").to(bundle.model.device)
    L = bundle.middle_layer
    direction = np.zeros(bundle.d_model, dtype=np.float32)
    direction[3] = 1.0
    delta = steering_delta(direction, strength=0.1, layer_norm=137.0)
    # `steer()`'in GERÇEKTE kurduğu delta ile AYNI yuvarlama: float64 ->
    # modelin dtype'ı (bf16). Karşılaştırmadan idealize float64 hatasını
    # çıkarır.
    delta_bf16 = torch.tensor(delta, dtype=bundle.model.dtype, device=bundle.model.device)

    with torch.no_grad():
        base = bundle.model(**enc, output_hidden_states=True).hidden_states
    with steer(bundle, layer=L, direction=direction, strength=0.1,
               layer_norm=137.0), torch.no_grad():
        got = bundle.model(**enc, output_hidden_states=True).hidden_states

    # Fark bf16'nın KENDİSİNDE hesaplanır — erken float32'ye cast'lenmez.
    diff_bf16 = got[L + 1] - base[L + 1]
    expected_bf16 = delta_bf16.broadcast_to(diff_bf16.shape)

    # Tolerans SADECE `direction`'ın sıfır-olmadığı boyut(lar)dan türetilir
    # — tüm tensörün maksimumundan DEĞİL (yukarıdaki docstring'e bakınız:
    # bu modelde global maksimum, steering'le ilgisiz bir BOS-token aykırı
    # değeriydi ve toleransı ayırt-edemez hâle getiriyordu). Ölçek, `got`
    # DEĞİL `base + beklenen_delta`'dan gelir — dairesellikten kaçınmak
    # için (bkz. docstring).
    mask = delta_bf16 != 0
    reference_scale = (base[L + 1] + expected_bf16)[..., mask].abs().max().item()
    atol = torch.finfo(torch.bfloat16).eps * reference_scale

    assert torch.allclose(diff_bf16.float(), expected_bf16.float(), atol=atol), (
        "hedef katman deltası beklenenden farklı"
    )
    assert torch.allclose(got[L], base[L]), "steering'den ÖNCEKİ katman değişmemeli"


@pytest.mark.gpu
def test_hook_is_removed_on_exit_and_on_exception():
    import torch

    from aax.model import load_hf_model
    from aax.steering import steer

    bundle = load_hf_model()
    tok = bundle.tokenizer
    enc = tok("Test.", return_tensors="pt").to(bundle.model.device)
    direction = np.zeros(bundle.d_model, dtype=np.float32)
    direction[0] = 1.0

    with torch.no_grad():
        base = bundle.model(**enc).logits

    with pytest.raises(RuntimeError, match="bilerek"):
        with steer(bundle, layer=bundle.middle_layer, direction=direction,
                   strength=0.5, layer_norm=137.0):
            raise RuntimeError("bilerek fırlatıldı")

    with torch.no_grad():
        after = bundle.model(**enc).logits
    assert torch.equal(base, after), "istisna sonrası hook sızmış"
