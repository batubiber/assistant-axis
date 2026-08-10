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

    class FakeBundle:
        def __init__(self):
            self.model = FakeModel()
            self.n_layers = 1

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

    with torch.no_grad():
        base = bundle.model(**enc, output_hidden_states=True).hidden_states
    with steer(bundle, layer=L, direction=direction, strength=0.1,
               layer_norm=137.0), torch.no_grad():
        got = bundle.model(**enc, output_hidden_states=True).hidden_states

    diff = (got[L + 1] - base[L + 1]).float().cpu().numpy()
    expected = np.broadcast_to(delta, diff.shape)
    assert np.allclose(diff, expected, atol=1e-1), "hedef katman deltası beklenenden farklı"
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
