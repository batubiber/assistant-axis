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
    """Katman sayısı ve genişlik config'ten okunmalı, sabit yazılmamalı.

    Model id verilmiyor: `load_hf_model()` `config.TARGET_MODEL`'e düşer, yani
    test gerçek hedef modelde (Qwen3-1.7B) koşar. Küçük bir modelde geçip
    gerçek modelde kırılacak bir hook/geometri varsayımı bu şekilde
    yakalanamazdı.
    """
    from aax.model import load_hf_model

    bundle = load_hf_model()
    assert bundle.n_layers == bundle.model.config.num_hidden_layers
    assert bundle.d_model == bundle.model.config.hidden_size
    assert bundle.middle_layer == bundle.n_layers // 2
    assert len(bundle.model.model.layers) == bundle.n_layers
