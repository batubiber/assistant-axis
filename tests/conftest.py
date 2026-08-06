"""Test paketi geneli koruma: hiçbir test gerçek soket açamaz.

Bu proje tek bir uzak uca konuşuyor: `hakem-llm`, **paylaşımlı bir production
vLLM sunucusu**. Sunucu tarafında hız sınırlama yok; koruma tamamen bu
repodaki kodda. Bugüne kadar testlerin ağa çıkmaması disipline dayanıyordu —
her test ya sahte bir transport enjekte ediyor ya da `build_default_client`'ı
monkeypatch'liyordu. Tek bir unutulmuş yama `pytest`'i canlı sunucuya
bağlardı.

Buradaki autouse fixture bunu yapıya bağlar: soket bağlantısı denemesi
`AgErisimiEngellendi` ile patlar. Bir test gerçekten ağa çıkmak isterse bunu
görünür biçimde (fixture'ı açıkça devre dışı bırakarak) yapmak zorundadır —
sessizce olamaz.
"""
from __future__ import annotations

import socket

import pytest

from aax.gateway import reset_shared_state


class AgErisimiEngellendi(RuntimeError):
    """Bir test gerçek ağ bağlantısı açmaya çalıştı.

    Bu bir test hatasıdır, bir ağ hatası değil: gateway testleri sahte
    transport enjekte etmeli, script testleri `build_default_client`'ı
    monkeypatch'lemelidir.
    """


def _reddet(*args, **kwargs):
    raise AgErisimiEngellendi(
        "Test ağa çıkmaya çalıştı. Bu repoda testler SIFIR gerçek istek atar: "
        "sahte bir transport enjekte et (GatewayClient(..., transport=...)) veya "
        "build_default_client'ı monkeypatch'le."
    )


@pytest.fixture(autouse=True)
def ag_erisimi_kapali(monkeypatch):
    """Soket bağlantısı ve DNS çözümlemesini tüm testler için kapat.

    `socket.socket` nesnesinin oluşturulması engellenmiyor (pytest'in ve
    standart kütüphanenin kendi iç kullanımlarını bozmamak için) — engellenen
    şey karşı tarafa **bağlanmak**: `connect`, `connect_ex`,
    `create_connection` ve isim çözümleme.

    `HF_HUB_OFFLINE=1` de aynı nedenle burada: cache'te olan bir model bile
    `huggingface_hub`'ın etag/güncellik kontrolü yüzünden ağa çıkmayı dener.
    O deneme yukarıdaki soket kilidine çarpıp özel bir hata fırlatıyor, ama
    `huggingface_hub` bu hatayı "bağlantı yok, cache'e düş" sinyali olarak
    tanımıyor (yalnızca `httpx.ConnectError`/`TimeoutException` bekliyor) —
    sonuç, gerçek nedeni gizleyen genel bir `OSError`. Bu değişken
    `huggingface_hub`'ı hiç denemeden doğrudan cache'e yönlendirir; soket
    kilidi olduğu gibi, ikinci bir savunma katmanı olarak duruyor.
    """
    monkeypatch.setattr(socket.socket, "connect", _reddet, raising=True)
    monkeypatch.setattr(socket.socket, "connect_ex", _reddet, raising=True)
    monkeypatch.setattr(socket, "create_connection", _reddet, raising=True)
    monkeypatch.setattr(socket, "getaddrinfo", _reddet, raising=True)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    yield


@pytest.fixture(autouse=True)
def temiz_gateway_durumu():
    """Her testi temiz paylaşılan gateway durumuyla başlat.

    Hız sınırlayıcı, semafor ve devre kesici `base_url` ile anahtarlanan modül
    düzeyinde bir kayıt defterinde (süreç genelinde paylaşılıyor). Testlerin
    çoğu aynı sahte `base_url`'i kullandığı için, devreyi açan bir test onu
    kapatmadan sonrakilere sızdırırdı.
    """
    reset_shared_state()
    yield
    reset_shared_state()
