"""`tests/conftest.py`'deki ağ kilidinin gerçekten kilitlediğini doğrular.

Bu meta-testtir: koruma bozulursa (fixture kaldırılır, adı değişir, autouse
olmaktan çıkar) buradaki testler kırmızıya döner. Koruma olmadan bu dosyadaki
denemelerin bazıları paylaşımlı production sunucusuna kadar gidebilirdi.
"""
from __future__ import annotations

import socket

import pytest

from conftest import AgErisimiEngellendi  # pytest tests/ dizinini sys.path'e ekler


def test_socket_connect_engellenir():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(AgErisimiEngellendi):
            sock.connect(("example.invalid", 80))
    finally:
        sock.close()


def test_socket_connect_ex_engellenir():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(AgErisimiEngellendi):
            sock.connect_ex(("example.invalid", 80))
    finally:
        sock.close()


def test_create_connection_engellenir():
    with pytest.raises(AgErisimiEngellendi):
        socket.create_connection(("example.invalid", 80), timeout=0.1)


def test_isim_cozumleme_engellenir():
    with pytest.raises(AgErisimiEngellendi):
        socket.getaddrinfo("gateway.invalid", 443)


def test_httpx_gercek_istek_atamaz():
    """Asıl senaryo: birileri sahte transport enjekte etmeyi unutursa."""
    httpx = pytest.importorskip("httpx")
    with httpx.Client(timeout=0.5) as client:
        with pytest.raises(BaseException) as excinfo:
            client.get("https://example.invalid/")
    # httpx bağlantı hatalarını kendi istisnalarına sarabilir; kök neden
    # her hâlükârda bizim kilidimiz olmalı.
    chain = []
    exc: BaseException | None = excinfo.value
    while exc is not None:
        chain.append(type(exc))
        exc = exc.__cause__ or exc.__context__
    assert AgErisimiEngellendi in chain, chain
