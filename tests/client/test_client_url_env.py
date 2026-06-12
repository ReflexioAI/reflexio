from reflexio.client.client import ReflexioClient


def test_client_reads_reflexio_url(monkeypatch):
    monkeypatch.delenv("REFLEXIO_API_URL", raising=False)
    monkeypatch.setenv("REFLEXIO_URL", "http://example.test:9999")
    c = ReflexioClient()
    assert "example.test:9999" in c.base_url


def test_client_ignores_legacy_reflexio_api_url(monkeypatch):
    """The legacy ``REFLEXIO_API_URL`` name is no longer honored.

    Only ``REFLEXIO_URL`` is read; with it unset the client must fall back to
    ``BACKEND_URL`` and never pick up the legacy value.
    """
    from reflexio.client.client import BACKEND_URL

    monkeypatch.delenv("REFLEXIO_URL", raising=False)
    monkeypatch.setenv("REFLEXIO_API_URL", "http://legacy.test:1234")
    c = ReflexioClient()
    assert "legacy.test:1234" not in c.base_url
    assert c.base_url == BACKEND_URL
