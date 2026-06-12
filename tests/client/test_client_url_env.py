from reflexio.client.client import ReflexioClient


def test_client_reads_reflexio_url(monkeypatch):
    monkeypatch.delenv("REFLEXIO_API_URL", raising=False)
    monkeypatch.setenv("REFLEXIO_URL", "http://example.test:9999")
    c = ReflexioClient()
    assert "example.test:9999" in c.base_url
