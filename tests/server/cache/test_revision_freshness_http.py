"""Real config-file and HTTP-route refresh under uncertain revision probes."""

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from reflexio.models.config_schema import Config
from reflexio.server.auth import default_get_org_id
from reflexio.server.cache import reflexio_cache as cache
from reflexio.server.routes.config import router
from reflexio.server.services.configurator.configurator import DefaultConfigurator
from reflexio.server.services.configurator.local_file_config_storage import (
    LocalFileConfigStorage,
)

pytestmark = pytest.mark.integration


def test_http_config_reload_is_fresh_or_fails_without_replacing_originals(
    tmp_path, monkeypatch
):
    org = "cache-http-org"
    storage = LocalFileConfigStorage(org, base_dir=str(tmp_path))
    storage.save_config(Config(storage_config=None, enable_document_expansion=False))
    monkeypatch.setattr(
        DefaultConfigurator, "_select_config_storage", lambda *_: storage
    )
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[default_get_org_id] = lambda: org
    cache.clear_reflexio_cache()
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            first = client.get("/api/get_config")
            assert first.status_code == 200
            assert first.json()["enable_document_expansion"] is False
            assert cache.get_cache_stats()["current_size"] == 1
            storage.save_config(
                Config(storage_config=None, enable_document_expansion=True)
            )
            original = Path(storage.config_file).read_bytes()

            def fail_probe():
                raise OSError("synthetic metadata unavailable")

            with monkeypatch.context() as patch:
                patch.setattr(storage, "get_version", fail_probe)
                changed = client.get("/api/get_config")
                assert changed.status_code == 200
                assert changed.json()["enable_document_expansion"] is True
                assert cache.get_cache_stats()["current_size"] == 0
            recovered = client.get("/api/get_config")
            assert recovered.status_code == 200
            assert recovered.json()["enable_document_expansion"] is True
            assert cache.get_cache_stats()["current_size"] == 1

            def fail_load():
                raise OSError("synthetic config unavailable")

            with monkeypatch.context() as patch:
                patch.setattr(storage, "get_version", fail_probe)
                patch.setattr(storage, "load_config", fail_load)
                refused = client.get("/api/get_config")
                assert refused.status_code == 500
                assert cache.get_cache_stats()["current_size"] == 0
                assert Path(storage.config_file).read_bytes() == original
            final = client.get("/api/get_config")
            assert final.status_code == 200
            assert final.json()["enable_document_expansion"] is True
    finally:
        cache.clear_reflexio_cache()
