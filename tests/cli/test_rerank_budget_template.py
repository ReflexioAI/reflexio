"""Qualify the actual SDK template through provisioning and read-only startup."""

import os
from pathlib import Path

import pytest

from reflexio.cli import env_loader
from reflexio.server.llm.rerank.common import reranker_budget_ms


@pytest.mark.parametrize("provision", [False, True])
def test_real_default_template_admits_rerank_budget(monkeypatch, tmp_path, provision):
    expected_template = Path(__file__).resolve().parents[2] / ".env.example"
    target = tmp_path / "home" / ".env"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("REFLEXIO_ENV_FILE", str(target))
    monkeypatch.delenv("PYTHON_DOTENV_DISABLED", raising=False)
    monkeypatch.delenv("DEPLOYMENT_MODE", raising=False)
    for name in (
        "REFLEXIO_RERANK_BUDGET_MS",
        "REFLEXIO_RERANK_ENABLED",
        "REFLEXIO_RERANK_SERVICE_TIMEOUT_MS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(env_loader, "_USER_ENV_DIR", target.parent)
    monkeypatch.setattr(env_loader, "_USER_ENV_FILE", target)
    monkeypatch.setattr(env_loader, "_env_search_paths", lambda: [target])
    # Exercise actual source/package discovery, not a mocked template string.
    assert env_loader._find_env_example("reflexio") == expected_template.read_text()
    snapshot = dict(os.environ)
    try:
        result = (
            env_loader.load_reflexio_env(auto_generate_keys=[])
            if provision
            else env_loader.load_existing_reflexio_env()
        )
        assert result == (target if provision else None)
        assert target.exists() is provision
        assert reranker_budget_ms() == 5000
        assert "REFLEXIO_RERANK_ENABLED" not in os.environ
        assert "REFLEXIO_RERANK_SERVICE_TIMEOUT_MS" not in os.environ
        if provision:
            assert "REFLEXIO_RERANK_BUDGET_MS=5000" in target.read_text()
    finally:
        os.environ.clear()
        os.environ.update(snapshot)
