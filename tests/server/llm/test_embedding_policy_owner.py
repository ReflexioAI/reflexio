"""Embedding selection must not need a second opt-in boolean."""

import pytest

from reflexio.server.llm.model_defaults import ModelRole, resolve_model_name
from reflexio.server.llm.providers import embedding_service_provider as service
from reflexio.server.llm.providers import local_embedding_provider as local

LEGACY = "CLAUDE_SMART_USE_LOCAL_EMBEDDING"


def test_dependency_advertisement_is_not_model_activation(monkeypatch):
    monkeypatch.delenv(LEGACY, raising=False)
    monkeypatch.setattr(local.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(
        local, "register_if_available", lambda: pytest.fail("activation")
    )
    monkeypatch.setattr(
        local, "LocalEmbedder", lambda **_kwargs: pytest.fail("model load")
    )
    assert local.is_local_embedder_available()
    assert (
        resolve_model_name(ModelRole.EMBEDDING, config_override="openai/chosen")
        == "openai/chosen"
    )


@pytest.mark.parametrize("value", ["", "0", "1", "true", "invalid"])
@pytest.mark.parametrize("canonical", [None, "off", "local_service"])
def test_old_name_is_rejected_even_with_canonical_policy(monkeypatch, value, canonical):
    monkeypatch.setenv(LEGACY, value)
    if canonical:
        monkeypatch.setenv("REFLEXIO_EMBEDDING_PROVIDER", canonical)
    else:
        monkeypatch.delenv("REFLEXIO_EMBEDDING_PROVIDER", raising=False)
    with pytest.raises(ValueError, match=LEGACY):
        service.embedding_provider_mode("local/minilm-l6-v2")


@pytest.mark.parametrize("source", ["process", "file"])
@pytest.mark.parametrize("value", ["", "0", "private-value-not-to-log"])
def test_actual_loader_rejects_old_assignment_without_writing_or_leaking(
    monkeypatch, tmp_path, source, value
):
    import os
    import subprocess
    import sys

    selected = tmp_path / "selected.env"
    content = f"{LEGACY}={value}\n" if source == "file" else ""
    selected.write_text(content)
    env = os.environ.copy()
    env.pop(LEGACY, None)
    env.pop("PYTHON_DOTENV_DISABLED", None)
    env["REFLEXIO_ENV_FILE"] = str(selected)
    env["REFLEXIO_EMBEDDING_PROVIDER"] = "local_service"
    if source == "process":
        env[LEGACY] = value
    result = subprocess.run(  # noqa: S603 - fixed interpreter/code; fixture values only enter env.
        [
            sys.executable,
            "-c",
            "from reflexio.cli.env_loader import load_reflexio_env; load_reflexio_env()",
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert LEGACY in result.stderr
    assert "REFLEXIO_EMBEDDING_PROVIDER" in result.stderr
    assert "private-value-not-to-log" not in result.stderr
    assert selected.read_text() == content


@pytest.mark.parametrize("mode", ["local_service", "internal_service", "cloud", "off"])
@pytest.mark.parametrize("milliseconds", [None, "9000", "30000"])
def test_canonical_policy_controls_actual_http_route_and_budget(
    monkeypatch, mode, milliseconds
):
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    import httpx

    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, payload))
            body = json.dumps(
                {"data": [{"index": 0, "embedding": [1.0, 0.0]}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(
        target=lambda: server.serve_forever(poll_interval=0.01), daemon=True
    )
    thread.start()
    monkeypatch.delenv(LEGACY, raising=False)
    monkeypatch.setenv("REFLEXIO_EMBEDDING_PROVIDER", mode)
    monkeypatch.setenv(
        "REFLEXIO_EMBEDDING_SERVICE_URL", f"http://127.0.0.1:{server.server_port}"
    )
    monkeypatch.delenv("REFLEXIO_QUERY_EMBEDDING_SERVICE_URL", raising=False)
    if milliseconds is None:
        monkeypatch.delenv("REFLEXIO_EMBEDDING_SERVICE_TIMEOUT_MS", raising=False)
    else:
        monkeypatch.setenv("REFLEXIO_EMBEDDING_SERVICE_TIMEOUT_MS", milliseconds)
    try:
        with httpx.Client(trust_env=False) as client:
            monkeypatch.setattr(service, "_http_client", lambda: client)
            if mode == "off":
                with pytest.raises(service.EmbeddingUnavailableError, match="disabled"):
                    service.get_service_embeddings(
                        ["query: owned"], model="local/minilm-l6-v2", dimensions=None
                    )
                assert not requests
            else:
                assert service.get_service_embeddings(
                    ["query: owned"], model="local/minilm-l6-v2", dimensions=None
                ) == [[1.0, 0.0]]
                assert len(requests) == 1
                path, payload = requests[0]
                assert path == "/v1/embeddings"
                assert payload["model"] == "local/minilm-l6-v2"
                assert payload["input"] == ["query: owned"]
                assert payload["timeout_ms"] == (
                    int(milliseconds) if milliseconds else 2000
                )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


def test_embedding_default_does_not_consult_generation_credentials(monkeypatch):
    from reflexio.server.llm import model_defaults

    monkeypatch.setattr(
        model_defaults,
        "detect_available_providers",
        lambda *_args: pytest.fail("credential-dependent embedding choice"),
    )
    assert resolve_model_name(ModelRole.EMBEDDING) == "local/minilm-l6-v2"


def test_local_capability_never_grants_generation_or_cli_access(monkeypatch):
    from reflexio.server.llm import model_defaults

    for name in model_defaults.provider_env_vars():
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)
    monkeypatch.delenv("OPENCLAW_SMART_USE_LOCAL_CLI", raising=False)
    monkeypatch.setattr(
        local, "are_local_embedding_dependencies_available", lambda: True
    )
    assert model_defaults.detect_available_providers() == ["local"]
    with pytest.raises(RuntimeError, match="No generation-capable"):
        model_defaults.validate_llm_availability()


@pytest.mark.parametrize(
    "mode,attached,default_seconds",
    [
        (None, False, 2),
        ("local_service", False, 30),
        ("local_service", True, 2),
        ("internal_service", True, 2),
    ],
)
@pytest.mark.parametrize("milliseconds", [None, "9000", "30000"])
def test_unhealthy_cached_reranker_refresh_retains_its_budget(
    monkeypatch, mode, attached, default_seconds, milliseconds
):
    import os

    import httpx

    model, reranker = "local/minilm-l6-v2", "local/reranker"
    shared, query = "http://owned.invalid/shared", "http://owned.invalid/query"
    monkeypatch.delenv(LEGACY, raising=False)
    for key, value in {
        "REFLEXIO_EMBEDDING_PROVIDER": mode,
        "REFLEXIO_EMBEDDING_SERVICE_URL": shared if attached else None,
        "REFLEXIO_EMBEDDING_SERVICE_TIMEOUT_MS": milliseconds,
        "REFLEXIO_QUERY_EMBEDDING_SERVICE_URL": query,
        "REFLEXIO_SEARCH_RERANK_SERVICE_URL": query,
    }.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    monkeypatch.setattr(service, "inference_service_url", lambda: shared)
    pid = os.getpid()
    monkeypatch.setattr(
        service,
        "_configured_model_cache",
        {
            (pid, shared): service.InferenceServiceCapabilities(
                model, reranker, True, True
            ),
            (pid, query): service.InferenceServiceCapabilities(
                model, reranker, True, False, "search"
            ),
        },
    )
    monkeypatch.setattr(
        service, "_query_contract_cache", {(pid, shared, query): (model, "pinned", 2)}
    )
    requests = []

    def health(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "configured_model": model,
                "configured_reranker_model": reranker,
                "workload": "search",
                "reranker_enabled": True,
                "reranker_ready": True,
            },
        )

    with httpx.Client(transport=httpx.MockTransport(health)) as client:
        monkeypatch.setattr(service, "_http_client", lambda: client)
        assert service.validate_search_reranker_service() == query
    assert len(requests) == 1
    assert str(requests[0].url) == query + "/health"
    expected = int(milliseconds) / 1000 if milliseconds else default_seconds
    assert requests[0].extensions["timeout"]["read"] == pytest.approx(expected)
    assert service._configured_model_cache[(pid, query)].reranker_ready
