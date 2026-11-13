"""Azure v1 resource compilation precedes dispatch and request-contract binding."""

from unittest.mock import Mock

import pytest

from reflexio.models.config_schema import APIKeyConfig, AzureOpenAIConfig, OpenAIConfig
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig


def client(endpoint, version="v1"):
    return LiteLLMClient(
        LiteLLMConfig(
            model="azure/gpt-4o-mini",
            api_key_config=APIKeyConfig(
                openai=OpenAIConfig(
                    azure_config=AzureOpenAIConfig(
                        api_key="synthetic-azure-key",
                        endpoint=endpoint,
                        api_version=version,
                    )
                )
            ),
            max_tokens=32,
            fallback_models=[],
        )
    )


@pytest.mark.parametrize("version", ["v1", "preview", "latest"])
@pytest.mark.parametrize("suffix", ["/", "/openai/v1/", "/resource/openai/v1/"])
@pytest.mark.unit
def test_v1_contract_contains_the_resource_base_exactly_once(version, suffix):
    selected = client("https://resource.example" + suffix, version)
    contract = selected.get_completion_request_contract()
    expected = "https://resource.example" + (
        "/resource" if suffix.startswith("/resource") else ""
    )
    assert contract["api_base"] == expected
    assert contract["api_version"] == version
    assert contract["model"] == "azure/gpt-4o-mini"
    keys = selected.config.api_key_config
    assert keys is not None and keys.openai is not None
    assert keys.openai.azure_config is not None
    assert str(keys.openai.azure_config.endpoint) == "https://resource.example" + suffix


@pytest.mark.unit
def test_selected_caller_endpoint_and_explicit_transport_own_normalization():
    selected = client("https://configured.example/")
    for key in ("api_base", "base_url"):
        contract = selected.get_completion_request_contract(
            **{key: "https://caller.example/prefix/openai/v1/"}
        )
        assert contract[key] == "https://caller.example/prefix"
    contract = selected.get_completion_request_contract(
        api_base="https://opaque.example/openai/v1/?route=unchanged",
        custom_llm_provider="openai",
    )
    assert contract["api_base"] == "https://opaque.example/openai/v1/?route=unchanged"
    assert contract["custom_llm_provider"] == "openai"


@pytest.mark.parametrize("suffix", ["/?route=private", "/#private-fragment"])
@pytest.mark.unit
def test_ambiguous_v1_url_refuses_before_dispatch_or_fallback(monkeypatch, suffix):
    selected = client("https://resource.example" + suffix)
    selected.config.fallback_models = ["gpt-4o-mini"]
    dispatch = Mock()
    monkeypatch.setattr(selected, "_completion_with_hard_timeout", dispatch)
    with pytest.raises(ValueError, match="Azure v1") as error:
        selected.generate_response("synthetic request")
    assert "private" not in str(error.value)
    dispatch.assert_not_called()


@pytest.mark.unit
def test_dated_transport_endpoint_remains_unmodified():
    endpoint = "https://resource.example/prefix/"
    contract = client(endpoint, "2024-02-15-preview").get_completion_request_contract()
    assert contract["api_base"] == endpoint
    assert contract["api_version"] == "2024-02-15-preview"


@pytest.mark.unit
def test_selected_base_url_does_not_validate_a_shadowed_configured_endpoint():
    selected = client("https://configured.example/?route=private")
    contract = selected.get_completion_request_contract(
        base_url="https://caller.example/openai/v1/"
    )
    assert contract["base_url"] == "https://caller.example"
    assert contract["api_base"] == "https://configured.example/?route=private"


@pytest.mark.unit
def test_equal_endpoint_declarations_stay_equal_after_compilation():
    endpoint = "https://configured.example/openai/v1/"
    contract = client(endpoint).get_completion_request_contract(base_url=endpoint)
    assert contract["api_base"] == contract["base_url"] == "https://configured.example"


@pytest.mark.unit
def test_budgeted_request_still_refuses_conflicting_endpoint_declarations(monkeypatch):
    from reflexio.server.llm.input_budget import (
        InputBudgetConfigurationError,
        model_input_budget,
    )

    selected = client("https://configured.example/?route=private")
    dispatch = Mock()
    monkeypatch.setattr(selected, "_completion_with_hard_timeout", dispatch)
    with (
        model_input_budget(),
        pytest.raises(InputBudgetConfigurationError, match="Conflicting"),
    ):
        selected.generate_response(
            "synthetic", base_url="https://caller.example/openai/v1/"
        )
    dispatch.assert_not_called()


@pytest.mark.parametrize("suffix", ["/?route=private", "/#private-fragment"])
@pytest.mark.unit
def test_ambiguous_embedding_endpoint_refuses_before_compute(monkeypatch, suffix):
    from reflexio.server.llm import _litellm_embedding
    from reflexio.server.llm.litellm_client import LiteLLMClientError

    selected = client("https://resource.example" + suffix)
    dispatch = Mock()
    admission = Mock()
    monkeypatch.setattr(_litellm_embedding.litellm, "embedding", dispatch)
    monkeypatch.setattr(_litellm_embedding, "admit_external_compute", admission)
    with pytest.raises(LiteLLMClientError, match="Azure v1") as error:
        selected.get_embeddings(["synthetic"], model="azure/text-embedding-3-small")
    assert "private" not in str(error.value)
    dispatch.assert_not_called()
    admission.assert_not_called()


def _owned_wire_probe():
    """Run in a clean interpreter so LiteLLM's pytest mock cannot satisfy it."""
    import json
    import os
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    assert "BRAINTRUST_API_KEY" not in os.environ
    assert "REFLEXIO_TEST_ENV_CANARY" not in os.environ
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            # self.path normalizes a leading //; requestline preserves the wire.
            target = self.requestline.split()[1]
            calls.append(
                (target, body["model"], bool(self.headers.get("Authorization")))
            )
            if "embeddings" in target:
                payload = {
                    "object": "list",
                    "model": "served-embedding",
                    "data": [
                        {
                            "object": "embedding",
                            "index": i,
                            "embedding": [0.1, 0.2, 0.3],
                        }
                        for i, _ in enumerate(body["input"])
                    ],
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                }
            else:
                payload = {
                    "id": "owned-chat",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "served-chat",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "owned-ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                }
            encoded = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *_args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            for version, suffix in [
                ("2024-02-15-preview", "/"),
                ("v1", "/"),
                ("v1", "/openai/v1/"),
                ("v1", "/resource"),
                ("preview", "/resource/openai/v1/"),
                ("latest", "/resource/"),
            ]:
                selected = client(
                    f"http://127.0.0.1:{server.server_port}{suffix}", version
                )
                selected.config.timeout = 10
                assert selected.generate_response("synthetic request") == "owned-ok"
                assert selected.get_embedding(
                    "synthetic input",
                    model="azure/text-embedding-3-small",
                    dimensions=3,
                ) == [0.1, 0.2, 0.3]
                assert (
                    selected.get_embeddings(
                        ["first", "second"],
                        model="azure/text-embedding-3-small",
                        dimensions=3,
                    )
                    == [[0.1, 0.2, 0.3]] * 2
                )
                prefix = "/resource" if suffix.startswith("/resource") else ""
                for target, model, authenticated in calls[-3:]:
                    operation = (
                        "embeddings" if "embedding" in model else "chat/completions"
                    )
                    if version == "2024-02-15-preview":
                        assert (
                            target
                            == f"/openai/deployments/{model}/{operation}?api-version={version}"
                        )
                    else:
                        assert target == f"{prefix}/openai/v1/{operation}"
                    assert model in {"gpt-4o-mini", "text-embedding-3-small"}
                    assert authenticated
            endpoint = f"http://127.0.0.1:{server.server_port}"
            assert (
                selected.generate_response(
                    "caller override", api_base=endpoint + "/caller/openai/v1/"
                )
                == "owned-ok"
            )
            assert calls[-1][0] == "/caller/openai/v1/chat/completions"
            assert (
                selected.generate_response(
                    "explicit HTTP transport",
                    custom_llm_provider="openai",
                    api_base=endpoint + "/opaque/openai/v1/",
                )
                == "owned-ok"
            )
            assert calls[-1][:2] == (
                "/opaque/openai/v1/chat/completions",
                "azure/gpt-4o-mini",
            )
            assert (
                selected.generate_response(
                    "explicit Azure transport",
                    model="gpt-4o-mini",
                    custom_llm_provider="azure",
                    api_version="v1",
                    api_key="synthetic-azure-key",
                    api_base=endpoint + "/explicit/openai/v1/",
                )
                == "owned-ok"
            )
            assert calls[-1][:2] == (
                "/explicit/openai/v1/chat/completions",
                "gpt-4o-mini",
            )
            shadowed = client(endpoint + "/unused/?route=private")
            assert (
                shadowed.generate_response(
                    "selected URL", base_url=endpoint + "/selected/openai/v1/"
                )
                == "owned-ok"
            )
            assert calls[-1][0] == "/selected/openai/v1/chat/completions"
            assert len(calls) == 22
            print("OWNED_AZURE_WIRE_OK")
        finally:
            server.shutdown()
            worker.join(timeout=2)


@pytest.mark.integration
def test_unpatched_chat_and_single_batch_embedding_wire(tmp_path, monkeypatch):
    import os
    import subprocess
    import sys
    from pathlib import Path

    monkeypatch.setenv("BRAINTRUST_API_KEY", "synthetic-parent-telemetry-canary")
    monkeypatch.setenv("REFLEXIO_TEST_ENV_CANARY", "parent-only")
    env = {
        key: os.environ[key]
        for key in (
            "PATH",
            "TMPDIR",
            "TMP",
            "TEMP",
            "SystemRoot",
            "SYSTEMROOT",
            "WINDIR",
            "COMSPEC",
            "PATHEXT",
            "LANG",
            "LC_ALL",
        )
        if key in os.environ
    }
    home = tmp_path / "home"
    home.mkdir()
    empty = tmp_path / "empty.env"
    empty.touch()
    env.update(
        HOME=str(home),
        USERPROFILE=str(home),
        APPDATA=str(home / "appdata"),
        LOCALAPPDATA=str(home / "localappdata"),
        REFLEXIO_ENV_FILE=str(empty),
        PYTHON_DOTENV_DISABLED="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[3]),
        LITELLM_LOCAL_MODEL_COST_MAP="true",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        RAYON_NUM_THREADS="1",
    )
    result = subprocess.run(  # noqa: S603 — fixed interpreter and owned test script.
        [sys.executable, str(Path(__file__).resolve())],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert "OWNED_AZURE_WIRE_OK" in result.stdout


@pytest.mark.parametrize("prefix", ["", "/resource"])
@pytest.mark.unit
def test_qualification_identity_uses_the_same_compiled_resource(prefix):
    from reflexio.server.services.offline_tuner.open_world.analyst import (
        _litellm_transport_identity,
    )

    root = client("https://resource.example" + prefix + "/")
    full = client("https://resource.example" + prefix + "/openai/v1/")
    assert (
        root.get_completion_request_contract() == full.get_completion_request_contract()
    )
    assert (
        _litellm_transport_identity(root).provider
        == _litellm_transport_identity(full).provider
    )
    other = client("https://other-resource.example" + prefix + "/")
    assert (
        _litellm_transport_identity(root).provider
        != _litellm_transport_identity(other).provider
    )


if __name__ == "__main__":
    _owned_wire_probe()
