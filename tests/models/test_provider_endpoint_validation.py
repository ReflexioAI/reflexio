"""Provider configuration crosses the same URL-admission interface."""

import socket

import pytest
from pydantic import ValidationError

from reflexio.models.config_schema import APIKeyConfig, DashScopeConfig
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def controlled_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    def resolve(host, port, *args, **kwargs):
        address = (
            "169.254.169.254" if host == "metadata-alias.example" else "93.184.216.34"
        )
        return [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                (address, port or 0),
            )
        ]

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    monkeypatch.delenv("REFLEXIO_BLOCK_PRIVATE_URLS", raising=False)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize(
    "endpoint",
    [
        "http://169.254.169.254/latest/meta-data",
        "http://[fd00:ec2::254]/latest/meta-data",
        "http://metadata.google.internal/computeMetadata/v1",
        "http://metadata-alias.example/latest/meta-data",
        "file:///etc/passwd",
        "ftp://public.example/api",
    ],
)
def test_dashscope_rejects_unsafe_endpoints_before_dispatch(
    endpoint: str, nested: bool
) -> None:
    raw = {"api_key": "synthetic-provider-key", "api_base": endpoint}
    with pytest.raises(ValidationError):
        if nested:
            APIKeyConfig.model_validate({"dashscope": raw})
        else:
            DashScopeConfig.model_validate(raw)


@pytest.mark.parametrize("endpoint", ["http://127.0.0.1:8080/v1", "http://10.1.2.3/v1"])
def test_dashscope_retains_existing_private_url_policy(
    endpoint: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert DashScopeConfig(api_key="synthetic", api_base=endpoint).api_base == endpoint
    monkeypatch.setenv("REFLEXIO_BLOCK_PRIVATE_URLS", "true")
    with pytest.raises(ValidationError):
        DashScopeConfig(api_key="synthetic", api_base=endpoint)


@pytest.mark.parametrize(
    "endpoint",
    [
        None,
        "",
        "https://dashscope-intl.aliyuncs.com/v1",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ],
)
def test_dashscope_default_and_regional_endpoint_survive_save_and_dispatch(
    endpoint: str | None,
) -> None:
    config = APIKeyConfig(
        dashscope=DashScopeConfig(api_key="synthetic-provider-key", api_base=endpoint)
    )
    restored = APIKeyConfig.model_validate_json(config.model_dump_json())
    client = LiteLLMClient(
        LiteLLMConfig(model="dashscope/qwen-turbo", api_key_config=restored)
    )
    key, target, version = client._resolve_api_key()
    assert key == "synthetic-provider-key"
    assert target == (endpoint or None)
    assert target is None or isinstance(target, str)
    assert version is None
