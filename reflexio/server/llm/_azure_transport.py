"""Compile explicit Azure v1 endpoints before admission and native dispatch."""

from typing import Any
from urllib.parse import urlsplit, urlunsplit


def normalize_azure_v1_endpoints(params: dict[str, Any]) -> None:
    """Give LiteLLM a resource base; it appends the v1 route itself.

    These three names select LiteLLM's OpenAI-client transport, not its dated
    deployment-path transport. Explicit provider overrides own interpretation.
    No endpoint is guessed from environment or another provider's credentials.
    """
    provider = params.get("custom_llm_provider")
    azure = (
        provider == "azure"
        if provider
        else str(params.get("model", "")).startswith("azure/")
    )
    if not azure or params.get("api_version") not in {"v1", "latest", "preview"}:
        return
    # LiteLLM gives a supplied base_url precedence, even over api_base. An
    # unused declaration must not block its selected route. The model-input
    # budget still owns its independent conflicting-declaration guard.
    key = "base_url" if params.get("base_url") is not None else "api_base"
    if not params.get(key):
        return
    value = str(params[key])
    message = "Azure v1 requires an absolute HTTP(S) endpoint without query or fragment"
    try:
        url = urlsplit(value)
    except ValueError:
        raise ValueError(message) from None
    if (
        url.scheme not in {"http", "https"}
        or not url.netloc
        or url.query
        or url.fragment
        or any(ord(char) <= 32 for char in value)
    ):
        raise ValueError(message)
    path = url.path.rstrip("/").removesuffix("/openai/v1")
    compiled = urlunsplit((url.scheme, url.netloc, path, "", ""))
    if key == "base_url" and str(params.get("api_base", "")).rstrip(
        "/"
    ) == value.rstrip("/"):
        # Equal declarations must not become a false conflict by compiling
        # only one spelling. Genuinely different declarations remain distinct.
        params["api_base"] = compiled
    params[key] = compiled
