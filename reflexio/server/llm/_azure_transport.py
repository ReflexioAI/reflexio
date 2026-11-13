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
    for key in ("api_base", "base_url"):
        if not params.get(key):
            continue
        value = str(params[key])
        message = (
            "Azure v1 requires an absolute HTTP(S) endpoint without query or fragment"
        )
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
        params[key] = urlunsplit((url.scheme, url.netloc, path, "", ""))
