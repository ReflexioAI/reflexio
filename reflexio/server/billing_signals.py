"""OSS-pure billing signal helpers (no reflexio_ext imports).

Single source of truth for whether the platform supplies the LLM, so the OSS
generation service and the enterprise attribution resolver never diverge.
"""

from __future__ import annotations

from typing import Any


def platform_llm_from_config(config: Any) -> bool:
    """Return True iff Reflexio (not the customer) supplies the LLM for ``config``.

    A populated ``api_key_config`` provider sub-config means BYO-LLM → False.
    A missing/empty ``api_key_config`` means platform-supplied → True.

    Args:
        config: The org's resolved ``Config`` object, or None.

    Returns:
        bool: True when the platform supplies the LLM; False when the customer
            has configured a BYO provider key.
    """
    api_key_config = getattr(config, "api_key_config", None)
    if api_key_config is None:
        return True
    data = api_key_config.model_dump(exclude_none=True) if hasattr(api_key_config, "model_dump") else {}
    return not any(bool(v) for v in data.values())
