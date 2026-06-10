"""OSS-pure billing signal helpers (no reflexio_ext imports).

Single source of truth for whether the platform supplies the LLM, so the OSS
generation service and the enterprise attribution resolver never diverge.
Also provides the one canonical input-anchored tokenizer for the whole platform.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import tiktoken

# One fixed canonical encoding for the whole platform. Changing this is a
# price-book-affecting decision; do not make it model-dependent.
_CANONICAL_ENCODING_NAME = "cl100k_base"
_encoding = tiktoken.get_encoding(_CANONICAL_ENCODING_NAME)


def count_input_tokens(text: str) -> int:
    """Count billable input tokens for ``text`` under the canonical cl100k_base encoding.

    This is the single canonical tokenizer for the whole platform. Enterprise
    code delegates here so OSS and enterprise always use the same encoding.

    Args:
        text (str): The input text to tokenize.

    Returns:
        int: Number of tokens under cl100k_base, or 0 for empty input.
    """
    if not text:
        return 0
    return len(_encoding.encode(text, disallowed_special=()))


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
    data = (
        api_key_config.model_dump(exclude_none=True)
        if hasattr(api_key_config, "model_dump")
        else {}
    )
    # Guard against a model_dump that returns a non-mapping: without this,
    # `.values()` would raise. Default to platform-supplied (True) on a shape
    # we don't understand rather than misclassifying the org as BYO-LLM.
    if not isinstance(data, Mapping):
        return True
    return not any(bool(v) for v in data.values())
