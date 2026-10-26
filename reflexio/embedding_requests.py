"""Shared client request capacity for embedding HTTP callers."""

import logging
import os

_LOGGER = logging.getLogger(__name__)
_ENV_MAX_TEXTS_PER_REQUEST = "REFLEXIO_EMBEDDING_SERVICE_MAX_TEXTS_PER_REQUEST"
_DEFAULT_MAX_TEXTS_PER_REQUEST = 32


def embedding_request_max_texts() -> int:
    """Resolve the existing positive per-request input limit; blank means unset.

    This bounds HTTP request size, independently of server encoder batching and
    model microbatch capacity. Invalid values retain the historical default.
    """
    raw = os.environ.get(_ENV_MAX_TEXTS_PER_REQUEST, "").strip()
    if not raw:
        return _DEFAULT_MAX_TEXTS_PER_REQUEST
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value < 1:
        _LOGGER.warning(
            "%s must be a positive integer; using %d",
            _ENV_MAX_TEXTS_PER_REQUEST,
            _DEFAULT_MAX_TEXTS_PER_REQUEST,
        )
        return _DEFAULT_MAX_TEXTS_PER_REQUEST
    return value
