"""The truncation fingerprint tells thinking modes apart, and nothing else new.

A thinking-disabled resend is a different request with its own truncation
budget. A request that chooses no thinking mode keeps the fingerprint its
saved checkpoint counts were written under, so a deploy does not reset them.
"""

import threading

from reflexio.server.llm._model_compat import with_thinking_disabled
from reflexio.server.services.extraction.overflow_coordinator import (
    MAX_TRUNCATED_REQUEST_DISPATCHES,
    OverflowCoordinator,
)

_FIELDS = (
    "model",
    "messages",
    "tools",
    "tool_choice",
    "response_format",
    "max_tokens",
    "max_completion_tokens",
    "api_base",
    "base_url",
)
REQUEST = {
    "model": "minimax/MiniMax-M3",
    "messages": [{"role": "user", "content": "extract"}],
    "max_tokens": 8192,
    "extra_headers": {"Idempotency-Key": "k"},
}


def _fingerprint(params: dict) -> str:
    return OverflowCoordinator._request_fingerprint(OverflowCoordinator, params)  # type: ignore[arg-type]


def _coordinator(checkpoint: dict) -> OverflowCoordinator:
    coordinator = object.__new__(OverflowCoordinator)
    coordinator._lock = threading.RLock()
    coordinator._manifest = {"checkpoint": checkpoint}
    return coordinator


def test_default_request_keeps_its_saved_fingerprint():
    legacy = OverflowCoordinator.candidate_key(
        {key: REQUEST.get(key) for key in _FIELDS}
    )
    assert _fingerprint(REQUEST) == legacy
    # Unrelated provider options still do not make a request "changed".
    assert _fingerprint({**REQUEST, "extra_body": {"reasoning_split": True}}) == legacy


def test_thinking_mode_is_part_of_the_request():
    disabled = {**REQUEST, "extra_body": with_thinking_disabled(None)}
    assert _fingerprint(disabled) != _fingerprint(REQUEST)
    assert _fingerprint(
        {**REQUEST, "extra_body": with_thinking_disabled({"reasoning_split": True})}
    ) == _fingerprint(disabled)
    assert _fingerprint({**REQUEST, "thinking": {"type": "enabled"}}) != _fingerprint(
        REQUEST
    )


def test_recorded_or_capped_request_goes_out_thinking_disabled():
    fingerprint = _fingerprint(REQUEST)
    assert not _coordinator({}).reasoning_exhausted(REQUEST)
    assert _coordinator(
        {"reasoning_exhausted_requests": [fingerprint]}
    ).reasoning_exhausted(REQUEST)
    # Held before the record existed: refused thinking-on, never tried off.
    assert _coordinator(
        {"truncated_requests": {fingerprint: MAX_TRUNCATED_REQUEST_DISPATCHES}}
    ).reasoning_exhausted(REQUEST)
    assert not _coordinator(
        {"truncated_requests": {fingerprint: MAX_TRUNCATED_REQUEST_DISPATCHES - 1}}
    ).reasoning_exhausted(REQUEST)
