"""Optional fixed-model Jev transport; native-only mode has no client or key use.

Provider admission and caller deadlines stay with the shared owners. The small
HTTP pool is separate only to preserve Jev's direct, proxy-free wire policy.
"""

from __future__ import annotations

import atexit
import gzip
import json
import math
import os
import threading
from time import monotonic
from typing import Any, Literal

import httpx

from reflexio.server import search_runtime
from reflexio.server.env_utils import env_str
from reflexio.server.llm._provider_concurrency import named_provider_slot
from reflexio.server.services.retrieval.judge_payload import MODEL
from reflexio.server.services.storage.error import ReadinessUnavailableError

# Literal selected-image metadata consumed by deployment's source renderer.
SEARCH_JUDGE_CONTRACT = {
    "mode_env": "REFLEXIO_SEARCH_JUDGE",
    "default": "off",
    "enabled": "jev",
    "required_env_secrets": ("TYPESAFE_API_KEY",),
}

_MODE_ENV = "REFLEXIO_SEARCH_JUDGE"
_KEY_ENV = "TYPESAFE_API_KEY"
_MODEL = MODEL
_URL = "https://api.typesafe.ai/v1/systemone"
_ALLOWANCE = 0.7
_UNAVAILABLE = "Search judge is unavailable"
_client_lock = threading.Lock()
_client_instance: httpx.Client | None = None
_client_pid: int | None = None


def judge_mode() -> Literal["off", "jev"]:
    """Resolve explicit activation only; credentials never activate this method."""
    mode = env_str(_MODE_ENV, "off")
    if mode == "off":
        return "off"
    if mode == "jev":
        return "jev"
    raise ReadinessUnavailableError("REFLEXIO_SEARCH_JUDGE must be off or jev")


def _after_fork() -> None:
    # Never reuse inherited HTTP locks/sockets, including if another parent
    # thread held the construction lock at fork. No client is built here.
    global _client_lock, _client_instance, _client_pid
    _client_lock = threading.Lock()
    _client_instance = None
    _client_pid = None


def _client() -> httpx.Client:
    global _client_instance, _client_pid
    with _client_lock:
        pid = os.getpid()
        if _client_pid != pid:
            stale, _client_instance = _client_instance, None
            _client_pid = pid
            if stale is not None:
                stale.close()
        if _client_instance is None:
            _client_instance = httpx.Client(
                trust_env=False,
                follow_redirects=False,
                event_hooks={"request": [search_runtime.http_request_deadline]},
                limits=httpx.Limits(keepalive_expiry=50.0),
            )
        return _client_instance


def close_judge_client() -> None:
    """Idempotent shutdown hook; never constructs a client or reads a key."""
    global _client_instance, _client_pid
    with _client_lock:
        client, _client_instance = _client_instance, None
        _client_pid = None
    if client is not None:
        try:
            client.close()
        except Exception:
            raise ReadinessUnavailableError(
                "Search judge client shutdown failed"
            ) from None


def _remaining(deadline: float) -> float:
    try:
        search_runtime.checkpoint()
        caller = search_runtime.remaining(_ALLOWANCE)
    except search_runtime.SearchDeadlineError:
        raise search_runtime.SearchDeadlineError() from None
    remaining = min(caller, deadline - monotonic())
    if remaining <= 0:
        raise search_runtime.SearchDeadlineError() from None
    return remaining


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _constant(_value: str) -> Any:
    raise ValueError("Nonfinite JSON number")


def _float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("Nonfinite JSON number")
    return parsed


def _decode(response: httpx.Response) -> dict[str, Any]:
    body = json.loads(
        response.content,
        object_pairs_hook=_object,
        parse_constant=_constant,
        parse_float=_float,
    )
    if (
        not isinstance(body, dict)
        or body.get("model") != _MODEL
        or not isinstance(body.get("answers"), dict)
    ):
        raise ValueError("Judge response envelope differs")
    return body


def call_jev(payload: dict) -> dict:
    """One direct attempt, no retries or alternate provider/native fallback.

    Complete answer keys/types/probability validation belongs to the pure core.
    The provider lease encloses encoding, HTTP and parsing, so a late response
    releases ownership only after the physical HTTP call has finished.
    """
    deadline = monotonic() + _ALLOWANCE
    if judge_mode() != "jev":
        raise ReadinessUnavailableError("Search judge is disabled")
    _remaining(deadline)
    if not isinstance(payload, dict) or payload.get("model") != _MODEL:
        raise ReadinessUnavailableError("Search judge requires model jev-1.14.0")
    key = env_str(_KEY_ENV)
    if not key:
        raise ReadinessUnavailableError("Search judge requires TYPESAFE_API_KEY")
    try:
        with named_provider_slot("typesafe", max_wait=_remaining(deadline)):
            _remaining(deadline)
            wire = httpx.Request("POST", _URL, json=payload).content
            compressed = gzip.compress(wire, compresslevel=1, mtime=0)
            _remaining(deadline)
            client = _client()
            timeout = _remaining(deadline)
            response = client.post(
                _URL,
                content=compressed,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                    "Content-Encoding": "gzip",
                },
                timeout=timeout,
                follow_redirects=False,
            )
            _remaining(deadline)
            response.raise_for_status()
            body = _decode(response)
            _remaining(deadline)
            return body
    except search_runtime.SearchDeadlineError:
        raise
    except Exception:
        _remaining(deadline)
        raise ReadinessUnavailableError(_UNAVAILABLE) from None


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)
atexit.register(close_judge_client)
