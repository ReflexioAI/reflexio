"""Subprocess-isolation helpers for the hard-timeout completion path (Tier-2.5 leaf).

The picklable ``_Completion*Snapshot`` dataclasses plus the
snapshot builders and the ``_litellm_completion_worker`` that runs
``litellm.completion`` inside a child process. Used by ``TextGenerationMixin``'s
``_completion_with_hard_timeout`` which sizes the subprocess deadline. The parent
retires every worker through one helper; asynchronous reaping retains its existing
provider permit until process exit, without extending caller latency.

LLM-mock: the worker calls ``litellm.completion`` via the MODULE ATTR (never
``from litellm import completion``) so the global ``patch("litellm.completion")``
mock is inherited across the ``fork`` start-method. The multiprocessing
start-method is intentionally left unchanged.

SINK-2 (identity): the 6 snapshot classes and ``_litellm_completion_worker`` are
re-exported by import binding from the facade — they must be the SAME class/object
the worker constructs and tests ``isinstance``-check. Bodies moved VERBATIM.
"""

import logging
import multiprocessing
import pickle
import threading
import time
from dataclasses import dataclass, field
from multiprocessing.process import BaseProcess
from typing import Any

import litellm
from litellm.exceptions import ContextWindowExceededError

from reflexio.server.llm._litellm_types import LLMHardTimeoutError, completion_metadata
from reflexio.server.llm._model_compat import model_completion
from reflexio.server.llm._provider_concurrency import retain_current_provider_slot

logger = logging.getLogger(__name__)


def retire_completion_worker(
    process: BaseProcess, result_queue: multiprocessing.Queue
) -> None:
    """Stop a completed/abandoned call; hold admission until its worker is reaped.

    No cleanup grace extends caller latency. Only a still-live process needs an
    asynchronous reaper; it retains the existing permit, not another capacity
    budget. The caller has already drained the response or abandoned the read.
    """

    def close_handles() -> None:
        try:
            result_queue.close()
            result_queue.join_thread()
        finally:
            process.close()

    if process.pid is None:
        close_handles()
        return
    if not process.is_alive():
        process.join()
        close_handles()
        return

    release = retain_current_provider_slot()

    def reap() -> None:
        try:
            process.join()
        except Exception:
            logger.exception(
                "Completion worker could not be reaped; admission retained"
            )
            return
        try:
            close_handles()
        finally:
            release()

    try:
        process.kill()
        threading.Thread(
            target=reap, name="reflexio-completion-reaper", daemon=True
        ).start()
    except Exception:
        # Do not free a possibly live worker's capacity if cleanup cannot start.
        # If the kill already completed, no asynchronous owner is needed.
        if not process.is_alive():
            reap()
        else:
            logger.exception("Completion worker cleanup failed; admission retained")
            raise


@dataclass
class _CompletionMessageSnapshot:
    content: str | None = None
    tool_calls: Any | None = None
    reasoning_content: str | None = None
    thinking_blocks: Any | None = None
    provider_specific_fields: Any | None = None
    reasoning_items: Any | None = None


@dataclass
class _CompletionChoiceSnapshot:
    message: _CompletionMessageSnapshot
    finish_reason: str | None = None


@dataclass
class _PromptTokenDetailsSnapshot:
    cached_tokens: int = 0


@dataclass
class _CompletionTokenDetailsSnapshot:
    reasoning_tokens: int | None = None


@dataclass
class _CompletionUsageSnapshot:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    prompt_tokens_details: _PromptTokenDetailsSnapshot | None = None
    completion_tokens_details: _CompletionTokenDetailsSnapshot | None = None
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None


@dataclass
class _CompletionResponseSnapshot:
    choices: list[_CompletionChoiceSnapshot]
    usage: _CompletionUsageSnapshot | None = None
    model: str | None = None
    _hidden_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class _CompletionErrorSnapshot:
    type_name: str
    message: str
    model: str | None = None
    llm_provider: str | None = None
    input_capacity_exceeded: bool = False
    status_code: int | None = None


def _snapshot_completion_error(
    exc: BaseException, params: dict[str, Any]
) -> _CompletionErrorSnapshot:
    model = getattr(exc, "model", None) or params.get("model")
    llm_provider = getattr(exc, "llm_provider", None)
    status_code = getattr(exc, "status_code", None)
    return _CompletionErrorSnapshot(
        type_name=type(exc).__name__,
        message=str(exc),
        model=str(model) if model else None,
        llm_provider=str(llm_provider) if llm_provider else None,
        input_capacity_exceeded=isinstance(exc, ContextWindowExceededError),
        status_code=status_code if isinstance(status_code, int) else None,
    )


def _ensure_picklable(value: Any) -> Any:
    try:
        pickle.dumps(value)
    except Exception:
        return repr(value)
    return value


def _snapshot_completion_response(response: Any) -> _CompletionResponseSnapshot:
    choices: list[_CompletionChoiceSnapshot] = []
    for choice in getattr(response, "choices", []) or []:
        message = getattr(choice, "message", None)
        choices.append(
            _CompletionChoiceSnapshot(
                message=_CompletionMessageSnapshot(
                    content=getattr(message, "content", None),
                    tool_calls=_ensure_picklable(getattr(message, "tool_calls", None)),
                    **completion_metadata(message),
                ),
                finish_reason=getattr(choice, "finish_reason", None),
            )
        )

    usage = getattr(response, "usage", None)
    usage_snapshot = None
    if usage is not None:
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        prompt_details_snapshot = None
        if prompt_details is not None:
            prompt_details_snapshot = _PromptTokenDetailsSnapshot(
                cached_tokens=int(getattr(prompt_details, "cached_tokens", 0) or 0)
            )
        completion_details = getattr(usage, "completion_tokens_details", None)
        completion_details_snapshot = None
        if completion_details is not None:
            reasoning_tokens = getattr(completion_details, "reasoning_tokens", None)
            completion_details_snapshot = _CompletionTokenDetailsSnapshot(
                reasoning_tokens=reasoning_tokens
                if isinstance(reasoning_tokens, int)
                else None
            )
        usage_snapshot = _CompletionUsageSnapshot(
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
            prompt_tokens_details=prompt_details_snapshot,
            completion_tokens_details=completion_details_snapshot,
            cache_creation_input_tokens=getattr(
                usage, "cache_creation_input_tokens", None
            ),
            cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", None),
        )

    hidden_params = getattr(response, "_hidden_params", {}) or {}
    if not isinstance(hidden_params, dict):
        hidden_params = {}

    return _CompletionResponseSnapshot(
        choices=choices,
        usage=usage_snapshot,
        model=getattr(response, "model", None),
        _hidden_params={str(k): _ensure_picklable(v) for k, v in hidden_params.items()},
    )


def _picklable_completion_result(response: Any) -> Any:
    try:
        pickle.dumps(response)
    except Exception:
        return _snapshot_completion_response(response)
    return response


def _reset_llm_client_state_after_fork() -> None:
    """Drop litellm's module-level HTTP client cache in the forked CHILD.

    ``_litellm_completion_worker`` runs in a ``multiprocessing`` fork child, which
    inherits a *copy* of the parent's ``litellm.in_memory_llm_clients_cache`` —
    including any warm keep-alive HTTPS connection the parent held. Reusing such
    an inherited socket in the child shares one TLS/HTTP connection across two
    processes and corrupts the child's first completion (surfaces as
    ``[SSL] record layer failure`` / ``Server disconnected`` / a garbled request
    the provider rejects as missing auth). Clearing the cache here forces the
    child to build a fresh client (and socket) for its one completion. Best-effort
    and fully guarded: a litellm build without these attributes is a no-op, never
    a crash. Only the child's copy is mutated, so the parent cache is untouched.
    """
    try:
        cache = getattr(litellm, "in_memory_llm_clients_cache", None)
        if cache is not None:
            flush = getattr(cache, "flush_cache", None)
            if callable(flush):
                flush()
            cache_dict = getattr(cache, "cache_dict", None)
            if isinstance(cache_dict, dict):
                cache_dict.clear()
    except Exception:  # noqa: BLE001, S110 — a reset failure must never break the call
        pass


def _register_local_cli_providers() -> None:
    """Register the opt-in CLI providers in this child, if enabled.

    Under ``fork`` the child inherits the parent's ``custom_provider_map``. But
    uvicorn starts its workers with ``spawn``, and a spawned process inherits
    ``spawn`` as its default start method, so the completion child in a
    multi-worker server re-imports only this module and never runs the
    registration ``litellm_client`` does at import. Every ``openclaw/...`` or
    ``claude-code/...`` call then failed with "LLM Provider NOT provided".
    Both registrations are idempotent and no-ops unless their env opt-in is set.
    """
    from reflexio.server.llm.providers.claude_code_provider import (
        register_if_enabled as register_claude_code,
    )
    from reflexio.server.llm.providers.openclaw_provider import (
        register_if_enabled as register_openclaw,
    )

    register_claude_code()
    register_openclaw()


def _litellm_completion_worker(
    params: dict[str, Any], result_queue: multiprocessing.Queue, deadline: float
) -> None:
    _reset_llm_client_state_after_fork()
    _register_local_cli_providers()
    try:
        if time.monotonic() >= deadline:
            raise LLMHardTimeoutError(
                "LLM request deadline expired during child startup"
            )
        result_queue.put(("ok", _picklable_completion_result(model_completion(params))))
    except BaseException as exc:
        result_queue.put(("error", _snapshot_completion_error(exc, params)))
