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

# Complete provider initialization before importing the third-party library.
import reflexio.provider_bootstrap  # noqa: F401

# isort: split

import json
import logging
import multiprocessing
import pickle
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from multiprocessing.process import BaseProcess
from typing import Any, NoReturn
from uuid import uuid4

import litellm
from litellm.exceptions import ContextWindowExceededError

from reflexio.server.llm._litellm_types import (
    LiteLLMClientError,
    LLMHardTimeoutError,
    ProviderRequestGuardError,
    completion_metadata,
)
from reflexio.server.llm._model_compat import model_completion
from reflexio.server.llm._native_cli import (
    NativeCLIContext,
    NativeCLIOwner,
    native_cli_worker,
)
from reflexio.server.llm._native_model_ref import (
    authorize_native_model,
    native_request_authority,
)
from reflexio.server.llm._provider_concurrency import retain_current_provider_slot

logger = logging.getLogger(__name__)


def retire_completion_worker(
    process: BaseProcess,
    result_queue: multiprocessing.Queue,
    native_owner: NativeCLIOwner | None = None,
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
            try:
                process.close()
            except ValueError:
                # join() returned without recording an exit code: another
                # thread's poll (multiprocessing reaps finished children on
                # every Process.start) collected this worker's status first.
                # The worker is gone, so there is nothing left to close.
                logger.info(
                    "event=completion_worker_exit_collected_elsewhere pid=%s",
                    process.pid,
                )
            finally:
                if native_owner is not None:
                    native_owner.worker_exited()

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
class _CompletionModelSnapshot:
    """Transport plain provider fields; build deferred SDK models in the parent."""

    fields: dict[str, Any]
    hidden_params: dict[str, Any]

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        # A rich model can pickle successfully while its nested serializers
        # were built only in the child. Reconstruct through the SDK constructor
        # during unpickling, before any parent consumer sees the response.
        return _restore_completion_model, (self.fields, self.hidden_params)


def _restore_completion_model(
    fields: dict[str, Any], hidden_params: dict[str, Any]
) -> Any:
    response = litellm.ModelResponse(**fields)
    response._hidden_params = hidden_params
    return response


@dataclass
class _CompletionErrorSnapshot:
    type_name: str
    message: str
    model: str | None = None
    llm_provider: str | None = None
    input_capacity_exceeded: bool = False
    status_code: int | None = None
    hard_timeout: bool = False
    request_guard_error: bool = False


def _snapshot_completion_error(
    exc: BaseException, params: dict[str, Any]
) -> _CompletionErrorSnapshot:
    model = getattr(exc, "model", None) or params.get("model")
    llm_provider = getattr(exc, "llm_provider", None)
    status_code = getattr(exc, "status_code", None)
    return _CompletionErrorSnapshot(
        type_name=type(exc).__name__,
        hard_timeout=isinstance(exc, LLMHardTimeoutError),
        request_guard_error=isinstance(exc, ProviderRequestGuardError),
        message=str(exc),
        model=str(model) if model else None,
        llm_provider=str(llm_provider) if llm_provider else None,
        input_capacity_exceeded=isinstance(exc, ContextWindowExceededError),
        status_code=status_code if isinstance(status_code, int) else None,
    )


def _raise_completion_error(payload: _CompletionErrorSnapshot) -> NoReturn:
    """Restore typed local expiry before treating provider failures generically."""
    from reflexio.server.llm.input_budget import extraction_provider_guard

    if payload.request_guard_error:
        raise ProviderRequestGuardError(payload.message)
    if payload.hard_timeout:
        raise LLMHardTimeoutError(payload.message)
    if payload.input_capacity_exceeded and extraction_provider_guard.get() is not None:
        from reflexio.server.llm.input_budget import (
            ProviderInputOverflowError,
            extraction_request_identity,
        )

        # Only a typed upstream capacity error qualifies. Generic400,
        # provider codes/messages, timeouts and schema failures do not.
        raise ProviderInputOverflowError(extraction_request_identity.get())
    # The worker always reports errors as a picklable snapshot.
    context_parts = [f"model={payload.model}"]
    if payload.llm_provider:
        context_parts.append(f"provider={payload.llm_provider}")
    raise LiteLLMClientError(
        "litellm.completion failed in isolated worker: "
        f"{payload.type_name}: {payload.message} "
        f"({', '.join(context_parts)})",
        upstream_error_type=payload.type_name,
        upstream_status_code=payload.status_code,
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
    if isinstance(response, litellm.ModelResponse):
        hidden_params = getattr(response, "_hidden_params", {}) or {}
        return _CompletionModelSnapshot(
            fields=response.model_dump(mode="json"),
            hidden_params={
                str(key): _ensure_picklable(value)
                for key, value in hidden_params.items()
            }
            if isinstance(hidden_params, dict)
            else {},
        )
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
    """Register native name resolution and the independently opted-in OpenClaw.

    Under ``fork`` the child inherits the parent's ``custom_provider_map``. But
    uvicorn starts its workers with ``spawn``, and a spawned process inherits
    ``spawn`` as its default start method, so the completion child in a
    multi-worker server re-imports only this module and never runs the
    registration ``litellm_client`` does at import. Every ``openclaw/...`` or
    ``local-cli/...`` call then failed with "LLM Provider NOT provided".
    Both registrations are idempotent. Native registration grants no execution;
    exact operator authorization is repeated inside the handler.
    """
    from reflexio.server.llm.providers.native_cli_provider import (
        register_native_provider as register_native_cli,
    )
    from reflexio.server.llm.providers.openclaw_provider import (
        register_if_enabled as register_openclaw,
    )

    register_native_cli()
    register_openclaw()


def _litellm_completion_worker(
    params: dict[str, Any],
    result_queue: multiprocessing.Queue,
    deadline: float,
    native_context: NativeCLIContext | None = None,
) -> None:
    _reset_llm_client_state_after_fork()
    with native_request_authority() as authority:
        try:
            if deadline <= time.monotonic():
                raise LLMHardTimeoutError(
                    "LLM request deadline expired during child startup"
                )
            authorize_native_model(params["model"], params.get("custom_llm_provider"))
            _register_local_cli_providers()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise LLMHardTimeoutError(
                    "LLM request deadline expired during child startup"
                )
            # Child startup spends the same accepted-work budget as provider IO.
            params = {
                **params,
                "timeout": min(float(params.get("timeout", remaining)), remaining),
            }
            private = params.pop("reflexio_private_provider_io", False)
            native_cli_worker(native_context, result_queue, deadline)
            response = (
                _private_completion(params) if private else model_completion(params)
            )
            authority.raise_if_rejected()
            result_queue.put(("ok", _picklable_completion_result(response)))
        except BaseException as exc:
            result_queue.put(
                ("error", _snapshot_completion_error(authority.error or exc, params))
            )


def _private_completion(params: dict[str, Any]) -> Any:
    """Silence hooks in the per-call child before any private provider I/O."""
    from litellm import utils
    from litellm.litellm_core_utils import litellm_logging
    from litellm.litellm_core_utils.rules import Rules

    # These rules can enforce required request/response policy. A private
    # request cannot share raw content with them or silently bypass them.
    if litellm.pre_call_rules or litellm.post_call_rules:
        raise ProviderRequestGuardError(
            "Private provider I/O cannot run with global raw-content rules"
        )
    # LiteLLM's no-log covers ordinary outcome callbacks, but its raw pre/post
    # hooks can still invoke integrations or print request and response bodies.
    # The child serves one completion and owns these inherited settings.
    logging.disable(logging.CRITICAL)
    litellm.suppress_debug_info = True
    # Older supported LiteLLM versions also print directly under this flag.
    litellm.set_verbose = False  # pyright: ignore[reportPrivateImportUsage]
    litellm.callbacks = []
    litellm.input_callback = []
    litellm.success_callback = []
    litellm.failure_callback = []
    litellm.service_callback = []
    litellm.cache = None
    params["caching"] = False
    litellm.client_session = None
    litellm.aclient_session = None
    for transport in ("sync_transport", "async_transport"):
        setattr(litellm, transport, None)
    for diagnostic_module in (utils, litellm_logging):
        for hook in ("add_breadcrumb", "capture_exception"):
            setattr(diagnostic_module, hook, None)
    litellm._async_input_callback = []
    litellm._async_success_callback = []
    litellm._async_failure_callback = []
    finish_reasons: list[str | None] = []

    def capture_finish(original_response: Any, *_args: Any, **_kwargs: Any) -> None:
        # LiteLLM normalizes null and unfamiliar reasons to "stop". Keep
        # only the raw reason from the native OpenAI-compatible response.
        try:
            if (
                isinstance(original_response, str)
                and len(original_response) <= 1_000_000
            ):
                original_response = json.loads(original_response)
            if not isinstance(original_response, dict):
                return
            raw = original_response["choices"][0].get("finish_reason")
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            return
        finish_reasons.append(raw if raw in ("stop", "length") else None)

    params.setdefault("litellm_call_id", str(uuid4()))
    logging_obj, request = utils.function_setup(
        "completion", Rules(), datetime.now(), is_async_call=False, **params
    )
    logging_obj.post_call = capture_finish
    response = model_completion({**request, "litellm_logging_obj": logging_obj})
    if isinstance(response, litellm.ModelResponse):
        response._hidden_params["reflexio_provider_finish_reason"] = (
            finish_reasons[0] if len(finish_reasons) == 1 else None
        )
    return response
