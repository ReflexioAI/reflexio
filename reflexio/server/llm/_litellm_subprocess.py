"""Subprocess-isolation helpers for the hard-timeout completion path (Tier-2.5 leaf).

Stateless leaf module: the picklable ``_Completion*Snapshot`` dataclasses plus the
snapshot builders and the ``_litellm_completion_worker`` that runs
``litellm.completion`` inside a child process. Used by ``TextGenerationMixin``'s
``_completion_with_hard_timeout`` (Task 4) which sizes and kills the subprocess.

LLM-mock: the worker calls ``litellm.completion`` via the MODULE ATTR (never
``from litellm import completion``) so the global ``patch("litellm.completion")``
mock is inherited across the ``fork`` start-method. The multiprocessing
start-method is intentionally left unchanged.

SINK-2 (identity): the 6 snapshot classes and ``_litellm_completion_worker`` are
re-exported by import binding from the facade — they must be the SAME class/object
the worker constructs and tests ``isinstance``-check. Bodies moved VERBATIM.
"""

import multiprocessing
import pickle
from dataclasses import dataclass, field
from typing import Any

import litellm


@dataclass
class _CompletionMessageSnapshot:
    content: str | None = None
    tool_calls: Any | None = None


@dataclass
class _CompletionChoiceSnapshot:
    message: _CompletionMessageSnapshot
    finish_reason: str | None = None


@dataclass
class _PromptTokenDetailsSnapshot:
    cached_tokens: int = 0


@dataclass
class _CompletionUsageSnapshot:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    prompt_tokens_details: _PromptTokenDetailsSnapshot | None = None
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


def _snapshot_completion_error(
    exc: BaseException, params: dict[str, Any]
) -> _CompletionErrorSnapshot:
    model = getattr(exc, "model", None) or params.get("model")
    llm_provider = getattr(exc, "llm_provider", None)
    return _CompletionErrorSnapshot(
        type_name=type(exc).__name__,
        message=str(exc),
        model=str(model) if model else None,
        llm_provider=str(llm_provider) if llm_provider else None,
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
        usage_snapshot = _CompletionUsageSnapshot(
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
            prompt_tokens_details=prompt_details_snapshot,
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


def _litellm_completion_worker(
    params: dict[str, Any], result_queue: multiprocessing.Queue
) -> None:
    try:
        result_queue.put(
            ("ok", _picklable_completion_result(litellm.completion(**params)))
        )
    except BaseException as exc:
        result_queue.put(("error", _snapshot_completion_error(exc, params)))
