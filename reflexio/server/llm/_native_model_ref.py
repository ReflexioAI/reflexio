"""One operator-owned native target, distinct from a requested role model."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

from reflexio.server.env_utils import env_str, env_truthy
from reflexio.server.llm._litellm_types import ProviderRequestGuardError

PROVIDER_KEY = "local-cli"
MODEL_PREFIX = f"{PROVIDER_KEY}/"
ENV_NATIVE_MODEL = "REFLEXIO_NATIVE_CLI_MODEL"
_HOSTS = frozenset({"claude-code", "codex", "opencode"})


class NativeCLISelectionError(ProviderRequestGuardError):
    """Terminal native selection/adoption failure, containing no input values."""


@dataclass(frozen=True)
class NativeModelRef:
    host: str
    model: str | None = None

    @property
    def reference(self) -> str:
        suffix = "" if self.model is None else f"/{self.model}"
        return f"{MODEL_PREFIX}{self.host}{suffix}"


@dataclass
class _NativeRequestAuthority:
    # The same request-local object is copied into async worker contexts. The
    # rejection survives a third-party exception wrapper without parsing text
    # or inspecting arbitrary exception chains.
    error: NativeCLISelectionError | None = None

    def raise_if_rejected(self) -> None:
        if self.error is not None:
            raise self.error


_authority: ContextVar[_NativeRequestAuthority | None] = ContextVar(
    "native_request_authority", default=None
)


@contextmanager
def native_request_authority() -> Iterator[_NativeRequestAuthority]:
    """Own the terminal rejection record for one physical provider attempt."""
    state = _NativeRequestAuthority()
    token = _authority.set(state)
    try:
        yield state
    finally:
        _authority.reset(token)


def reject_native_selection(message: str) -> NoReturn:
    error = NativeCLISelectionError(message)
    state = _authority.get()
    if state is not None:
        if state.error is None:
            state.error = error
        state.raise_if_rejected()
    raise error


def parse_native_model_ref(value: str) -> NativeModelRef:
    """Keep an opaque model remainder; an omitted model delegates to the CLI."""
    if not isinstance(value, str) or not value.startswith(MODEL_PREFIX):
        reject_native_selection("Native model requires a local-cli host reference")
    host, separator, model = value[len(MODEL_PREFIX) :].partition("/")
    if host not in _HOSTS or "\x00" in value:
        reject_native_selection(
            "Native model reference has an unsupported host or invalid content"
        )
    if separator and not model.strip():
        reject_native_selection("Native model reference has an empty explicit model")
    return NativeModelRef(host, model if separator else None)


def configured_native_model() -> NativeModelRef | None:
    """Read operator intent; executable presence never grants that intent."""
    value = os.environ.get(ENV_NATIVE_MODEL, "") if env_str(ENV_NATIVE_MODEL) else ""
    # Old suffixes were logical role labels, never physical model IDs. Refuse
    # ambiguous adoption instead of guessing a host/model or selecting HTTP.
    old_enable = env_str("CLAUDE_SMART_USE_LOCAL_CLI")
    if (not value and env_truthy(old_enable)) or (
        value and (old_enable or env_str("CLAUDE_SMART_CLI_MODEL"))
    ):
        reject_native_selection(
            "Native model adoption requires REFLEXIO_NATIVE_CLI_MODEL and removal of "
            "CLAUDE_SMART_USE_LOCAL_CLI / CLAUDE_SMART_CLI_MODEL backend selectors"
        )
    return parse_native_model_ref(value) if value else None


def authorize_native_model(
    model: str, custom_llm_provider: str | None = None
) -> NativeModelRef | None:
    """Validate reserved intent before role/gateway resolution can erase it."""
    state = _authority.get()
    if state is not None:
        state.raise_if_rejected()
    prefixed = model.startswith(MODEL_PREFIX)
    if prefixed and custom_llm_provider not in {None, PROVIDER_KEY}:
        reject_native_selection("Native model conflicts with the requested transport")
    if custom_llm_provider == "claude-code" or (
        custom_llm_provider is None and model.startswith("claude-code/")
    ):
        reject_native_selection(
            "Legacy native role labels require explicit native model adoption"
        )
    if not prefixed and custom_llm_provider != PROVIDER_KEY:
        return None
    reference = model if prefixed else MODEL_PREFIX + model
    requested = parse_native_model_ref(reference)
    configured = configured_native_model()
    if configured is None or requested != configured:
        reject_native_selection("Native model is not the operator-configured target")
    return requested


def native_policy_digest() -> str:
    """Bind operator native intent and executable selection without disclosing it."""
    target = configured_native_model()
    policy = {
        "target": target.reference if target else None,
        "cwd": str(Path.cwd()),
        "execution": {
            name: os.environ.get(name)
            for name in (
                "CLAUDE_SMART_CLI_PATH",
                "CLAUDE_SMART_CLI_TIMEOUT",
                "CLAUDE_SMART_CODEX_PATH",
                "PATH",
                "PATHEXT",
            )
        },
    }
    return hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()
