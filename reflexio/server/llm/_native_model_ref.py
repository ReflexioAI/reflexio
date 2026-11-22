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
CLI_TIMEOUT_ENV = "CLAUDE_SMART_CLI_TIMEOUT"
# Process-owned launch metadata from the packaged plugin, bound to its exact path.
PLUGIN_OPENCODE_EXECUTABLE_ENV = "_CLAUDE_SMART_NATIVE_OPENCODE_EXECUTABLE"
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
    """Bind native intent and launch context without reading account contents."""
    target = configured_native_model()
    policy = {
        "target": target.reference if target else None,
        "cwd": str(Path.cwd()),
        "home": str(Path.home()),
        "execution": {
            name: os.environ.get(name)
            for name in (
                "CLAUDE_SMART_CLI_PATH",
                PLUGIN_OPENCODE_EXECUTABLE_ENV,
                CLI_TIMEOUT_ENV,
                "CLAUDE_SMART_CODEX_PATH",
                "CODEX_HOME",
                "CLAUDE_CONFIG_DIR",
                "PATH",
                "PATHEXT",
            )
        },
    }
    return hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()


# Executable override and account-directory variables that decide which CLI
# binary answers and under which account.
_EXECUTABLE_OVERRIDE_ENVS = (
    "CLAUDE_SMART_CLI_PATH",
    PLUGIN_OPENCODE_EXECUTABLE_ENV,
    "CLAUDE_SMART_CODEX_PATH",
)
_ACCOUNT_DIR_ENVS = ("CLAUDE_CONFIG_DIR", "CODEX_HOME")


def native_launch_identity_digest() -> str:
    """Digest only what decides WHO answers a native call: binary and account.

    ``native_policy_digest`` binds the whole launch environment (cwd, PATH,
    PATHEXT, timeout, ...) and suits a plan frozen for one job. A DURABLE
    identity cannot: it must survive a plugin upgrade that moves the working
    directory and a restart from another shell whose PATH differs but resolves
    to the same binary. So this digest binds the configured native target, the
    executable the native provider would launch (resolved by the provider's
    own resolver; a PATH lookup contributes its absolute result, never PATH
    itself), the account/config directories (HOME, ``CLAUDE_CONFIG_DIR``,
    ``CODEX_HOME``) and the executable override variables -- and nothing else.

    Returns:
        str: Hex SHA-256 of the launch identity.
    """
    from reflexio.server.llm.providers.native_cli_provider import (
        _resolve_cli_path,  # noqa: PLC2701 - the one resolver dispatch uses
    )

    target = configured_native_model()
    executable = _resolve_cli_path(target.host) if target is not None else None
    identity = {
        "target": target.reference if target is not None else None,
        # Absolute, but symlinks are NOT followed: the launch path is the
        # identity, so an in-place CLI upgrade behind it does not move it.
        "executable": str(Path(executable).absolute()) if executable else None,
        "home": str(Path.home()),
        "accounts": {name: os.environ.get(name) for name in _ACCOUNT_DIR_ENVS},
        "overrides": {name: os.environ.get(name) for name in _EXECUTABLE_OVERRIDE_ENVS},
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
