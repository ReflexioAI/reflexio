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


# Executable override variables: any of them can change which binary runs.
# The retired `CLAUDE_SMART_CODEX_PATH` is not one: it never selects a binary,
# it only refuses Codex selection until migrated (no launch, so no identity),
# and is otherwise ignored. Its effect is already bound through the resolved
# executable, so removing a stale assignment does not move the identity.
_EXECUTABLE_OVERRIDE_ENVS = (
    "CLAUDE_SMART_CLI_PATH",
    PLUGIN_OPENCODE_EXECUTABLE_ENV,
)
# Per host: the account/configuration directories the CLI reads, each with its
# documented default under HOME when unset.
_HOST_ACCOUNT_DIRS: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "claude-code": (("CLAUDE_CONFIG_DIR", (".claude",)),),
    "codex": (("CODEX_HOME", (".codex",)),),
    "opencode": (
        ("XDG_DATA_HOME", (".local", "share")),
        ("XDG_CONFIG_HOME", (".config",)),
    ),
}
# Per host: the environment namespace the CLI reads to choose its backend,
# endpoint, credential source and model -- e.g. Claude Code's
# `ANTHROPIC_BASE_URL`, `ANTHROPIC_API_KEY` (which "is used instead of your ...
# subscription" in `-p` mode), `ANTHROPIC_DEFAULT_SONNET_MODEL` and the
# `CLAUDE_CODE_USE_BEDROCK`/`_VERTEX`/`_FOUNDRY` switches. Bound by prefix
# rather than by a hand list, so a newly documented variable is covered.
_HOST_ENV_NAMESPACES: dict[str, tuple[str, ...]] = {
    "claude-code": ("ANTHROPIC_", "CLAUDE_CODE_USE_"),
    "codex": ("CODEX_", "OPENAI_"),
    "opencode": ("OPENCODE_",),
}
# Claude Code on a cloud provider (any `CLAUDE_CODE_USE_*` switch set) takes
# its account and region from that cloud's own SDK variables (e.g. an AWS
# profile or region for Bedrock). Bound only while such a switch is set, so an
# unrelated cloud setting cannot refuse a direct-API daemon.
_CLAUDE_CODE_CLOUD_SWITCH = "CLAUDE_CODE_USE_"
_CLAUDE_CODE_CLOUD_NAMESPACES = ("AWS_", "GOOGLE_", "CLOUD_ML_")
# Per host: variables outside its namespaces that still select its account.
# Claude Code's `CLAUDE_CODE_OAUTH_TOKEN` is a subscription credential (bound
# by presence through the secret rule below, never by value).
_HOST_ENV_NAMES: dict[str, tuple[str, ...]] = {
    "claude-code": ("CLAUDE_CODE_OAUTH_TOKEN",),
    # OpenCode's own provider docs (opencode.ai/docs/providers) name these as
    # what selects a provider's account, project, region or credential:
    "opencode": (
        # Google Vertex: "Point GOOGLE_APPLICATION_CREDENTIALS at the
        # service-account JSON file" (a path); project from
        # GOOGLE_VERTEX_PROJECT, GOOGLE_CLOUD_PROJECT, GCP_PROJECT or
        # GCLOUD_PROJECT; location from GOOGLE_VERTEX_LOCATION,
        # GOOGLE_CLOUD_LOCATION or VERTEX_LOCATION.
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_VERTEX_PROJECT",
        "GOOGLE_CLOUD_PROJECT",
        "GCP_PROJECT",
        "GCLOUD_PROJECT",
        "GOOGLE_VERTEX_LOCATION",
        "GOOGLE_CLOUD_LOCATION",
        "VERTEX_LOCATION",
        # Amazon Bedrock: "A configured profile, AWS_PROFILE, AWS_ACCESS_KEY_ID,
        # web identity token file ... activates the provider"; region from
        # AWS_REGION then AWS_DEFAULT_REGION; API keys in
        # AWS_BEARER_TOKEN_BEDROCK. The profile and token FILES are paths.
        "AWS_PROFILE",
        "AWS_ACCESS_KEY_ID",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        "AWS_SHARED_CREDENTIALS_FILE",
        "AWS_CONFIG_FILE",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "AWS_BEARER_TOKEN_BEDROCK",
        # Azure: AZURE_RESOURCE_NAME "supplies the resource name"; the legacy
        # AZURE_COGNITIVE_SERVICES_RESOURCE_NAME "also works".
        "AZURE_RESOURCE_NAME",
        "AZURE_COGNITIVE_SERVICES_RESOURCE_NAME",
        # OpenRouter: "Provide OPENROUTER_API_KEY to the OpenCode server".
        "OPENROUTER_API_KEY",
    ),
}
# Codex injects these into every command it runs (codex-rs
# `protocol/src/shell_environment.rs`, `core/src/exec_env.rs`): ids of the
# PARENT Codex session, thread and tool call. A daemon started from inside a
# Codex session inherits them; they select no model, endpoint or account, so a
# restart from another session must not move the identity. Matched without
# case, as Windows environment names are. Kept bound on purpose:
# `CODEX_SANDBOX*` (changes how the launched CLI's HTTP client behaves) and
# `CODEX_INTERNAL_ORIGINATOR_OVERRIDE` (a stable wrapper label, not per
# session).
_PARENT_SESSION_ENVS = frozenset(
    {"CODEX_SESSION_ID", "CODEX_THREAD_ID", "CODEX_TOOL_CALL_ID"}
)
# A credential's VALUE never enters an identity: rotating a key must not move
# it. Only whether one is set (which selects the credential source) does.
_SECRET_NAME_PARTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "HEADERS")
# Variables whose value is the PATH of a credential/config file, not a secret:
# which file is read decides the account, so the resolved absolute path is
# bound (the file is never opened). Checked before the secret rule, which
# would otherwise reduce e.g. `GOOGLE_APPLICATION_CREDENTIALS` to presence.
# Any `*_FILE` variable names a file the same way.
_FILE_PATH_ENVS = frozenset({"GOOGLE_APPLICATION_CREDENTIALS"})
_FILE_PATH_SUFFIX = "_FILE"


def _file_path_env(name: str) -> bool:
    upper = name.upper()
    return upper in _FILE_PATH_ENVS or upper.endswith(_FILE_PATH_SUFFIX)


def _bound_env_value(name: str, value: str) -> str | bool | None:
    """How one host variable enters the identity: path, presence or value.

    A secret binds only NONEMPTY presence: a blank one selects no credential
    source, so it is omitted (``None``) exactly as if unset.
    """
    if _file_path_env(name):
        return str(Path(value).expanduser().absolute()) if value else value
    if any(part in name.upper() for part in _SECRET_NAME_PARTS):
        return True if value else None
    return value


def _account_dir(host: str, name: str, default: tuple[str, ...]) -> str:
    """The directory a CLI actually uses for ``name``, as an absolute path.

    Claude Code and Codex run in this process's working directory, so a
    relative value is anchored there. OpenCode runs in a private per-call
    directory, so a relative value names no stable location and is bound
    as written.
    """
    value = os.environ.get(name)
    if not value:
        return str(Path.home().joinpath(*default))
    path = Path(value)
    if path.is_absolute() or host == "opencode":
        return str(path)
    return str(path.absolute())


def _host_environment(host: str) -> dict[str, str | bool]:
    """The host's own namespace: values, file paths resolved, secrets by presence."""
    bound = {name for name, _ in _HOST_ACCOUNT_DIRS.get(host, ())}
    prefixes = _HOST_ENV_NAMESPACES.get(host, ())
    names = _HOST_ENV_NAMES.get(host, ())
    if host == "claude-code" and any(
        name.upper().startswith(_CLAUDE_CODE_CLOUD_SWITCH) and value
        for name, value in os.environ.items()
    ):
        prefixes = (*prefixes, *_CLAUDE_CODE_CLOUD_NAMESPACES)
    environment: dict[str, str | bool] = {}
    for name, value in sorted(os.environ.items()):
        if (
            (name.upper().startswith(prefixes) or name.upper() in names)
            and name not in bound
            and name.upper() not in _PARENT_SESSION_ENVS
        ):
            bound_value = _bound_env_value(name, value)
            if bound_value is not None:
                environment[name] = bound_value
    return environment


def native_launch_identity_digest() -> str:
    """Digest only what decides WHO answers a native call: binary and account.

    ``native_policy_digest`` binds the whole launch environment (cwd, PATH,
    PATHEXT, timeout, ...) and suits a plan frozen for one job. A DURABLE
    identity cannot: it must survive a plugin upgrade that moves the working
    directory and a restart from another shell whose PATH differs but resolves
    to the same binary. So this digest binds:

    * the configured native target;
    * the executable the native provider will launch -- the very absolute path
      its resolver returns for dispatch, never PATH itself;
    * HOME and the host's account/configuration directories at their
      effective absolute locations (documented HOME-based defaults when
      unset);
    * the executable override variables;
    * the host's own environment namespace (backend, endpoint, credential
      source and model selection), with credentials bound by presence only,
      credential/config FILE paths by their absolute location, and parent
      Codex session ids excluded.

    Excluded: cwd, PATH, PATHEXT and the timeout, except through what they
    resolve to above.

    Returns:
        str: Hex SHA-256 of the launch identity.
    """
    from reflexio.server.llm.providers.native_cli_provider import (
        _resolve_cli_path,  # noqa: PLC2701 - the one resolver dispatch uses
    )

    target = configured_native_model()
    host = target.host if target is not None else None
    identity = {
        "target": target.reference if target is not None else None,
        "executable": _resolve_cli_path(host) if host is not None else None,
        "home": str(Path.home()),
        "accounts": {
            name: _account_dir(host, name, default)
            for name, default in _HOST_ACCOUNT_DIRS.get(host or "", ())
        }
        if host is not None
        else {},
        "overrides": {name: os.environ.get(name) for name in _EXECUTABLE_OVERRIDE_ENVS},
        "environment": _host_environment(host) if host is not None else {},
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
