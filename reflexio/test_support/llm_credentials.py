"""Provider-credential floor for the OS and enterprise test suites.

Service constructors resolve their default model eagerly — e.g.
``AgentSuccessEvaluator.__init__`` and ``SQLiteStorage.__init__`` both call
``model_defaults.resolve_model_name``, which raises ``RuntimeError: No LLM
provider available`` when the environment carries no provider key. Unit and
integration tests never make a real call (``litellm.completion`` is patched
globally) but they still run through those constructors, so on a machine with
no credentials the suites collapse at fixture setup rather than testing
anything.

A developer machine hides this: ``./.env``, ``~/.reflexio/.env``, and litellm's
import-time dotenv walk-up all feed real keys into ``os.environ``. CI has none,
which is why the same suite can be green locally and ~1,250 errors deep in CI.
"""

from __future__ import annotations

import os
from collections.abc import Collection

from reflexio.server.llm.model_defaults import (
    GENERATION_CAPABLE_PROVIDERS,
    detect_available_providers,
    provider_env_vars,
)

# Matches the string already used by the `storage` fixture in
# tests/server/llm/conftest.py and by the repo's heavy-skill preflight docs.
_PLACEHOLDER_KEY = "sk-placeholder-mock"


def ensure_provider_credential() -> None:
    """Pin a placeholder provider key when the environment offers none.

    Only fills a vacuum — a real key, or an opted-in local CLI provider, is
    left untouched. The placeholder that lands here is a real value to every
    env-var reader, including :func:`detect_available_providers`, so tests
    gating a live provider call must ask :func:`real_provider_key` rather than
    ``os.getenv``.

    Gating on :func:`detect_available_providers` rather than a hand-copied list
    of env var names keeps this from drifting, and correctly treats an
    embedding-only provider (``local``) as "no generation provider" — that case
    resolves embeddings fine but still raises for every other model role.
    """
    if any(p in GENERATION_CAPABLE_PROVIDERS for p in detect_available_providers()):
        return
    os.environ["OPENAI_API_KEY"] = _PLACEHOLDER_KEY


def real_provider_key(name: str) -> str | None:
    """Return ``name``'s value, or None when it is only the placeholder.

    A drop-in replacement for ``os.getenv(name)`` at any site that gates a
    **live** provider call. :func:`detect_available_providers` deliberately
    *does* count the placeholder — that is the entire mechanism, since model
    resolution reads the same env vars it does — so a raw ``os.getenv`` cannot
    tell a real key from the one the floor just pinned, and a
    ``requires_credentials`` test that skipped cleanly on a bare machine would
    instead run against a credential that authenticates with nothing.

    Args:
        name (str): Provider env var name, e.g. ``"OPENAI_API_KEY"``.

    Returns:
        str | None: The configured credential, or None when unset, blank, or
            the placeholder.
    """
    value = os.environ.get(name, "")
    return None if value in ("", _PLACEHOLDER_KEY) else value


def real_generation_provider(
    allowed: Collection[str] | None = None,
) -> str | None:
    """Return a provider that holds a real key and can serve generation.

    The provider-agnostic counterpart to :func:`real_provider_key`, for live
    tests that resolve their model through
    ``resolve_model_name(ModelRole.GENERATION)`` rather than naming one. Gating
    such a test on ``OPENAI_API_KEY``/``ANTHROPIC_API_KEY`` skips it on a
    machine whose only credential is, say, MiniMax -- even though model
    resolution would have picked MiniMax and the test would have run.

    Selection follows the same priority order as
    :func:`detect_available_providers`, so the provider named here is the one
    ``resolve_model_name`` will pick.

    Args:
        allowed (Collection[str] | None): Restrict to these provider keys, e.g.
            ``{"openai", "anthropic", "minimax"}`` for a test whose assertions
            are only known to hold on those models. None accepts any
            generation-capable provider.

    Returns:
        str | None: Provider key, or None when no eligible provider has a real
            key configured.
    """
    env_by_provider = {
        provider: env_var for env_var, provider in provider_env_vars().items()
    }
    for provider in detect_available_providers():
        if provider not in GENERATION_CAPABLE_PROVIDERS:
            continue
        if allowed is not None and provider not in allowed:
            continue
        env_var = env_by_provider.get(provider)
        if env_var and real_provider_key(env_var):
            return provider
    return None
