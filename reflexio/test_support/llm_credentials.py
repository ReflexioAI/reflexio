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

from reflexio.server.llm.model_defaults import (
    GENERATION_CAPABLE_PROVIDERS,
    detect_available_providers,
)

# Matches the string already used by the `storage` fixture in
# tests/server/llm/conftest.py and by the repo's heavy-skill preflight docs.
_PLACEHOLDER_KEY = "sk-placeholder-mock"


def ensure_provider_credential() -> None:
    """Pin a placeholder provider key when the environment offers none.

    Only fills a vacuum — a real key, or an opted-in local CLI provider, is
    left untouched, so the ``requires_credentials`` tier still runs against
    whatever the developer has configured.

    Gating on :func:`detect_available_providers` rather than a hand-copied list
    of env var names keeps this from drifting, and correctly treats an
    embedding-only provider (``local``) as "no generation provider" — that case
    resolves embeddings fine but still raises for every other model role.
    """
    if any(p in GENERATION_CAPABLE_PROVIDERS for p in detect_available_providers()):
        return
    os.environ["OPENAI_API_KEY"] = _PLACEHOLDER_KEY
