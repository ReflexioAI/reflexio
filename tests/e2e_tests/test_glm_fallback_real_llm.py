"""Opt-in real-API proof of the MiniMax->GLM fallback through the faithful pipeline.

Lives under ``tests/e2e_tests/`` so the global litellm mock is bypassed
(path-based). The real-API tests hit GLM (Z.ai), a FIXED-QUOTA coding-plan
subscription, so they publish a single small interaction set and never loop.

Run:
    cd open_source/reflexio && \
    RUN_LOW_PRIORITY=1 ZAI_API_KEY=... \
    uv run pytest tests/e2e_tests/test_glm_fallback_real_llm.py -v -o 'addopts=' -s

Coverage (Task 9):
  * ``test_glm_as_primary_pipeline`` (run #1) — GLM forced as the primary
    generation model via the ``default_generation_model_name`` site var; a real
    publish must produce schema-valid profile + playbook rows.
  * ``test_minimax_to_glm_fallback`` (run #2, HIGHEST) — prod-shape ladder
    ``minimax/MiniMax-M3`` (site var) + ``REFLEXIO_LLM_FALLBACK_MODELS=zai/glm-5.2``
    with an UNROUTABLE MiniMax key, so the reflexio-owned walk MUST advance to
    GLM. Asserts publish succeeds, rows exist, and the
    ``event=llm_fallback_used ... served_model=zai/glm-5.2 reason=transport_error``
    signal was emitted (at least once).

Run #3b (a configured GLM fallback with no provider key failing the Task-4 boot
check ``validate_llm_availability``) is deliberately NOT in this file — it makes
no real API call, so it belongs in the unit tier and is already covered by
``tests/server/llm/test_model_defaults.py::test_configured_fallback_without_provider_key_raises``.

The faithful path is mandatory: the primary is forced through the site var, the
fallback through the env var, both resolved by ``resolve_model_name``, and the
publish is driven by the real ``Reflexio`` library entrypoint. No hand-injected
``LiteLLMClient``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from reflexio.lib.reflexio_lib import Reflexio
from reflexio.models.api_schema.service_schemas import InteractionData
from reflexio.models.config_schema import (
    SINGLETON_USER_PLAYBOOK_NAME,
    Config,
    PlaybookConfig,
    ProfileExtractorConfig,
    StorageConfigSQLite,
)
from reflexio.server.services.configurator.configurator import DefaultConfigurator
from reflexio.server.services.storage.storage_base import BaseStorage
from reflexio.server.services.tagging.tagging_scheduler import drain_tagging
from reflexio.server.site_var.site_var_manager import SiteVarManager
from reflexio.test_support.skip_decorators import skip_low_priority

pytestmark = [pytest.mark.e2e, pytest.mark.requires_credentials]

_GLM_MODEL = "zai/glm-5.2"
_MINIMAX_MODEL = "minimax/MiniMax-M3"
# A syntactically valid but unroutable key so the MiniMax rung fails transport
# (auth) fast, forcing the walk to advance to the GLM fallback.
_UNROUTABLE_MINIMAX_KEY = "sk-unroutable-minimax-key-for-fallback-test"

_real_get_site_var = SiteVarManager.get_site_var


@contextmanager
def _force_generation_model(model_name: str) -> Iterator[None]:
    """Faithfully force the generation model via the ``llm_model_setting`` site var.

    Patches ``SiteVarManager.get_site_var`` so every publish-path component
    (profile / playbook extractor, tagging, dedup) resolves
    ``default_generation_model_name`` to ``model_name`` through the real
    ``resolve_model_name`` chain. All other site vars pass through to the real
    loader so feature flags etc. behave normally.

    Args:
        model_name: The LiteLLM model name to force as the generation primary.
    """
    overlay = {
        "should_run_model_name": "",
        "default_generation_model_name": model_name,
        "default_evaluate_model_name": model_name,
        "embedding_model_name": "",
        "pre_retrieval_model_name": "",
    }

    def _patched(self: SiteVarManager, name: str):  # type: ignore[no-untyped-def]
        if name == "llm_model_setting":
            return dict(overlay)
        return _real_get_site_var(self, name)

    with patch.object(SiteVarManager, "get_site_var", _patched):
        yield


def _require_glm_key() -> None:
    """Skip when the real GLM key is absent (nothing to authenticate against)."""
    if not os.environ.get("ZAI_API_KEY"):
        pytest.skip("ZAI_API_KEY not set — real GLM key required for this test")


def _build_instance(storage_config: StorageConfigSQLite, org_id: str) -> Reflexio:
    """Build a Reflexio instance with a profile + playbook config.

    Mirrors the conftest ``reflexio_instance`` config (minus agent-success, which
    is not on the publish path). Constructed inside the test body so that the
    ``self.llm_client`` built here reads ``REFLEXIO_LLM_FALLBACK_MODELS`` from the
    environment set moments earlier — the faithful prod ordering (env is present
    before the process builds its client).

    Args:
        storage_config: Per-test SQLite storage config (isolated tmp dir).
        org_id: Per-worker test org id.

    Returns:
        Reflexio: A fresh instance whose generation client carries the env-derived
        fallback ladder.
    """
    config = Config(
        storage_config=storage_config,
        agent_context_prompt="this is a sales agent",
        skip_should_run_check=True,
        window_size=20,
        profile_extractor_config=ProfileExtractorConfig(
            extractor_name="test_profile_extractor",
            context_prompt=(
                "Conversation between sales agent and user, extract any "
                "information from the interaction if it contains information "
                "listed under definition"
            ),
            extraction_definition_prompt=(
                "name, occupation, location, membership tier, order context, "
                "communication preferences, and other durable customer-support "
                "personalization facts from the conversation"
            ),
            tagging_definition_prompt="choice of ['basic_info', 'conversation_intent']",
        ),
        user_playbook_extractor_config=PlaybookConfig(
            extractor_name="test_playbook",
            extraction_definition_prompt=(
                "playbook should be something the user told you to do differently "
                "in the next session. content is what the agent should do "
                "differently, as actionable as possible."
            ),
        ),
    )
    configurator = DefaultConfigurator(org_id=org_id, config=config)
    return Reflexio(org_id=org_id, configurator=configurator)


def _publish_priya(
    instance: Reflexio, user_id: str, interactions: list[InteractionData]
) -> None:
    """Drive a real publish through the faithful library entrypoint."""
    response = instance.publish_interaction(
        {
            "user_id": user_id,
            "interaction_data_list": interactions,
            "source": "test_glm_fallback",
            "agent_version": "glm_fallback_v1",
            "session_id": "glm_fallback_session",
        }
    )
    assert response.success is True, "publish_interaction should succeed"


def _assert_rows_produced(storage: BaseStorage) -> None:
    """Assert the publish produced schema-valid profile and playbook rows."""
    profiles = storage.get_all_profiles()
    assert profiles, "publish should produce at least one profile row"
    assert profiles[0].content.strip(), "profile content should be non-empty"

    playbooks = storage.get_user_playbooks(playbook_name=SINGLETON_USER_PLAYBOOK_NAME)
    assert playbooks, "publish should produce at least one user playbook row"
    assert playbooks[0].content.strip(), "playbook content should be non-empty"


@skip_low_priority
def test_glm_as_primary_pipeline(
    sqlite_storage_config: StorageConfigSQLite,
    test_org_id: str,
    sample_interaction_requests: list[InteractionData],
    monkeypatch: pytest.MonkeyPatch,
):
    """Run #1: GLM-5.2 as the sole generation model drives the full publish pipeline.

    Builds a fresh instance (mirroring run #2) AFTER clearing any stray
    ``REFLEXIO_LLM_FALLBACK_MODELS`` from the environment, so the client's
    ``LiteLLMConfig.fallback_models`` (read at construction time) is provably
    empty. That way a real publish through GLM proves GLM-as-primary is the only
    path — not a pass smuggled in via an unrelated fallback provider.
    """
    _require_glm_key()

    # Clear any stray fallback ladder BEFORE the client is built.
    monkeypatch.delenv("REFLEXIO_LLM_FALLBACK_MODELS", raising=False)

    with _force_generation_model(_GLM_MODEL):
        instance = _build_instance(sqlite_storage_config, test_org_id)
        storage = instance.request_context.storage
        assert storage is not None
        _publish_priya(instance, "glm_primary_user", sample_interaction_requests)
        # Drain background tagging while the site-var patch is still active so the
        # tagging pass also resolves to GLM (the callback constructs its client
        # lazily and reads the site var at run time).
        assert drain_tagging(timeout_seconds=60.0)

    _assert_rows_produced(storage)


@skip_low_priority
def test_minimax_to_glm_fallback(
    sqlite_storage_config: StorageConfigSQLite,
    test_org_id: str,
    sample_interaction_requests: list[InteractionData],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
):
    """Run #2 (HIGHEST): prod-shape ladder must advance from a dead MiniMax to GLM.

    Primary ``minimax/MiniMax-M3`` (site var) with an unroutable key +
    ``REFLEXIO_LLM_FALLBACK_MODELS=zai/glm-5.2``. The MiniMax rung fails transport
    (unroutable key), so the reflexio-owned walk advances to GLM, which actually
    serves the request. This is the only real-API exercise of the rewritten
    per-rung fallback code path. The test asserts the fallback signal was
    emitted (at least once), not that it fired on every generation call.

    The fallback ladder lives on the generation client's ``LiteLLMConfig``, whose
    ``fallback_models`` is read from ``REFLEXIO_LLM_FALLBACK_MODELS`` at client
    construction. In prod that env is set before the process boots, so the
    instance is built here *after* the env is set (faithful ordering) rather than
    reusing the shared fixture whose client predates the env.
    """
    _require_glm_key()

    # Set the ladder + dead primary key BEFORE the client is built.
    monkeypatch.setenv("REFLEXIO_LLM_FALLBACK_MODELS", _GLM_MODEL)
    monkeypatch.setenv("MINIMAX_API_KEY", _UNROUTABLE_MINIMAX_KEY)

    with (
        caplog.at_level(logging.INFO, logger="reflexio.server.llm.litellm_client"),
        _force_generation_model(_MINIMAX_MODEL),
    ):
        instance = _build_instance(sqlite_storage_config, test_org_id)
        storage = instance.request_context.storage
        assert storage is not None
        _publish_priya(instance, "fallback_user", sample_interaction_requests)
        assert drain_tagging(timeout_seconds=60.0)

    _assert_rows_produced(storage)

    fallback_lines = [
        r.getMessage()
        for r in caplog.records
        if "event=llm_fallback_used" in r.getMessage()
    ]
    assert fallback_lines, (
        "expected an event=llm_fallback_used signal — the walk did not advance "
        "past the dead MiniMax primary"
    )
    joined = "\n".join(fallback_lines)
    assert f"primary_model={_MINIMAX_MODEL}" in joined, joined
    assert f"served_model={_GLM_MODEL}" in joined, joined
    assert "reason=transport_error" in joined, joined
