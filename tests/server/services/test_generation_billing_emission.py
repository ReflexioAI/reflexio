"""Tests for ② Learning billing event emission at the generation service layer.

Verifies that:
- A real extraction emits ``extraction_tokens`` and ``learnings_generated`` events.
- A should_run-skipped request emits NO billing learning events (but the ops gate
  event still fires with ``outcome=should_skip``).
- AgentSuccessEvaluationService (EMITS_LEARNING_BILLING=False) never emits ②
  Learning events — the gate blocks it regardless of outcome.

Uses the verified SQLite + litellm mock harness so no real LLM calls are made.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from reflexio.models.api_schema.service_schemas import Interaction, Request
from reflexio.models.config_schema import (
    Config,
    ProfileExtractorConfig,
    StorageConfigSQLite,
)
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.billing_signals import count_input_tokens
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.services.base_generation_service import (
    BaseGenerationService,
    PreparedGenerationRun,
)
from reflexio.server.services.profile.profile_generation_service_utils import (
    ProfileGenerationRequest,
)
from reflexio.server.services.profile.service import (
    ProfileGenerationService,
    ProfileGenerationServiceConfig,
)
from reflexio.server.services.service_utils import format_sessions_to_history_string
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.usage_metrics import UsageEvent, configure_usage_event_recorder

# ---------------------------------------------------------------------------
# Helpers: build a RequestContext backed by a real SQLiteStorage, mirroring
# the pattern from tests/e2e_tests/test_resumable_extraction_e2e.py.
# ---------------------------------------------------------------------------

_ORG_ID = "billing_emission_test_org"
_USER_ID = "billing_emission_user"
_REQUEST_ID = "billing_emission_req_1"


def _request_context(storage: SQLiteStorage) -> RequestContext:
    """Build a RequestContext wired to ``storage``, with a fixed Config."""
    ctx = RequestContext.__new__(RequestContext)
    ctx.org_id = _ORG_ID
    ctx.storage = storage
    ctx.storage_base_dir = None
    ctx.configurator = MagicMock()
    ctx.configurator.get_config.return_value = Config(
        storage_config=StorageConfigSQLite(),
        profile_extractor_config=ProfileExtractorConfig(
            extractor_name="billing_test_extractor",
            extraction_definition_prompt="Extract durable preferences.",
        ),
    )
    ctx.configurator.get_agent_context.return_value = "Billing test agent context"
    ctx.prompt_manager = MagicMock()
    ctx.prompt_manager.render_prompt.side_effect = lambda prompt_id, variables: (
        f"{prompt_id}: {variables}"
    )
    return ctx


def _build_sqlite_storage(tmp_path: Any) -> SQLiteStorage:
    """Create a SQLiteStorage with _get_embedding patched to a zero vector."""
    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        return SQLiteStorage(org_id=_ORG_ID, db_path=str(tmp_path / "reflexio.db"))


def _seed_interactions(storage: SQLiteStorage) -> None:
    """Seed substantive interactions that should trigger extraction."""
    storage.add_request(
        Request(
            request_id=_REQUEST_ID,
            user_id=_USER_ID,
            created_at=1_000,
            source="api",
            agent_version="v1",
            session_id=_REQUEST_ID,
        )
    )
    storage._insert_interaction(
        Interaction(
            interaction_id=1,
            user_id=_USER_ID,
            request_id=_REQUEST_ID,
            created_at=1_000,
            role="user",
            content=(
                "Please remember that I always prefer dark-mode UIs and prefer "
                "concise answers without emojis. This is important for my workflow."
            ),
        )
    )
    storage._insert_interaction(
        Interaction(
            interaction_id=2,
            user_id=_USER_ID,
            request_id=_REQUEST_ID,
            created_at=1_001,
            role="assistant",
            content="Understood, I will remember those preferences.",
        )
    )


def _run_profile_generation(storage: SQLiteStorage, *, auto_run: bool = True) -> None:
    """Run one ProfileGenerationService extraction over the given storage."""
    ctx = _request_context(storage)
    llm_config = LiteLLMConfig(model="gpt-4o-mini")
    llm_client = LiteLLMClient(llm_config)
    service = ProfileGenerationService(llm_client=llm_client, request_context=ctx)
    request = ProfileGenerationRequest(
        user_id=_USER_ID,
        request_id=_REQUEST_ID,
        source="api",
        auto_run=auto_run,
    )
    service.run(request)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_real_extraction_emits_tokens_and_learnings(tmp_path):
    """A successful extraction emits extraction_tokens + learnings_generated.

    MOCK_LLM_RESPONSE=true is active (autouse conftest), so the profile extractor
    takes the deterministic mock path and we verify the billing events it triggers.
    auto_run=False bypasses the should_run gate so extraction always fires.
    """
    events: list[UsageEvent] = []
    configure_usage_event_recorder(events.append)
    try:
        with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
            storage = SQLiteStorage(
                org_id=_ORG_ID, db_path=str(tmp_path / "reflexio.db")
            )
            _seed_interactions(storage)
            _run_profile_generation(storage, auto_run=False)
    finally:
        configure_usage_event_recorder(None)

    learning = [e for e in events if e.event_category == "learning"]
    names = {e.event_name for e in learning}
    assert "extraction_tokens" in names, f"expected extraction_tokens in {names}"
    assert "learnings_generated" in names, f"expected learnings_generated in {names}"

    # extraction_tokens.count_value is the input-anchored basis (> 0 because the
    # interactions are non-empty), independent of whatever the mock reports.
    tok = next(e for e in learning if e.event_name == "extraction_tokens")
    assert tok.count_value > 0
    assert tok.billing_input_tokens == tok.count_value
    assert tok.platform_llm is True  # no api_key_config in the seeded Config

    # learnings_generated.count_value must equal the existing generation_succeeded count.
    gen = next(e for e in learning if e.event_name == "learnings_generated")
    succ = next(e for e in events if e.event_name == "generation_succeeded")
    assert gen.count_value == succ.count_value


def test_should_run_skip_emits_no_learning_billing(tmp_path, monkeypatch):
    """A should_run-gated skip emits NO extraction_tokens / learnings_generated.

    The ops gate event still fires with outcome=should_skip. We:
    - Clear MOCK_LLM_RESPONSE so the pre-extraction gate is not bypassed.
    - Set stride_size=1 and seed a short interaction so the stride check passes
      but the cheap pre-filter rejects the batch (all_user_turns_too_short).
    """
    monkeypatch.delenv("MOCK_LLM_RESPONSE", raising=False)

    events: list[UsageEvent] = []
    configure_usage_event_recorder(events.append)
    try:
        with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
            storage = SQLiteStorage(
                org_id=_ORG_ID, db_path=str(tmp_path / "reflexio.db")
            )
            # Seed a very short interaction (< 30 chars) that the cheap pre-filter
            # rejects as "all_user_turns_too_short".
            storage.add_request(
                Request(
                    request_id=_REQUEST_ID,
                    user_id=_USER_ID,
                    created_at=1_000,
                    source="api",
                    session_id=_REQUEST_ID,
                )
            )
            storage._insert_interaction(
                Interaction(
                    interaction_id=1,
                    user_id=_USER_ID,
                    request_id=_REQUEST_ID,
                    created_at=1_000,
                    role="User",
                    content="hi",  # << 30 chars → all_user_turns_too_short
                )
            )

            # Build a custom request context with stride_size=1 so the stride
            # pre-filter passes (new=1, stride_size=1).
            ctx = RequestContext.__new__(RequestContext)
            ctx.org_id = _ORG_ID
            ctx.storage = storage
            ctx.storage_base_dir = None
            ctx.configurator = MagicMock()
            ctx.configurator.get_config.return_value = Config(
                storage_config=StorageConfigSQLite(),
                profile_extractor_config=ProfileExtractorConfig(
                    extractor_name="billing_test_extractor",
                    extraction_definition_prompt="Extract durable preferences.",
                ),
                stride_size=1,
            )
            ctx.configurator.get_agent_context.return_value = "Billing skip test"
            ctx.prompt_manager = MagicMock()
            ctx.prompt_manager.render_prompt.side_effect = lambda prompt_id, variables: (
                f"{prompt_id}: {variables}"
            )

            llm_config = LiteLLMConfig(model="gpt-4o-mini")
            llm_client = LiteLLMClient(llm_config)
            service = ProfileGenerationService(
                llm_client=llm_client, request_context=ctx
            )
            request = ProfileGenerationRequest(
                user_id=_USER_ID,
                request_id=_REQUEST_ID,
                source="api",
                auto_run=True,
            )
            service.run(request)
    finally:
        configure_usage_event_recorder(None)

    billing_learning = [
        e
        for e in events
        if e.event_category == "learning"
        and e.event_name in {"extraction_tokens", "learnings_generated"}
    ]
    assert billing_learning == [], f"unexpected billing events: {billing_learning}"

    # The existing ops event still fires with the skip outcome.
    gate_events = [e for e in events if e.event_name == "generation_gate_evaluated"]
    assert any(e.outcome == "should_skip" for e in gate_events), (
        f"expected should_skip gate event, got: {gate_events}"
    )


# ---------------------------------------------------------------------------
# Gate test: non-learning services must NEVER emit ② Learning billing events
# ---------------------------------------------------------------------------


def _make_minimal_request_context() -> RequestContext:
    """Return a RequestContext with a minimal Config (no profile/playbook config)."""
    ctx = RequestContext.__new__(RequestContext)
    ctx.org_id = _ORG_ID
    ctx.storage = MagicMock()
    ctx.storage_base_dir = None
    ctx.configurator = MagicMock()
    ctx.configurator.get_config.return_value = Config(
        storage_config=StorageConfigSQLite(),
    )
    ctx.configurator.get_agent_context.return_value = "gate test context"
    ctx.prompt_manager = MagicMock()
    return ctx


class _StubService(BaseGenerationService):  # type: ignore[type-arg]
    """Minimal concrete subclass with EMITS_LEARNING_BILLING=False (the default).

    Used to assert that the gate blocks ② Learning emission when a service has
    not opted in.  Mirrors what AgentSuccessEvaluationService and any other
    bundled service would look like.
    """

    EMITS_LEARNING_BILLING: bool = False  # explicit; also inherited default

    def _load_extractor_config(self):  # pragma: no cover
        return None

    def _load_generation_service_config(self, request):  # pragma: no cover
        return request

    def _create_extractor(self, extractor_config, service_config):  # pragma: no cover
        return MagicMock()

    def _get_service_name(self) -> str:
        return "stub_evaluation"

    def _get_base_service_name(self) -> str:
        return "stub_evaluation"

    def _process_results(self, results):  # pragma: no cover
        pass

    def _should_track_in_progress(self) -> bool:
        return False

    def _get_lock_scope_id(self, request):  # pragma: no cover
        return None


def test_non_learning_service_emits_no_learning_billing_events():
    """Services with EMITS_LEARNING_BILLING=False must emit zero ② Learning events.

    The test directly calls ``_record_billing_learning_events`` (with a recorder
    installed) and asserts that neither ``learnings_generated`` nor
    ``extraction_tokens`` are emitted.  This proves the opt-in gate works
    regardless of ``generated_count``, mirroring how AgentSuccessEvaluationService
    behaves.
    """
    ctx = _make_minimal_request_context()
    llm_client = LiteLLMClient(LiteLLMConfig(model="gpt-4o-mini"))
    service = _StubService(llm_client=llm_client, request_context=ctx)
    # Prime service_config so _usage_context() doesn't choke
    service.service_config = MagicMock()

    prepared = PreparedGenerationRun(
        extractor_config=MagicMock(),
        extractor_name="stub_extractor",
        identifier="stub_user",
    )

    events: list[UsageEvent] = []
    configure_usage_event_recorder(events.append)
    try:
        service._record_billing_learning_events(prepared=prepared, generated_count=5)
    finally:
        configure_usage_event_recorder(None)

    learning_events = [
        e
        for e in events
        if e.event_name in {"learnings_generated", "extraction_tokens"}
    ]
    assert learning_events == [], (
        f"EMITS_LEARNING_BILLING=False service must not emit learning events; got: {learning_events}"
    )


# ---------------------------------------------------------------------------
# Dedup: billing reuses the should-run gate's already-fetched window instead of
# re-querying storage purely to recompute billing_input_tokens.
# ---------------------------------------------------------------------------


def _build_profile_service(storage: SQLiteStorage) -> ProfileGenerationService:
    """Build a ProfileGenerationService over ``storage`` with service_config primed."""
    ctx = _request_context(storage)
    llm_client = LiteLLMClient(LiteLLMConfig(model="gpt-4o-mini"))
    service = ProfileGenerationService(llm_client=llm_client, request_context=ctx)
    service.service_config = ProfileGenerationServiceConfig(
        user_id=_USER_ID,
        request_id=_REQUEST_ID,
        source="api",
        auto_run=True,
    )
    return service


def _known_sessions(storage: SQLiteStorage) -> list[Any]:
    """Return realistic RequestInteractionDataModel objects from real storage.

    Seeds substantive interactions and reads them back through the same storage
    method (`get_last_k_interactions_grouped`) the billing path uses, so the
    session models have the exact shape `format_sessions_to_history_string`
    consumes in production.
    """
    _seed_interactions(storage)
    sessions, _ = storage.get_last_k_interactions_grouped(
        user_id=_USER_ID, k=10, sources=["api"]
    )
    assert sessions, "expected seeded interactions to be returned"
    return sessions


def _prepared() -> PreparedGenerationRun[Any]:
    """A minimal prepared run with a real profile extractor config."""
    return PreparedGenerationRun(
        extractor_config=ProfileExtractorConfig(
            extractor_name="billing_test_extractor",
            extraction_definition_prompt="Extract durable preferences.",
        ),
        extractor_name="billing_test_extractor",
        identifier=_USER_ID,
    )


def test_billing_reuses_precheck_window_without_refetch(tmp_path):
    """When the gate stashed its window, billing reuses it and skips storage.

    Proves the dedup: `_extraction_input_text` returns the formatted text of the
    stashed sessions and never calls `get_last_k_interactions_grouped`.
    """
    storage = _build_sqlite_storage(tmp_path)
    known_sessions = _known_sessions(storage)
    expected_text = format_sessions_to_history_string(known_sessions)

    service = _build_profile_service(storage)
    service._last_precheck_sessions = known_sessions
    # Replace storage with a mock so any storage read would be observable.
    service.storage = MagicMock()

    text = service._extraction_input_text(_prepared())

    assert text == expected_text
    service.storage.get_last_k_interactions_grouped.assert_not_called()


def test_billing_refetches_when_no_precheck_window(tmp_path):
    """When the gate did NOT pre-fetch, billing falls back to its own storage read.

    Proves the fallback still works on bypass paths: with the stash None,
    `_extraction_input_text` calls `get_last_k_interactions_grouped` exactly once
    and returns the same formatted text.
    """
    storage = _build_sqlite_storage(tmp_path)
    known_sessions = _known_sessions(storage)
    expected_text = format_sessions_to_history_string(known_sessions)

    service = _build_profile_service(storage)
    service._last_precheck_sessions = None
    mock_storage = MagicMock()
    mock_storage.get_last_k_interactions_grouped.return_value = (known_sessions, None)
    service.storage = mock_storage

    text = service._extraction_input_text(_prepared())

    assert text == expected_text
    mock_storage.get_last_k_interactions_grouped.assert_called_once()


def test_billing_token_count_unchanged_reuse_vs_refetch(tmp_path):
    """Equivalence guard: the reuse path and the refetch path bill identically.

    For the same window, the reuse-path text must equal the refetch-path text and
    `count_input_tokens` must agree — the metered quantity is byte-identical, so
    the refactor cannot change a customer's bill.
    """
    storage = _build_sqlite_storage(tmp_path)
    known_sessions = _known_sessions(storage)

    # Reuse path: stash set, storage mocked to prove no read happens.
    reuse_service = _build_profile_service(storage)
    reuse_service._last_precheck_sessions = known_sessions
    reuse_service.storage = MagicMock()
    reuse_text = reuse_service._extraction_input_text(_prepared())
    reuse_service.storage.get_last_k_interactions_grouped.assert_not_called()

    # Refetch path: stash None, storage returns the same window.
    refetch_service = _build_profile_service(storage)
    refetch_service._last_precheck_sessions = None
    refetch_storage = MagicMock()
    refetch_storage.get_last_k_interactions_grouped.return_value = (
        known_sessions,
        None,
    )
    refetch_service.storage = refetch_storage
    refetch_text = refetch_service._extraction_input_text(_prepared())
    refetch_storage.get_last_k_interactions_grouped.assert_called_once()

    assert reuse_text == refetch_text
    assert count_input_tokens(reuse_text) == count_input_tokens(refetch_text)
