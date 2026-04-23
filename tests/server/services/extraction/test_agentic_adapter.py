"""Unit tests for the agentic extraction adapter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from reflexio.models.api_schema.domain.entities import (
    NEVER_EXPIRES_TIMESTAMP,
    Interaction,
    Request,
    UserPlaybook,
    UserProfile,
)
from reflexio.models.api_schema.domain.enums import ProfileTimeToLive, Status
from reflexio.models.api_schema.service_schemas import PublishUserInteractionRequest
from reflexio.models.config_schema import (
    Config,
    PlaybookAggregatorConfig,
    StorageConfigSQLite,
    UserPlaybookExtractorConfig,
)
from reflexio.server.services.extraction.agentic_adapter import (
    AgenticExtractionRunner,
    _compute_expiration,
    _vetted_to_user_playbook,
    _vetted_to_user_profile,
)
from reflexio.server.services.extraction.agentic_extraction_service import (
    ExtractionResult,
)
from reflexio.server.services.extraction.critics import VettedPlaybook, VettedProfile

# ---------------- TTL mapping ---------------- #


def test_ttl_infinity_maps_to_never_expires():
    assert (
        _compute_expiration("infinity", now_ts=1_700_000_000) == NEVER_EXPIRES_TIMESTAMP
    )


def test_ttl_one_week_maps_to_seven_days_out():
    now = 1_700_000_000
    assert _compute_expiration("one_week", now_ts=now) == now + 7 * 86_400


def test_ttl_one_year_maps_to_three_sixty_five_days():
    now = 1_700_000_000
    assert _compute_expiration("one_year", now_ts=now) == now + 365 * 86_400


# ---------------- converters ---------------- #


def test_vetted_profile_conversion_preserves_agentic_fields():
    vp = VettedProfile(
        content="User prefers polars.",
        time_to_live="infinity",
        source_span="I use polars",
        notes="high-confidence",
        reader_angle="facts",
    )
    out = _vetted_to_user_profile(
        vp,
        user_id="u_test",
        request_id="req_abc",
        source="cli",
        now_ts=1_700_000_000,
    )

    assert isinstance(out, UserProfile)
    assert out.user_id == "u_test"
    assert out.content == "User prefers polars."
    assert out.generated_from_request_id == "req_abc"
    assert out.source == "cli"
    assert out.profile_time_to_live == ProfileTimeToLive.INFINITY
    assert out.expiration_timestamp == NEVER_EXPIRES_TIMESTAMP
    assert out.source_span == "I use polars"
    assert out.notes == "high-confidence"
    assert out.reader_angle == "facts"
    assert out.extractor_names == ["agentic"]
    assert out.profile_id  # a UUID was generated


def test_vetted_playbook_conversion_fills_enterprise_fields():
    vpb = VettedPlaybook(
        trigger="user says ship",
        content="run tests then deploy",
        rationale="after the april regression",
        source_span="run tests then deploy",
        notes="from playbook critic",
        reader_angle="rationale",
    )
    out = _vetted_to_user_playbook(
        vpb,
        user_id="u_test",
        request_id="req_abc",
        agent_version="v1",
        source="cli",
        now_ts=1_700_000_000,
    )

    assert isinstance(out, UserPlaybook)
    assert out.user_id == "u_test"
    assert out.request_id == "req_abc"
    assert out.agent_version == "v1"
    assert out.created_at == 1_700_000_000
    assert out.trigger == "user says ship"
    assert out.content == "run tests then deploy"
    assert out.rationale == "after the april regression"
    assert out.source == "cli"
    assert out.source_span == "run tests then deploy"
    assert out.reader_angle == "rationale"
    assert out.user_playbook_id == 0  # DB autoincrement placeholder


def test_vetted_playbook_with_none_content_becomes_empty_string():
    """UserPlaybook.content has a non-None contract; the adapter must coerce."""
    vpb = VettedPlaybook(trigger="x", content=None, rationale=None)
    out = _vetted_to_user_playbook(
        vpb,
        user_id="u",
        request_id="r",
        agent_version="v",
        source=None,
        now_ts=1,
    )
    assert out.content == ""


# ---------------- AgenticExtractionRunner ---------------- #


def _make_interaction(role: str, content: str, user_id: str = "u_test") -> Interaction:
    return Interaction(
        interaction_id=0,
        user_id=user_id,
        request_id="req_abc",
        role=role,
        content=content,
    )


def _make_request(session_id: str = "s1") -> Request:
    return Request(
        request_id="req_abc",
        user_id="u_test",
        source="cli",
        agent_version="v1",
        session_id=session_id,
    )


def _make_publish_request(
    *, force_extraction: bool = False, skip_aggregation: bool = False
) -> PublishUserInteractionRequest:
    return PublishUserInteractionRequest(
        user_id="u_test",
        interaction_data_list=[{"role": "User", "content": "hi"}],  # type: ignore[list-item]
        source="cli",
        agent_version="v1",
        force_extraction=force_extraction,
        skip_aggregation=skip_aggregation,
    )


def _make_runner(
    storage: MagicMock | None = None,
    *,
    service_result: ExtractionResult | None = None,
) -> AgenticExtractionRunner:
    rc = MagicMock()
    rc.storage = storage if storage is not None else MagicMock()
    rc.prompt_manager = MagicMock()
    rc.configurator = MagicMock()
    rc.org_id = "test-org"

    runner = AgenticExtractionRunner(
        llm_client=MagicMock(),
        request_context=rc,
        org_id="test-org",
    )
    # Replace the underlying service with a MagicMock that returns the
    # provided ExtractionResult. Prevents real LLM / ThreadPoolExecutor work.
    runner.service = MagicMock()
    runner.service.run.return_value = (
        service_result if service_result is not None else ExtractionResult()
    )
    return runner


def test_runner_pre_filter_skips_zero_user_turn_session():
    """No User-role interactions → pre-filter rejects, service.run not called."""
    runner = _make_runner()
    publish_req = _make_publish_request()

    out = runner.run(
        publish_request=publish_req,
        request_id="req_abc",
        new_interactions=[_make_interaction("Agent", "hello")],  # no User turns
        new_request=_make_request(),
        config=Config(storage_config=StorageConfigSQLite()),
    )

    assert out == []
    runner.service.run.assert_not_called()  # type: ignore[attr-defined]


def test_runner_force_extraction_bypasses_pre_filter():
    """force_extraction=True makes the service run even when pre-filter would reject."""
    runner = _make_runner()
    publish_req = _make_publish_request(force_extraction=True)

    runner.run(
        publish_request=publish_req,
        request_id="req_abc",
        new_interactions=[_make_interaction("Agent", "no user turn here")],
        new_request=_make_request(),
        config=Config(storage_config=StorageConfigSQLite()),
    )

    runner.service.run.assert_called_once()  # type: ignore[attr-defined]


def test_runner_persists_profiles_and_playbooks_with_agentic_fields():
    """Happy path: vetted items → persisted with reader_angle / source_span populated."""
    storage = MagicMock()
    result = ExtractionResult(
        profiles=[
            VettedProfile(
                content="User is a Go engineer.",
                time_to_live="infinity",
                source_span="Go engineer",
                reader_angle="facts",
            ),
        ],
        playbooks=[
            VettedPlaybook(
                trigger="scheduling a review",
                content="avoid before 10am",
                rationale="user is on-call",
                reader_angle="behavior",
            ),
        ],
    )
    runner = _make_runner(storage=storage, service_result=result)

    with patch(
        "reflexio.server.services.extraction.agentic_adapter.is_deduplicator_enabled",
        return_value=False,
    ):
        warnings = runner.run(
            publish_request=_make_publish_request(),
            request_id="req_abc",
            new_interactions=[
                _make_interaction(
                    "User", "I'm a senior Go engineer and I prefer postgres for OLTP."
                )
            ],
            new_request=_make_request(),
            config=Config(storage_config=StorageConfigSQLite()),
        )

    assert warnings == []
    storage.add_user_profile.assert_called_once()
    persisted_profiles = storage.add_user_profile.call_args.args[1]
    assert persisted_profiles[0].reader_angle == "facts"
    assert persisted_profiles[0].source_span == "Go engineer"

    storage.save_user_playbooks.assert_called_once()
    persisted_playbooks = storage.save_user_playbooks.call_args.args[0]
    assert persisted_playbooks[0].reader_angle == "behavior"
    assert persisted_playbooks[0].user_id == "u_test"


def test_runner_dedup_invoked_when_feature_flag_enabled():
    result = ExtractionResult(
        profiles=[VettedProfile(content="x", time_to_live="infinity")],
    )
    runner = _make_runner(service_result=result)

    fake_dedup = MagicMock()
    fake_dedup.deduplicate.return_value = ([], ["existing_id_1"], [])
    with (
        patch(
            "reflexio.server.services.extraction.agentic_adapter.is_deduplicator_enabled",
            return_value=True,
        ),
        patch(
            "reflexio.server.services.extraction.agentic_adapter.ProfileDeduplicator",
            return_value=fake_dedup,
        ),
    ):
        runner.run(
            publish_request=_make_publish_request(),
            request_id="req_abc",
            new_interactions=[
                _make_interaction(
                    "User", "Long user message that passes the pre-filter length check"
                )
            ],
            new_request=_make_request(),
            config=Config(storage_config=StorageConfigSQLite()),
        )

    fake_dedup.deduplicate.assert_called_once()


def test_runner_dedup_skipped_when_feature_flag_disabled():
    result = ExtractionResult(
        profiles=[VettedProfile(content="x", time_to_live="infinity")],
    )
    runner = _make_runner(service_result=result)

    with (
        patch(
            "reflexio.server.services.extraction.agentic_adapter.is_deduplicator_enabled",
            return_value=False,
        ),
        patch(
            "reflexio.server.services.extraction.agentic_adapter.ProfileDeduplicator",
        ) as mock_dedup_cls,
    ):
        runner.run(
            publish_request=_make_publish_request(),
            request_id="req_abc",
            new_interactions=[
                _make_interaction(
                    "User", "Long user message that passes the pre-filter length check"
                )
            ],
            new_request=_make_request(),
            config=Config(storage_config=StorageConfigSQLite()),
        )

    mock_dedup_cls.assert_not_called()


def test_runner_aggregation_loops_over_configured_playbooks():
    """Aggregator runs once per playbook config that has aggregation_config."""
    result = ExtractionResult(
        playbooks=[VettedPlaybook(trigger="t", content="c")],
    )
    runner = _make_runner(service_result=result)

    cfg = Config(
        storage_config=StorageConfigSQLite(),
        user_playbook_extractor_configs=[
            UserPlaybookExtractorConfig(
                extractor_name="with_agg",
                extraction_definition_prompt="p",
                aggregation_config=PlaybookAggregatorConfig(),
            ),
            UserPlaybookExtractorConfig(
                extractor_name="without_agg",
                extraction_definition_prompt="p",
            ),
        ],
    )

    fake_agg_cls = MagicMock()
    fake_agg_cls.return_value.run.return_value = {}
    with (
        patch(
            "reflexio.server.services.extraction.agentic_adapter.is_deduplicator_enabled",
            return_value=False,
        ),
        patch(
            "reflexio.server.services.extraction.agentic_adapter.PlaybookAggregator",
            fake_agg_cls,
        ),
    ):
        runner.run(
            publish_request=_make_publish_request(),
            request_id="req_abc",
            new_interactions=[
                _make_interaction(
                    "User", "Long user message that passes the pre-filter length check"
                )
            ],
            new_request=_make_request(),
            config=cfg,
        )

    assert fake_agg_cls.return_value.run.call_count == 1
    aggregator_request = fake_agg_cls.return_value.run.call_args.args[0]
    assert aggregator_request.playbook_name == "with_agg"


def test_runner_skip_aggregation_short_circuits():
    result = ExtractionResult(
        playbooks=[VettedPlaybook(trigger="t", content="c")],
    )
    runner = _make_runner(service_result=result)

    cfg = Config(
        storage_config=StorageConfigSQLite(),
        user_playbook_extractor_configs=[
            UserPlaybookExtractorConfig(
                extractor_name="with_agg",
                extraction_definition_prompt="p",
                aggregation_config=PlaybookAggregatorConfig(),
            ),
        ],
    )

    fake_agg_cls = MagicMock()
    with (
        patch(
            "reflexio.server.services.extraction.agentic_adapter.is_deduplicator_enabled",
            return_value=False,
        ),
        patch(
            "reflexio.server.services.extraction.agentic_adapter.PlaybookAggregator",
            fake_agg_cls,
        ),
    ):
        runner.run(
            publish_request=_make_publish_request(skip_aggregation=True),
            request_id="req_abc",
            new_interactions=[
                _make_interaction(
                    "User", "Long user message that passes the pre-filter length check"
                )
            ],
            new_request=_make_request(),
            config=cfg,
        )

    fake_agg_cls.assert_not_called()


def test_runner_superseded_delete_failure_becomes_warning():
    result = ExtractionResult(
        profiles=[VettedProfile(content="x", time_to_live="infinity")],
    )
    storage = MagicMock()
    storage.delete_user_profile.side_effect = RuntimeError("boom")
    runner = _make_runner(storage=storage, service_result=result)

    fake_dedup = MagicMock()
    fake_dedup.deduplicate.return_value = ([], ["p_dead"], [])
    with (
        patch(
            "reflexio.server.services.extraction.agentic_adapter.is_deduplicator_enabled",
            return_value=True,
        ),
        patch(
            "reflexio.server.services.extraction.agentic_adapter.ProfileDeduplicator",
            return_value=fake_dedup,
        ),
    ):
        warnings = runner.run(
            publish_request=_make_publish_request(),
            request_id="req_abc",
            new_interactions=[
                _make_interaction(
                    "User", "Long user message that passes the pre-filter length check"
                )
            ],
            new_request=_make_request(),
            config=Config(storage_config=StorageConfigSQLite()),
        )

    assert any("delete superseded profile p_dead failed" in w for w in warnings)
    storage.delete_user_profile.assert_called_once()


def test_runner_skipped_result_returns_empty_warnings():
    result = ExtractionResult(skipped_reason="no sessions to extract")
    runner = _make_runner(service_result=result)

    out = runner.run(
        publish_request=_make_publish_request(force_extraction=True),
        request_id="req_abc",
        new_interactions=[
            _make_interaction(
                "User", "Long user message that passes the pre-filter length check"
            )
        ],
        new_request=_make_request(),
        config=Config(storage_config=StorageConfigSQLite()),
    )

    assert out == []


def test_runner_handles_missing_storage_gracefully():
    result = ExtractionResult(
        profiles=[VettedProfile(content="x", time_to_live="infinity")],
    )
    runner = _make_runner(storage=MagicMock(), service_result=result)
    runner.storage = None

    with patch(
        "reflexio.server.services.extraction.agentic_adapter.is_deduplicator_enabled",
        return_value=False,
    ):
        out = runner.run(
            publish_request=_make_publish_request(),
            request_id="req_abc",
            new_interactions=[
                _make_interaction(
                    "User", "Long user message that passes the pre-filter length check"
                )
            ],
            new_request=_make_request(),
            config=Config(storage_config=StorageConfigSQLite()),
        )

    # Returns cleanly with a warning-less list; doesn't crash.
    assert isinstance(out, list)


def test_runner_output_pending_status_propagates_to_persisted_profiles():
    result = ExtractionResult(
        profiles=[VettedProfile(content="x", time_to_live="infinity")],
    )
    storage = MagicMock()
    rc = MagicMock()
    rc.storage = storage
    rc.prompt_manager = MagicMock()
    rc.configurator = MagicMock()
    rc.org_id = "test-org"
    runner = AgenticExtractionRunner(
        llm_client=MagicMock(),
        request_context=rc,
        org_id="test-org",
        output_pending_status=True,
    )
    runner.service = MagicMock()
    runner.service.run.return_value = result

    with patch(
        "reflexio.server.services.extraction.agentic_adapter.is_deduplicator_enabled",
        return_value=False,
    ):
        runner.run(
            publish_request=_make_publish_request(),
            request_id="req_abc",
            new_interactions=[
                _make_interaction(
                    "User", "Long user message that passes the pre-filter length check"
                )
            ],
            new_request=_make_request(),
            config=Config(storage_config=StorageConfigSQLite()),
        )

    persisted = storage.add_user_profile.call_args.args[1]
    assert persisted[0].status == Status.PENDING


@pytest.mark.parametrize(
    "ttl,expected_delta",
    [
        ("one_day", 86_400),
        ("one_month", 30 * 86_400),
        ("one_quarter", 90 * 86_400),
    ],
)
def test_ttl_all_finite_literals_map_correctly(ttl, expected_delta):
    now = 1_700_000_000
    assert _compute_expiration(ttl, now_ts=now) == now + expected_delta


# ---------------- PlaybookDeduplicator wiring ---------------- #


def test_runner_playbook_dedup_invoked_when_feature_flag_enabled():
    """When is_deduplicator_enabled=True, PlaybookDeduplicator runs on agentic playbooks."""
    result = ExtractionResult(
        playbooks=[
            VettedPlaybook(trigger="t1", content="c1"),
            VettedPlaybook(trigger="t2", content="c2"),
        ],
    )
    storage = MagicMock()
    runner = _make_runner(storage=storage, service_result=result)

    fake_dedup = MagicMock()
    fake_dedup.deduplicate.return_value = (
        # Single retained playbook + one superseded ID on disk
        [
            UserPlaybook(
                user_id="u_test",
                agent_version="v1",
                request_id="req_abc",
                content="merged",
            )
        ],
        [42],
    )
    with (
        patch(
            "reflexio.server.services.extraction.agentic_adapter.is_deduplicator_enabled",
            return_value=True,
        ),
        patch(
            "reflexio.server.services.extraction.agentic_adapter.PlaybookDeduplicator",
            return_value=fake_dedup,
        ),
    ):
        runner.run(
            publish_request=_make_publish_request(),
            request_id="req_abc",
            new_interactions=[
                _make_interaction(
                    "User", "Long user message that passes the pre-filter length check"
                )
            ],
            new_request=_make_request(),
            config=Config(storage_config=StorageConfigSQLite()),
        )

    fake_dedup.deduplicate.assert_called_once()
    # Save ran with the deduped set (1 item, not 2)
    assert storage.save_user_playbooks.call_count == 1
    assert len(storage.save_user_playbooks.call_args.args[0]) == 1
    # Superseded ID was deleted AFTER save
    storage.delete_user_playbooks_by_ids.assert_called_once_with([42])


def test_runner_playbook_dedup_skipped_when_feature_flag_disabled():
    """Feature flag off → PlaybookDeduplicator never constructed; raw playbooks persist."""
    result = ExtractionResult(
        playbooks=[VettedPlaybook(trigger="t", content="c")],
    )
    storage = MagicMock()
    runner = _make_runner(storage=storage, service_result=result)

    with (
        patch(
            "reflexio.server.services.extraction.agentic_adapter.is_deduplicator_enabled",
            return_value=False,
        ),
        patch(
            "reflexio.server.services.extraction.agentic_adapter.PlaybookDeduplicator",
        ) as mock_dedup_cls,
    ):
        runner.run(
            publish_request=_make_publish_request(),
            request_id="req_abc",
            new_interactions=[
                _make_interaction(
                    "User", "Long user message that passes the pre-filter length check"
                )
            ],
            new_request=_make_request(),
            config=Config(storage_config=StorageConfigSQLite()),
        )

    mock_dedup_cls.assert_not_called()
    storage.save_user_playbooks.assert_called_once()
    storage.delete_user_playbooks_by_ids.assert_not_called()


def test_runner_playbook_dedup_passes_extractor_config_dedup_config():
    """dedup_config should be pulled from the first extractor config that has one."""
    from reflexio.models.config_schema import (
        DeduplicationConfig,
        UserPlaybookExtractorConfig,
    )

    result = ExtractionResult(
        playbooks=[VettedPlaybook(trigger="t", content="c")],
    )
    runner = _make_runner(service_result=result)

    expected_cfg = DeduplicationConfig(search_threshold=0.42)
    user_cfgs = [
        UserPlaybookExtractorConfig(
            extractor_name="no_dedup",
            extraction_definition_prompt="p",
        ),
        UserPlaybookExtractorConfig(
            extractor_name="with_dedup",
            extraction_definition_prompt="p",
            deduplication_config=expected_cfg,
        ),
    ]
    cfg = Config(
        storage_config=StorageConfigSQLite(),
        user_playbook_extractor_configs=user_cfgs,
    )

    constructed_kwargs = {}

    def fake_ctor(*args, **kwargs):
        constructed_kwargs.update(kwargs)
        m = MagicMock()
        m.deduplicate.return_value = ([], [])
        return m

    with (
        patch(
            "reflexio.server.services.extraction.agentic_adapter.is_deduplicator_enabled",
            return_value=True,
        ),
        patch(
            "reflexio.server.services.extraction.agentic_adapter.PlaybookDeduplicator",
            side_effect=fake_ctor,
        ),
    ):
        runner.run(
            publish_request=_make_publish_request(),
            request_id="req_abc",
            new_interactions=[
                _make_interaction(
                    "User", "Long user message that passes the pre-filter length check"
                )
            ],
            new_request=_make_request(),
            config=cfg,
        )

    assert constructed_kwargs.get("dedup_config") is expected_cfg


def test_runner_playbook_dedup_delete_failure_surfaces_as_warning():
    """Delete failure after save → warning, publish still returns."""
    result = ExtractionResult(
        playbooks=[VettedPlaybook(trigger="t", content="c")],
    )
    storage = MagicMock()
    storage.delete_user_playbooks_by_ids.side_effect = RuntimeError("delete boom")
    runner = _make_runner(storage=storage, service_result=result)

    fake_dedup = MagicMock()
    fake_dedup.deduplicate.return_value = (
        [
            UserPlaybook(
                user_id="u_test",
                agent_version="v1",
                request_id="req_abc",
                content="merged",
            )
        ],
        [99],
    )
    with (
        patch(
            "reflexio.server.services.extraction.agentic_adapter.is_deduplicator_enabled",
            return_value=True,
        ),
        patch(
            "reflexio.server.services.extraction.agentic_adapter.PlaybookDeduplicator",
            return_value=fake_dedup,
        ),
    ):
        warnings = runner.run(
            publish_request=_make_publish_request(),
            request_id="req_abc",
            new_interactions=[
                _make_interaction(
                    "User", "Long user message that passes the pre-filter length check"
                )
            ],
            new_request=_make_request(),
            config=Config(storage_config=StorageConfigSQLite()),
        )

    assert any("delete superseded playbooks failed" in w for w in warnings)
    storage.save_user_playbooks.assert_called_once()


def test_runner_playbook_dedup_failure_falls_back_to_raw_list():
    """If PlaybookDeduplicator raises, the raw playbooks are still saved + warning recorded."""
    vpb = VettedPlaybook(trigger="t", content="c")
    result = ExtractionResult(playbooks=[vpb])
    storage = MagicMock()
    runner = _make_runner(storage=storage, service_result=result)

    fake_dedup = MagicMock()
    fake_dedup.deduplicate.side_effect = RuntimeError("dedup boom")
    with (
        patch(
            "reflexio.server.services.extraction.agentic_adapter.is_deduplicator_enabled",
            return_value=True,
        ),
        patch(
            "reflexio.server.services.extraction.agentic_adapter.PlaybookDeduplicator",
            return_value=fake_dedup,
        ),
    ):
        warnings = runner.run(
            publish_request=_make_publish_request(),
            request_id="req_abc",
            new_interactions=[
                _make_interaction(
                    "User", "Long user message that passes the pre-filter length check"
                )
            ],
            new_request=_make_request(),
            config=Config(storage_config=StorageConfigSQLite()),
        )

    assert any("playbook deduplicator failed" in w for w in warnings)
    # Raw playbook still got saved despite the dedup failure
    storage.save_user_playbooks.assert_called_once()
    assert len(storage.save_user_playbooks.call_args.args[0]) == 1
