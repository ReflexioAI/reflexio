"""
Unit tests for PlaybookAggregator private helpers and run() orchestration.

Targets coverage gaps in:
- _should_run_aggregation (reaggregation_trigger_count defaults, threshold logic)
- _determine_cluster_changes (no previous clusters, fingerprint match/mismatch)
- _update_operation_state (empty list, normal update)
- _get_playbook_aggregator_config (match, no match, no configs)
- _compute_cluster_fingerprint (deterministic, order-independent)
- run() (rerun mode, no user playbooks, incremental no changes, save exception,
         full archive delete path, incremental archive delete)
"""

import logging
import re
from typing import Any
from unittest.mock import ANY, MagicMock, call, patch

import pytest

from reflexio.models.api_schema.service_schemas import (
    AgentPlaybook,
    PlaybookStatus,
    UserPlaybook,
)
from reflexio.models.config_schema import (
    SINGLETON_USER_PLAYBOOK_NAME,
    PlaybookAggregatorConfig,
    PlaybookConfig,
)
from reflexio.server.services.playbook.aggregation_prompt_processing import (
    AggregationPromptProcessingContext,
    PromptPostprocessResult,
    PromptPreprocessResult,
)
from reflexio.server.services.playbook.components.aggregator import PlaybookAggregator
from reflexio.server.services.playbook.playbook_service_utils import (
    PlaybookAggregationOutput,
    PlaybookAggregatorRequest,
    StructuredPlaybookContent,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_aggregator(
    storage: MagicMock | None = None,
    configurator: MagicMock | None = None,
    aggregation_prompt_processor: Any | None = None,
) -> Any:
    """Build an aggregator with fully mocked dependencies."""
    llm = MagicMock()
    ctx = MagicMock()
    ctx.storage = storage or MagicMock()
    ctx.configurator = configurator or MagicMock()
    ctx.org_id = "test-org"
    return PlaybookAggregator(
        llm_client=llm,
        request_context=ctx,
        agent_version="v1",
        aggregation_prompt_processor=aggregation_prompt_processor,
    )


def _raw(
    rid: int = 1,
    name: str = "test_fb",
    when: str | None = "when cond",
    do: str | None = "do action",
    dont: str | None = None,
) -> UserPlaybook:
    return UserPlaybook(
        user_playbook_id=rid,
        agent_version="v1",
        request_id=f"req-{rid}",
        playbook_name=name,
        content=f"content-{rid}",
        trigger=when,
    )


def _agent_playbook(
    fid: int = 1, name: str = "test_fb", content: str = "c"
) -> AgentPlaybook:
    return AgentPlaybook(
        agent_playbook_id=fid,
        playbook_name=name,
        agent_version="v1",
        content=content,
        playbook_status=PlaybookStatus.PENDING,
    )


# ---------------------------------------------------------------------------
# Aggregation prompt processing seam
# ---------------------------------------------------------------------------


class _MappingAwareProcessor:
    prompt_extra_instructions: str | None = None
    _OUTPUT_MARKER_RE = re.compile(r"<<TOKEN_\d+>>")

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def preprocess_prompt_text(
        self,
        text: str,
        *,
        shared_state: dict[str, Any] | None = None,
        context: AggregationPromptProcessingContext | None = None,  # noqa: ARG002
    ) -> PromptPreprocessResult:
        assert shared_state is not None
        original = text
        if "Project Zephyr" in text:
            shared_state.setdefault("zephyr", len(shared_state) + 1)
            text = text.replace("Project Zephyr", f"<<TOKEN_{shared_state['zephyr']}>>")
        if "Project Atlas" in text:
            shared_state.setdefault("atlas", len(shared_state) + 1)
            text = text.replace("Project Atlas", f"<<TOKEN_{shared_state['atlas']}>>")
        if "zephyr-access-code" in text:
            shared_state.setdefault("access_code", len(shared_state) + 1)
            text = text.replace(
                "zephyr-access-code", f"<<TOKEN_{shared_state['access_code']}>>"
            )
        if "handoff-window" in text:
            shared_state.setdefault("handoff_window", len(shared_state) + 1)
            text = text.replace(
                "handoff-window", f"<<TOKEN_{shared_state['handoff_window']}>>"
            )
        self.calls.append((text, id(shared_state)))
        return PromptPreprocessResult(text=text, changed=text != original)

    def prompt_instructions(
        self,
        context: AggregationPromptProcessingContext,  # noqa: ARG002
    ) -> str | None:
        return self.prompt_extra_instructions

    def _postprocess_text(
        self,
        text: str | None,
    ) -> tuple[str | None, int]:
        if text is None:
            return None, 0
        marker_count = len(self._OUTPUT_MARKER_RE.findall(text))
        return self._OUTPUT_MARKER_RE.sub("a processed artifact", text), marker_count

    def postprocess_aggregation_output(
        self,
        value: object,
        *,
        context: AggregationPromptProcessingContext | None = None,  # noqa: ARG002
    ) -> PromptPostprocessResult:
        processed, count = self._postprocess_value(value)
        return PromptPostprocessResult(value=processed, artifacts_removed=count)

    def _postprocess_value(self, value: object) -> tuple[object, int]:
        if isinstance(value, str):
            return self._postprocess_text(value)
        if isinstance(value, PlaybookAggregationOutput):
            if value.playbook is None:
                return value, 0
            updates: dict[str, str | None] = {}
            total = 0
            for field_name, field_value in value.playbook.model_dump().items():
                if not isinstance(field_value, str):
                    continue
                processed, count = self._postprocess_text(field_value)
                total += count
                if processed != field_value:
                    updates[field_name] = processed
            if not updates:
                return value, 0
            return value.model_copy(
                update={"playbook": value.playbook.model_copy(update=updates)}
            ), total
        if isinstance(value, dict):
            processed_dict: dict[object, object] = {}
            total = 0
            for key, item in value.items():
                processed_key, key_count = self._postprocess_value(key)
                processed_item, item_count = self._postprocess_value(item)
                processed_dict[processed_key] = processed_item
                total += key_count + item_count
            return processed_dict, total
        if isinstance(value, list):
            processed_items: list[object] = []
            total = 0
            for item in value:
                processed_item, count = self._postprocess_value(item)
                processed_items.append(processed_item)
                total += count
            return processed_items, total
        if isinstance(value, tuple):
            processed_items: list[object] = []
            total = 0
            for item in value:
                processed_item, count = self._postprocess_value(item)
                processed_items.append(processed_item)
                total += count
            return tuple(processed_items), total
        return value, 0


def test_aggregation_prompt_processing_protocol_types_importable():
    from reflexio.server.services.configurator.configurator import DefaultConfigurator
    from reflexio.server.services.playbook.aggregation_prompt_processing import (
        PassthroughPromptProcessor,
        PromptPostprocessResult,
        PromptPreprocessResult,
    )

    processor = PassthroughPromptProcessor()
    context = AggregationPromptProcessingContext()
    preprocess_result = processor.preprocess_prompt_text(
        "keep this",
        shared_state={},
        context=context,
    )
    postprocess_result = processor.postprocess_aggregation_output(
        {"content": "keep this"},
        context=context,
    )

    assert preprocess_result == PromptPreprocessResult(text="keep this")
    assert postprocess_result == PromptPostprocessResult(value={"content": "keep this"})
    assert context.changed is False
    assert (
        DefaultConfigurator(org_id="test-org").create_aggregation_prompt_processor()
        is None
    )


def test_aggregation_prompt_processor_preprocesses_cluster_playbooks_but_not_existing_agent_playbooks():
    processor = _MappingAwareProcessor()
    agg = _make_aggregator(aggregation_prompt_processor=processor)
    captured_messages: list[dict[str, str]] = []
    captured_variables: dict[str, str] = {}

    def render_prompt(_prompt_id: str, variables: dict[str, str]) -> str:
        captured_variables.update(variables)
        return (
            f"{variables['user_playbooks']}\n\nEXISTING:\n"
            f"{variables['existing_approved_playbooks']}"
        )

    agg.request_context.prompt_manager.render_prompt.side_effect = render_prompt
    agg.client.generate_chat_response.side_effect = lambda messages, **_kwargs: (
        captured_messages.extend(messages)
        or PlaybookAggregationOutput(
            playbook=StructuredPlaybookContent(
                content="Keep the shared operational rule.",
                trigger="When the shared workflow appears.",
                rationale="The rule is common across users.",
            )
        )
    )
    clusters = {
        0: [
            UserPlaybook(
                user_playbook_id=1,
                agent_version="v1",
                request_id="req-1",
                playbook_name="test_fb",
                content="Project Zephyr prefers the safety checklist.",
                trigger="When Project Zephyr opens a ticket.",
                rationale="Project Zephyr missed one step.",
            ),
            UserPlaybook(
                user_playbook_id=2,
                agent_version="v1",
                request_id="req-2",
                playbook_name="test_fb",
                content="Project Atlas asks for the same checklist.",
                trigger="When Project Atlas opens a ticket.",
                rationale="Project Atlas missed the same step.",
            ),
        ]
    }
    existing = [
        AgentPlaybook(
            agent_playbook_id=7,
            playbook_name="test_fb",
            agent_version="v1",
            content="Project Zephyr already has a checklist playbook.",
            playbook_status=PlaybookStatus.PENDING,
        )
    ]

    with patch.dict("os.environ", {"MOCK_LLM_RESPONSE": ""}):
        result = agg._generate_playbooks_with_source_clusters(clusters, existing)

    assert len(result) == 1
    rendered_prompt = captured_messages[0]["content"]
    user_prompt = captured_variables["user_playbooks"]
    existing_prompt = captured_variables["existing_approved_playbooks"]
    assert "Project Zephyr" not in user_prompt
    assert "Project Atlas" not in user_prompt
    assert "<<TOKEN_1>>" in rendered_prompt
    assert "<<TOKEN_2>>" in rendered_prompt
    assert "Project Zephyr already has a checklist playbook." in existing_prompt
    assert len({mapping_id for _text, mapping_id in processor.calls}) == 1
    assert clusters[0][0].content == "Project Zephyr prefers the safety checklist."
    assert existing[0].content == "Project Zephyr already has a checklist playbook."


def test_aggregation_prompt_processor_preprocesses_grouped_prompt_input():
    processor = _MappingAwareProcessor()
    agg = _make_aggregator(aggregation_prompt_processor=processor)
    captured_prompts: list[str] = []

    agg.request_context.prompt_manager.render_prompt.side_effect = (
        lambda _prompt_id, variables: (
            captured_prompts.append(variables["user_playbooks"])
            or variables["user_playbooks"]
        )
    )
    agg.client.generate_chat_response.return_value = PlaybookAggregationOutput(
        playbook=StructuredPlaybookContent(
            content="- Keep both unrelated operational rules.",
            trigger="When either operational condition applies.",
            rationale="The grouped prompt preserves both source groups.",
        )
    )
    clusters = {
        0: [
            UserPlaybook(
                user_playbook_id=1,
                agent_version="v1",
                request_id="req-1",
                playbook_name="test_fb",
                content="Project Zephyr checks deployment readiness.",
                trigger="When Project Zephyr reviews deployment readiness.",
                rationale="Project Zephyr owns the deployment checklist.",
            ),
            UserPlaybook(
                user_playbook_id=2,
                agent_version="v1",
                request_id="req-2",
                playbook_name="test_fb",
                content="Project Atlas audits billing anomalies.",
                trigger="When Project Atlas reviews billing anomalies.",
                rationale="Project Atlas owns the billing review.",
            ),
        ]
    }

    with patch.dict("os.environ", {"MOCK_LLM_RESPONSE": ""}):
        result = agg._generate_playbooks_with_source_clusters(clusters, [])

    assert len(result) == 1
    rendered_prompt = captured_prompts[0]
    assert "Group 1" in rendered_prompt
    assert "Project Zephyr" not in rendered_prompt
    assert "Project Atlas" not in rendered_prompt
    assert "<<TOKEN_1>>" in rendered_prompt
    assert "<<TOKEN_2>>" in rendered_prompt


def test_mock_llm_response_postprocesses_artifacts_before_storage():
    processor = _MappingAwareProcessor()
    agg = _make_aggregator(aggregation_prompt_processor=processor)
    clusters = {
        0: [
            UserPlaybook(
                user_playbook_id=1,
                agent_version="v1",
                request_id="req-1",
                playbook_name="test_fb",
                content="Project Zephyr prefers the safety checklist for zephyr-access-code and handoff-window.",
                trigger="When Project Zephyr opens a ticket with handoff-window.",
                rationale="Project Zephyr missed one step for zephyr-access-code.",
            )
        ]
    }

    with patch.dict("os.environ", {"MOCK_LLM_RESPONSE": "true"}):
        result = agg._generate_playbooks_with_source_clusters(clusters, [])

    assert len(result) == 1
    playbook, _sources = result[0]
    assert "<<TOKEN_" not in playbook.content
    assert "<<TOKEN_" not in (playbook.trigger or "")
    assert "a processed artifact" in playbook.content


def test_artifact_postprocessing_runs_before_response_logging_and_storage():
    agg = _make_aggregator(aggregation_prompt_processor=_MappingAwareProcessor())
    agg.request_context.prompt_manager.render_prompt.return_value = "prompt"
    raw_response = PlaybookAggregationOutput(
        playbook=StructuredPlaybookContent(
            content="- Ask <<TOKEN_1>> to confirm via <<TOKEN_2>>.",
            trigger="When <<TOKEN_3>> requests access.",
            rationale="<<TOKEN_1>>, <<TOKEN_2>>, and <<TOKEN_3>> all hit this case.",
        )
    )
    agg.client.generate_chat_response.return_value = raw_response
    cluster = [_raw(rid=1)]

    with (
        patch(
            "reflexio.server.services.playbook.components.aggregator.log_model_response"
        ) as mock_log_model_response,
        patch.dict("os.environ", {"MOCK_LLM_RESPONSE": ""}),
    ):
        result = agg._generate_playbook_from_cluster(cluster, "None")

    assert result is not None
    assert "<<TOKEN_" not in result.content
    assert "<<TOKEN_" not in (result.trigger or "")
    assert "<<TOKEN_" not in (result.rationale or "")
    assert "a processed artifact" in result.content
    assert "a processed artifact" in (result.trigger or "")
    logged_response = mock_log_model_response.call_args.args[2]
    assert isinstance(logged_response, PlaybookAggregationOutput)
    assert logged_response.playbook is not None
    assert "<<TOKEN_" not in (logged_response.playbook.content or "")
    assert "<<TOKEN_" not in (logged_response.playbook.trigger or "")
    assert "<<TOKEN_" not in (logged_response.playbook.rationale or "")


def test_artifact_postprocessing_runs_before_string_fallback_logging():
    agg = _make_aggregator(aggregation_prompt_processor=_MappingAwareProcessor())
    agg.request_context.prompt_manager.render_prompt.return_value = "prompt"
    agg.client.generate_chat_response.return_value = "invalid <<TOKEN_7>> response"
    cluster = [_raw(rid=1)]

    with (
        patch(
            "reflexio.server.services.playbook.components.aggregator.log_model_response"
        ) as mock_log_model_response,
        patch.dict("os.environ", {"MOCK_LLM_RESPONSE": ""}),
    ):
        result = agg._generate_playbook_from_cluster(cluster, "None")

    assert result is None
    logged_response = mock_log_model_response.call_args.args[2]
    assert logged_response == "invalid a processed artifact response"


def test_artifact_postprocessing_runs_before_dict_fallback_logging():
    agg = _make_aggregator(aggregation_prompt_processor=_MappingAwareProcessor())
    agg.request_context.prompt_manager.render_prompt.return_value = "prompt"
    agg.client.generate_chat_response.return_value = {
        "playbook": {
            "content": "Ask <<TOKEN_1>> to confirm.",
            "rationale": ["<<TOKEN_2>> saw this before."],
        },
        "<<TOKEN_3>>": "key should not leak either",
    }
    cluster = [_raw(rid=1)]

    with (
        patch(
            "reflexio.server.services.playbook.components.aggregator.log_model_response"
        ) as mock_log_model_response,
        patch.dict("os.environ", {"MOCK_LLM_RESPONSE": ""}),
    ):
        result = agg._generate_playbook_from_cluster(cluster, "None")

    assert result is None
    logged_response = mock_log_model_response.call_args.args[2]
    assert "<<TOKEN_" not in repr(logged_response)
    assert "a processed artifact" in repr(logged_response)


def test_artifact_postprocessing_runs_before_nested_sequence_logging():
    agg = _make_aggregator(aggregation_prompt_processor=_MappingAwareProcessor())

    processed, artifact_count = agg._postprocess_aggregation_output(
        (
            "Ask <<TOKEN_1>> to confirm.",
            ["Notify <<TOKEN_2>>.", {"owner": "<<TOKEN_3>>"}],
        )
    )

    assert artifact_count == 3
    assert "<<TOKEN_" not in repr(processed)
    assert "a processed artifact" in repr(processed)


def test_artifact_postprocessing_warning_does_not_write_usage_event(caplog):
    agg = _make_aggregator()

    with (
        caplog.at_level(
            logging.WARNING,
            logger="reflexio.server.services.playbook.components.aggregator",
        ),
        patch(
            "reflexio.server.services.playbook.components.aggregator.record_usage_event"
        ) as mock_record_usage_event,
    ):
        agg._record_postprocessing_artifacts(3)

    assert "Post-processed 3 residual artifacts" in caplog.text
    mock_record_usage_event.assert_not_called()


def test_artifact_postprocessing_runs_before_exception_logging(caplog):
    agg = _make_aggregator(aggregation_prompt_processor=_MappingAwareProcessor())
    agg.request_context.prompt_manager.render_prompt.return_value = "prompt"
    agg.client.generate_chat_response.side_effect = RuntimeError(
        "failed after <<TOKEN_9>> appeared in parse error"
    )
    cluster = [_raw(rid=1)]

    with (
        caplog.at_level(
            logging.ERROR,
            logger="reflexio.server.services.playbook.components.aggregator",
        ),
        patch.dict("os.environ", {"MOCK_LLM_RESPONSE": ""}),
    ):
        result = agg._generate_playbook_from_cluster(cluster, "None")

    assert result is None
    assert "<<TOKEN_" not in caplog.text
    assert "a processed artifact" in caplog.text


# ---------------------------------------------------------------------------
# _should_run_aggregation
# ---------------------------------------------------------------------------


class TestShouldRunAggregation:
    """Tests for _should_run_aggregation."""

    def test_reaggregation_trigger_count_zero_defaults_to_two(self):
        """When reaggregation_trigger_count <= 0 the method should default to 2."""
        agg = _make_aggregator()
        # Bypass Pydantic ge=1 validation to hit the <= 0 guard in source
        config = PlaybookAggregatorConfig.model_construct(
            min_cluster_size=2, reaggregation_trigger_count=0
        )
        agg.storage.count_user_playbooks.return_value = 2

        result = agg._should_run_aggregation("fb", config)

        assert result is True
        # count >= default(2) -> True

    def test_reaggregation_trigger_count_negative_defaults_to_two(self):
        """Negative reaggregation_trigger_count also defaults to 2."""
        agg = _make_aggregator()
        # Bypass Pydantic ge=1 validation to hit the <= 0 guard in source
        config = PlaybookAggregatorConfig.model_construct(
            min_cluster_size=2, reaggregation_trigger_count=-1
        )
        agg.storage.count_user_playbooks.return_value = 2

        result = agg._should_run_aggregation("fb", config)

        assert result is True

    def test_enough_new_playbooks_returns_true(self):
        agg = _make_aggregator()
        config = PlaybookAggregatorConfig(
            min_cluster_size=2, reaggregation_trigger_count=3
        )
        agg.storage.count_user_playbooks.return_value = 5

        assert agg._should_run_aggregation("fb", config) is True

    def test_not_enough_new_playbooks_returns_false(self):
        agg = _make_aggregator()
        config = PlaybookAggregatorConfig(
            min_cluster_size=2, reaggregation_trigger_count=3
        )
        agg.storage.count_user_playbooks.return_value = 1

        assert agg._should_run_aggregation("fb", config) is False

    def test_rerun_flag_passed_to_count(self):
        """rerun=True should be forwarded so all playbooks are counted."""
        agg = _make_aggregator()
        config = PlaybookAggregatorConfig(
            min_cluster_size=2, reaggregation_trigger_count=2
        )
        agg.storage.count_user_playbooks.return_value = 10

        agg._should_run_aggregation("fb", config, rerun=True)

        # rerun=True -> last_processed_id=0
        call_kwargs = agg.storage.count_user_playbooks.call_args
        assert (
            call_kwargs.kwargs.get("min_user_playbook_id") == 0
            or call_kwargs[1].get("min_user_playbook_id") == 0
        )


# ---------------------------------------------------------------------------
# _get_new_user_playbooks_count
# ---------------------------------------------------------------------------


class TestGetNewUserPlaybooksCount:
    def test_rerun_uses_zero_as_last_processed(self):
        agg = _make_aggregator()
        agg.storage.count_user_playbooks.return_value = 7

        result = agg._get_new_user_playbooks_count("fb", rerun=True)

        assert result == 7
        assert (
            agg.storage.count_user_playbooks.call_args.kwargs["min_user_playbook_id"]
            == 0
        )

    def test_non_rerun_reads_bookmark(self):
        agg = _make_aggregator()
        agg.storage.count_user_playbooks.return_value = 3

        with patch.object(PlaybookAggregator, "_create_state_manager") as mock_csm:
            mgr = MagicMock()
            mgr.get_aggregator_bookmark.return_value = 42
            mock_csm.return_value = mgr

            result = agg._get_new_user_playbooks_count("fb", rerun=False)

        assert result == 3
        assert (
            agg.storage.count_user_playbooks.call_args.kwargs["min_user_playbook_id"]
            == 42
        )

    def test_non_rerun_bookmark_none_defaults_to_zero(self):
        agg = _make_aggregator()
        agg.storage.count_user_playbooks.return_value = 5

        with patch.object(PlaybookAggregator, "_create_state_manager") as mock_csm:
            mgr = MagicMock()
            mgr.get_aggregator_bookmark.return_value = None
            mock_csm.return_value = mgr

            result = agg._get_new_user_playbooks_count("fb", rerun=False)

        assert result == 5
        assert (
            agg.storage.count_user_playbooks.call_args.kwargs["min_user_playbook_id"]
            == 0
        )


# ---------------------------------------------------------------------------
# _update_operation_state
# ---------------------------------------------------------------------------


class TestUpdateOperationState:
    def test_empty_list_returns_early(self):
        agg = _make_aggregator()
        agg._update_operation_state("fb", [])
        # No state manager interaction expected

    def test_updates_with_max_id(self):
        agg = _make_aggregator()
        raws = [_raw(rid=3), _raw(rid=10), _raw(rid=7)]

        with patch.object(PlaybookAggregator, "_create_state_manager") as mock_csm:
            mgr = MagicMock()
            mock_csm.return_value = mgr

            agg._update_operation_state("fb", raws)

        mgr.update_aggregator_bookmark.assert_called_once_with(
            name="fb", version="v1", last_processed_id=10
        )


# ---------------------------------------------------------------------------
# _compute_cluster_fingerprint
# ---------------------------------------------------------------------------


class TestComputeClusterFingerprint:
    def test_deterministic(self):
        raws = [_raw(rid=1), _raw(rid=2), _raw(rid=3)]
        fp1 = PlaybookAggregator._compute_cluster_fingerprint(raws)
        fp2 = PlaybookAggregator._compute_cluster_fingerprint(raws)
        assert fp1 == fp2

    def test_order_independent(self):
        raws_a = [_raw(rid=1), _raw(rid=3), _raw(rid=2)]
        raws_b = [_raw(rid=3), _raw(rid=1), _raw(rid=2)]
        assert PlaybookAggregator._compute_cluster_fingerprint(
            raws_a
        ) == PlaybookAggregator._compute_cluster_fingerprint(raws_b)

    def test_different_ids_produce_different_fingerprint(self):
        fp_a = PlaybookAggregator._compute_cluster_fingerprint([_raw(rid=1)])
        fp_b = PlaybookAggregator._compute_cluster_fingerprint([_raw(rid=2)])
        assert fp_a != fp_b

    def test_fingerprint_length(self):
        fp = PlaybookAggregator._compute_cluster_fingerprint([_raw(rid=1)])
        assert len(fp) == 16


# ---------------------------------------------------------------------------
# _determine_cluster_changes
# ---------------------------------------------------------------------------


class TestDetermineClusterChanges:
    def test_no_previous_fingerprints(self):
        """Empty prev_fingerprints => all clusters are changed, none to archive."""
        agg = _make_aggregator()
        clusters = {0: [_raw(rid=1), _raw(rid=2)]}

        changed, to_archive = agg._determine_cluster_changes(clusters, {})

        assert changed == clusters
        assert to_archive == []

    def test_fingerprint_match_no_changes(self):
        """Matching fingerprint => no changed clusters, none to archive."""
        agg = _make_aggregator()
        raws = [_raw(rid=1), _raw(rid=2)]
        clusters = {0: raws}
        fp = PlaybookAggregator._compute_cluster_fingerprint(raws)
        prev = {fp: {"agent_playbook_id": 10, "user_playbook_ids": [1, 2]}}

        changed, to_archive = agg._determine_cluster_changes(clusters, prev)

        assert changed == {}
        assert to_archive == []

    def test_fingerprint_mismatch_detects_change(self):
        """New fingerprint => cluster is changed; old fingerprint archived."""
        agg = _make_aggregator()
        raws_new = [_raw(rid=1), _raw(rid=2), _raw(rid=3)]
        clusters = {0: raws_new}
        prev = {"old_fp_hash": {"agent_playbook_id": 5, "user_playbook_ids": [1, 2]}}

        changed, to_archive = agg._determine_cluster_changes(clusters, prev)

        assert 0 in changed
        assert 5 in to_archive

    def test_disappeared_cluster_with_no_playbook_id(self):
        """Disappeared fingerprint with agent_playbook_id=None should not be archived."""
        agg = _make_aggregator()
        clusters = {0: [_raw(rid=99)]}
        prev = {"gone_fp": {"agent_playbook_id": None, "user_playbook_ids": [1]}}

        changed, to_archive = agg._determine_cluster_changes(clusters, prev)

        assert 0 in changed
        assert to_archive == []

    def test_multiple_clusters_mixed(self):
        """Some clusters match, some do not."""
        agg = _make_aggregator()
        raws_unchanged = [_raw(rid=1)]
        raws_new = [_raw(rid=5), _raw(rid=6)]
        clusters = {0: raws_unchanged, 1: raws_new}

        fp_unchanged = PlaybookAggregator._compute_cluster_fingerprint(raws_unchanged)
        prev = {
            fp_unchanged: {"agent_playbook_id": 10, "user_playbook_ids": [1]},
            "vanished_fp": {"agent_playbook_id": 20, "user_playbook_ids": [2, 3]},
        }

        changed, to_archive = agg._determine_cluster_changes(clusters, prev)

        assert 0 not in changed
        assert 1 in changed
        assert 20 in to_archive


# ---------------------------------------------------------------------------
# _get_playbook_aggregator_config
# ---------------------------------------------------------------------------


class TestGetPlaybookAggregatorConfig:
    def test_returns_matching_config(self):
        agg = _make_aggregator()
        fac = PlaybookAggregatorConfig(
            min_cluster_size=3, reaggregation_trigger_count=5
        )
        afc = PlaybookConfig(
            extractor_name="my_fb",
            extraction_definition_prompt="prompt",
            aggregation_config=fac,
        )
        agg.configurator.get_config.return_value.user_playbook_extractor_config = afc

        result = agg._get_playbook_aggregator_config()

        assert result is fac

    def test_returns_config_without_name_matching(self):
        agg = _make_aggregator()
        fac = PlaybookAggregatorConfig(
            min_cluster_size=3, reaggregation_trigger_count=5
        )
        afc = PlaybookConfig(
            extractor_name="other",
            extraction_definition_prompt="prompt",
            aggregation_config=fac,
        )
        agg.configurator.get_config.return_value.user_playbook_extractor_config = afc

        assert agg._get_playbook_aggregator_config() is fac

    def test_returns_none_when_no_playbook_configs(self):
        agg = _make_aggregator()
        agg.configurator.get_config.return_value.user_playbook_extractor_config = None

        assert agg._get_playbook_aggregator_config() is None


# ---------------------------------------------------------------------------
# run() orchestration
# ---------------------------------------------------------------------------


class TestRun:
    """Tests for the top-level run() method using mocks."""

    def _make_runnable_aggregator(self):
        """Return an aggregator wired for a successful run()."""
        agg = _make_aggregator()
        # config
        fac = PlaybookAggregatorConfig(
            min_cluster_size=2, reaggregation_trigger_count=2
        )
        afc = PlaybookConfig(
            extractor_name="fb",
            extraction_definition_prompt="prompt",
            aggregation_config=fac,
        )
        agg.configurator.get_config.return_value.user_playbook_extractor_config = afc
        # storage returns
        agg.storage.count_user_playbooks.return_value = 5
        agg.storage.get_agent_playbooks.return_value = []
        agg.storage.get_user_playbooks.return_value = [_raw(rid=1), _raw(rid=2)]
        agg.storage.save_agent_playbook_with_aggregate_event.return_value = (
            _agent_playbook(fid=100)
        )
        return agg

    def test_no_config_returns_early(self):
        agg = _make_aggregator()
        agg.configurator.get_config.return_value.user_playbook_extractor_config = None

        req = PlaybookAggregatorRequest(agent_version="v1")
        agg.run(req)

        agg.storage.get_user_playbooks.assert_not_called()

    def test_min_threshold_below_two_returns_early(self):
        agg = _make_aggregator()
        fac = PlaybookAggregatorConfig(
            min_cluster_size=1, reaggregation_trigger_count=2
        )
        afc = PlaybookConfig(
            extractor_name="fb",
            extraction_definition_prompt="prompt",
            aggregation_config=fac,
        )
        agg.configurator.get_config.return_value.user_playbook_extractor_config = afc

        req = PlaybookAggregatorRequest(agent_version="v1")
        agg.run(req)

        agg.storage.get_user_playbooks.assert_not_called()

    def test_not_enough_new_playbooks_skips(self):
        agg = _make_aggregator()
        fac = PlaybookAggregatorConfig(
            min_cluster_size=2, reaggregation_trigger_count=10
        )
        afc = PlaybookConfig(
            extractor_name="fb",
            extraction_definition_prompt="prompt",
            aggregation_config=fac,
        )
        agg.configurator.get_config.return_value.user_playbook_extractor_config = afc
        agg.storage.count_user_playbooks.return_value = 1

        req = PlaybookAggregatorRequest(agent_version="v1")
        agg.run(req)

        agg.storage.get_user_playbooks.assert_not_called()

    @patch.object(PlaybookAggregator, "get_clusters")
    @patch.object(PlaybookAggregator, "_generate_playbooks_with_source_clusters")
    def test_rerun_mode_archives_all(self, mock_gen, mock_clust):
        """rerun=True should call archive_agent_playbooks_by_playbook_name."""
        agg = self._make_runnable_aggregator()
        raws = [_raw(rid=1)]
        mock_clust.return_value = {0: raws}
        mock_gen.return_value = [(_agent_playbook(fid=100), raws)]
        agg.storage.save_agent_playbook_with_aggregate_event.return_value = (
            _agent_playbook(fid=100)
        )

        req = PlaybookAggregatorRequest(agent_version="v1", rerun=True)
        agg.run(req)

        agg.storage.archive_agent_playbooks_by_playbook_name.assert_has_calls(
            [
                call(SINGLETON_USER_PLAYBOOK_NAME, agent_version="v1"),
                call("test_fb", agent_version="v1"),
            ],
            any_order=True,
        )

    @patch.object(PlaybookAggregator, "get_clusters")
    @patch.object(PlaybookAggregator, "_generate_playbooks_with_source_clusters")
    def test_rerun_supersedes_archived_playbooks_after_success(
        self, mock_gen, mock_clust
    ):
        """After successful rerun, supersede_agent_playbooks_by_playbook_name is called (always soft)."""
        agg = self._make_runnable_aggregator()
        raws = [_raw(rid=1)]
        mock_clust.return_value = {0: raws}
        mock_gen.return_value = [(_agent_playbook(fid=100), raws)]
        agg.storage.save_agent_playbook_with_aggregate_event.return_value = (
            _agent_playbook(fid=100)
        )

        req = PlaybookAggregatorRequest(agent_version="v1", rerun=True)
        agg.run(req)

        agg.storage.supersede_agent_playbooks_by_playbook_name.assert_has_calls(
            [
                call(SINGLETON_USER_PLAYBOOK_NAME, agent_version="v1", request_id=ANY),
                call("test_fb", agent_version="v1", request_id=ANY),
            ],
            any_order=True,
        )
        agg.storage.delete_archived_agent_playbooks_by_playbook_name.assert_not_called()

    @patch.object(PlaybookAggregator, "get_clusters")
    @patch.object(PlaybookAggregator, "_generate_playbooks_with_source_clusters")
    def test_first_run_no_prev_fingerprints_full_archive(self, mock_gen, mock_clust):
        """First run (no previous fingerprints) triggers full archive."""
        agg = self._make_runnable_aggregator()
        raws = [_raw(rid=1), _raw(rid=2)]
        mock_clust.return_value = {0: raws}
        mock_gen.return_value = [(_agent_playbook(fid=100), raws)]
        agg.storage.save_agent_playbook_with_aggregate_event.return_value = (
            _agent_playbook(fid=100)
        )

        with patch.object(PlaybookAggregator, "_create_state_manager") as mock_csm:
            mgr = MagicMock()
            mgr.get_cluster_fingerprints.return_value = {}
            mock_csm.return_value = mgr

            req = PlaybookAggregatorRequest(agent_version="v1")
            agg.run(req)

        agg.storage.archive_agent_playbooks_by_playbook_name.assert_called()

    @patch.object(PlaybookAggregator, "get_clusters")
    def test_incremental_no_changes_updates_bookmark_only(self, mock_clust):
        """When no cluster changes detected, update bookmark and return."""
        agg = self._make_runnable_aggregator()
        raws = [_raw(rid=1)]
        agg.storage.get_user_playbooks.return_value = raws
        mock_clust.return_value = {0: raws}
        fp = PlaybookAggregator._compute_cluster_fingerprint(raws)

        with patch.object(PlaybookAggregator, "_create_state_manager") as mock_csm:
            mgr = MagicMock()
            mgr.get_cluster_fingerprints.return_value = {
                fp: {"agent_playbook_id": 10, "user_playbook_ids": [1]}
            }
            mock_csm.return_value = mgr

            req = PlaybookAggregatorRequest(agent_version="v1")
            agg.run(req)

        # Should NOT call save_agent_playbook_with_aggregate_event
        agg.storage.save_agent_playbook_with_aggregate_event.assert_not_called()

    @patch.object(PlaybookAggregator, "get_clusters")
    @patch.object(PlaybookAggregator, "_generate_playbooks_with_source_clusters")
    def test_incremental_with_changes_supersedes_selectively(
        self, mock_gen, mock_clust
    ):
        """Incremental mode with changed clusters soft-supersedes affected playbook_ids (always soft)."""
        agg = self._make_runnable_aggregator()
        raws_new = [_raw(rid=5), _raw(rid=6)]
        agg.storage.get_user_playbooks.return_value = raws_new
        mock_clust.return_value = {0: raws_new}
        mock_gen.return_value = [(_agent_playbook(fid=200), raws_new)]
        agg.storage.save_agent_playbook_with_aggregate_event.return_value = (
            _agent_playbook(fid=200)
        )

        with patch.object(PlaybookAggregator, "_create_state_manager") as mock_csm:
            mgr = MagicMock()
            mgr.get_cluster_fingerprints.return_value = {
                "old_fp": {"agent_playbook_id": 50, "user_playbook_ids": [5]}
            }
            mock_csm.return_value = mgr

            req = PlaybookAggregatorRequest(agent_version="v1")
            agg.run(req)

        agg.storage.archive_agent_playbooks_by_ids.assert_not_called()
        agg.storage.supersede_agent_playbooks_by_ids.assert_called_once_with(
            [50], request_id=ANY
        )
        agg.storage.delete_agent_playbooks_by_ids.assert_not_called()

    @patch.object(PlaybookAggregator, "get_clusters")
    @patch.object(PlaybookAggregator, "_generate_playbooks_with_source_clusters")
    def test_incremental_null_generation_preserves_existing_playbooks(
        self, mock_gen, mock_clust
    ):
        """Incremental mode preserves existing playbooks when LLM produces no replacement."""
        agg = self._make_runnable_aggregator()
        raws_new = [_raw(rid=5), _raw(rid=6)]
        agg.storage.get_user_playbooks.return_value = raws_new
        mock_clust.return_value = {0: raws_new}
        mock_gen.return_value = []

        with patch.object(PlaybookAggregator, "_create_state_manager") as mock_csm:
            mgr = MagicMock()
            mgr.get_cluster_fingerprints.return_value = {
                "old_fp": {"agent_playbook_id": 50, "user_playbook_ids": [1, 2]}
            }
            mock_csm.return_value = mgr

            req = PlaybookAggregatorRequest(agent_version="v1")
            agg.run(req)

        agg.storage.archive_agent_playbooks_by_ids.assert_not_called()
        agg.storage.supersede_agent_playbooks_by_ids.assert_not_called()
        agg.storage.delete_agent_playbooks_by_ids.assert_not_called()

    @patch.object(PlaybookAggregator, "get_clusters")
    @patch.object(PlaybookAggregator, "_generate_playbooks_with_source_clusters")
    def test_incremental_mixed_generation_supersedes_only_replaced_cluster(
        self, mock_gen, mock_clust
    ):
        """A null changed cluster keeps its old playbook when a sibling generates."""
        agg = self._make_runnable_aggregator()
        null_cluster = [_raw(rid=1), _raw(rid=2), _raw(rid=5)]
        generated_cluster = [_raw(rid=3), _raw(rid=4), _raw(rid=6)]
        agg.storage.get_user_playbooks.return_value = null_cluster + generated_cluster
        mock_clust.return_value = {0: null_cluster, 1: generated_cluster}
        generated = _agent_playbook(fid=200)
        generated.agent_playbook_id = 200
        mock_gen.return_value = [(generated, generated_cluster)]
        agg.storage.save_agent_playbook_with_aggregate_event.return_value = generated
        fp_null_old = PlaybookAggregator._compute_cluster_fingerprint(
            [_raw(rid=1), _raw(rid=2)]
        )
        fp_generated_old = PlaybookAggregator._compute_cluster_fingerprint(
            [_raw(rid=3), _raw(rid=4)]
        )
        fp_generated_new = PlaybookAggregator._compute_cluster_fingerprint(
            generated_cluster
        )

        with patch.object(PlaybookAggregator, "_create_state_manager") as mock_csm:
            mgr = MagicMock()
            mgr.get_cluster_fingerprints.return_value = {
                fp_null_old: {"agent_playbook_id": 50, "user_playbook_ids": [1, 2]},
                fp_generated_old: {
                    "agent_playbook_id": 60,
                    "user_playbook_ids": [3, 4],
                },
            }
            mock_csm.return_value = mgr

            req = PlaybookAggregatorRequest(agent_version="v1")
            agg.run(req)

        agg.storage.supersede_agent_playbooks_by_ids.assert_called_once_with(
            [60], request_id=ANY
        )
        call_kwargs = mgr.update_cluster_fingerprints.call_args
        new_fps = call_kwargs.kwargs.get("fingerprints") or call_kwargs[1].get(
            "fingerprints"
        )
        assert new_fps[fp_null_old]["agent_playbook_id"] == 50
        assert new_fps[fp_generated_new]["agent_playbook_id"] == 200
        assert fp_generated_old not in new_fps

    @patch.object(PlaybookAggregator, "get_clusters")
    @patch.object(PlaybookAggregator, "_generate_playbooks_with_source_clusters")
    def test_incremental_split_cluster_preserves_shared_prior_until_all_replaced(
        self, mock_gen, mock_clust
    ):
        """A split cluster keeps the old playbook while any split branch is null."""
        agg = self._make_runnable_aggregator()
        null_cluster = [_raw(rid=1), _raw(rid=2), _raw(rid=5)]
        generated_cluster = [_raw(rid=3), _raw(rid=4), _raw(rid=6)]
        agg.storage.get_user_playbooks.return_value = null_cluster + generated_cluster
        mock_clust.return_value = {0: null_cluster, 1: generated_cluster}
        generated = _agent_playbook(fid=200)
        generated.agent_playbook_id = 200
        mock_gen.return_value = [(generated, generated_cluster)]
        agg.storage.save_agent_playbook_with_aggregate_event.return_value = generated
        fp_old = PlaybookAggregator._compute_cluster_fingerprint(
            [_raw(rid=1), _raw(rid=2), _raw(rid=3), _raw(rid=4)]
        )
        fp_generated_new = PlaybookAggregator._compute_cluster_fingerprint(
            generated_cluster
        )

        with patch.object(PlaybookAggregator, "_create_state_manager") as mock_csm:
            mgr = MagicMock()
            mgr.get_cluster_fingerprints.return_value = {
                fp_old: {
                    "agent_playbook_id": 50,
                    "user_playbook_ids": [1, 2, 3, 4],
                },
            }
            mock_csm.return_value = mgr

            req = PlaybookAggregatorRequest(agent_version="v1")
            agg.run(req)

        agg.storage.supersede_agent_playbooks_by_ids.assert_not_called()
        call_kwargs = mgr.update_cluster_fingerprints.call_args
        new_fps = call_kwargs.kwargs.get("fingerprints") or call_kwargs[1].get(
            "fingerprints"
        )
        assert new_fps[fp_old]["agent_playbook_id"] == 50
        assert new_fps[fp_generated_new]["agent_playbook_id"] == 200

    @patch.object(PlaybookAggregator, "get_clusters")
    @patch.object(PlaybookAggregator, "_generate_playbooks_with_source_clusters")
    def test_save_exception_restores_full_archive(self, mock_gen, mock_clust):
        """Exception during save_agent_playbooks in full-archive mode restores playbooks."""
        agg = self._make_runnable_aggregator()
        mock_clust.return_value = {0: [_raw(rid=1)]}
        mock_gen.side_effect = RuntimeError("LLM failed")

        req = PlaybookAggregatorRequest(agent_version="v1", rerun=True)

        with pytest.raises(RuntimeError, match="LLM failed"):
            agg.run(req)

        agg.storage.restore_archived_agent_playbooks_by_playbook_name.assert_called()

    @patch.object(PlaybookAggregator, "get_clusters")
    @patch.object(PlaybookAggregator, "_generate_playbooks_with_source_clusters")
    def test_save_exception_restores_incremental_archive(self, mock_gen, mock_clust):
        """Exception during save_agent_playbooks in incremental mode restores by ids."""
        agg = self._make_runnable_aggregator()
        raws_new = [_raw(rid=5)]
        agg.storage.get_user_playbooks.return_value = raws_new
        mock_clust.return_value = {0: raws_new}
        mock_gen.side_effect = RuntimeError("Boom")

        with patch.object(PlaybookAggregator, "_create_state_manager") as mock_csm:
            mgr = MagicMock()
            mgr.get_cluster_fingerprints.return_value = {
                "old_fp": {"agent_playbook_id": 50, "user_playbook_ids": [1]}
            }
            mock_csm.return_value = mgr

            req = PlaybookAggregatorRequest(agent_version="v1")

            with pytest.raises(RuntimeError, match="Boom"):
                agg.run(req)

        agg.storage.restore_archived_agent_playbooks_by_ids.assert_called_once_with(
            [50]
        )

    @patch.object(PlaybookAggregator, "get_clusters")
    @patch.object(PlaybookAggregator, "_generate_playbooks_with_source_clusters")
    def test_run_fingerprint_state_updated(self, mock_gen, mock_clust):
        """Fingerprint state should be updated after a successful run."""
        agg = self._make_runnable_aggregator()
        raws = [_raw(rid=1), _raw(rid=2)]
        mock_clust.return_value = {0: raws}
        saved = _agent_playbook(fid=100)
        saved.agent_playbook_id = 100
        mock_gen.return_value = [(saved, raws)]
        agg.storage.save_agent_playbook_with_aggregate_event.return_value = saved

        with patch.object(PlaybookAggregator, "_create_state_manager") as mock_csm:
            mgr = MagicMock()
            mgr.get_cluster_fingerprints.return_value = {}
            mock_csm.return_value = mgr

            req = PlaybookAggregatorRequest(agent_version="v1")
            agg.run(req)

        mgr.update_cluster_fingerprints.assert_called_once()
        call_kwargs = mgr.update_cluster_fingerprints.call_args
        fingerprints_arg = call_kwargs.kwargs.get("fingerprints") or call_kwargs[1].get(
            "fingerprints"
        )
        assert fingerprints_arg is not None
        # The fingerprint for the cluster should have agent_playbook_id=100 assigned
        for fp_data in fingerprints_arg.values():
            if fp_data["agent_playbook_id"] is not None:
                assert fp_data["agent_playbook_id"] == 100

    @patch.object(PlaybookAggregator, "get_clusters")
    @patch.object(PlaybookAggregator, "_generate_playbooks_with_source_clusters")
    def test_incremental_changed_clusters_but_no_archived_ids(
        self, mock_gen, mock_clust
    ):
        """Branch 508->511: changed clusters exist but archived_playbook_ids is empty."""
        agg = self._make_runnable_aggregator()
        raws_new = [_raw(rid=5), _raw(rid=6)]
        agg.storage.get_user_playbooks.return_value = raws_new
        mock_clust.return_value = {0: raws_new}
        mock_gen.return_value = [(_agent_playbook(fid=200), raws_new)]
        agg.storage.save_agent_playbook_with_aggregate_event.return_value = (
            _agent_playbook(fid=200)
        )

        with patch.object(PlaybookAggregator, "_create_state_manager") as mock_csm:
            mgr = MagicMock()
            # prev fingerprints exist but the new cluster fingerprint is different,
            # and the old fingerprint has agent_playbook_id=None so nothing to archive
            mgr.get_cluster_fingerprints.return_value = {
                "old_fp": {"agent_playbook_id": None, "user_playbook_ids": [1, 2]}
            }
            mock_csm.return_value = mgr

            req = PlaybookAggregatorRequest(agent_version="v1")
            agg.run(req)

        # archive_agent_playbooks_by_ids should NOT be called (no ids to archive)
        agg.storage.archive_agent_playbooks_by_ids.assert_not_called()
        # delete_agent_playbooks_by_ids should NOT be called either (branch 627->exit)
        agg.storage.delete_agent_playbooks_by_ids.assert_not_called()

    @patch.object(PlaybookAggregator, "get_clusters")
    @patch.object(PlaybookAggregator, "_generate_playbooks_with_source_clusters")
    def test_saved_fb_without_playbook_id_skipped_in_fingerprint_assignment(
        self, mock_gen, mock_clust
    ):
        """Branch 577->576: saved_fb with falsy playbook_id skipped during fp assignment."""
        agg = self._make_runnable_aggregator()
        raws = [_raw(rid=1)]
        mock_clust.return_value = {0: raws}
        # AgentPlaybook with agent_playbook_id=0 (falsy)
        fb_no_id = _agent_playbook(fid=0, content="no id")
        fb_no_id.agent_playbook_id = 0
        mock_gen.return_value = [(fb_no_id, raws)]
        agg.storage.save_agent_playbook_with_aggregate_event.return_value = fb_no_id

        with patch.object(PlaybookAggregator, "_create_state_manager") as mock_csm:
            mgr = MagicMock()
            mgr.get_cluster_fingerprints.return_value = {}
            mock_csm.return_value = mgr

            req = PlaybookAggregatorRequest(agent_version="v1")
            agg.run(req)

        mgr.update_cluster_fingerprints.assert_called_once()
        call_kwargs = mgr.update_cluster_fingerprints.call_args
        new_fps = call_kwargs.kwargs.get("fingerprints") or call_kwargs[1].get(
            "fingerprints"
        )
        # The fingerprint should still have agent_playbook_id=None since fb_no_id.agent_playbook_id was falsy
        for fp_data in new_fps.values():
            assert fp_data["agent_playbook_id"] is None

    @patch.object(PlaybookAggregator, "get_clusters")
    @patch.object(PlaybookAggregator, "_generate_playbooks_with_source_clusters")
    def test_exception_in_incremental_no_archived_ids_still_raises(
        self, mock_gen, mock_clust
    ):
        """Branch 641->644: exception in incremental mode with empty archived_playbook_ids."""
        agg = self._make_runnable_aggregator()
        raws_new = [_raw(rid=5)]
        agg.storage.get_user_playbooks.return_value = raws_new
        mock_clust.return_value = {0: raws_new}
        mock_gen.side_effect = RuntimeError("Kaboom")

        with patch.object(PlaybookAggregator, "_create_state_manager") as mock_csm:
            mgr = MagicMock()
            # prev fingerprints with no playbook_id => no archived_playbook_ids
            mgr.get_cluster_fingerprints.return_value = {
                "old_fp": {"agent_playbook_id": None, "user_playbook_ids": [1]}
            }
            mock_csm.return_value = mgr

            req = PlaybookAggregatorRequest(agent_version="v1")

            with pytest.raises(RuntimeError, match="Kaboom"):
                agg.run(req)

        # Neither restore method should be called since archived_playbook_ids is empty
        # and full_archive is False
        agg.storage.restore_archived_agent_playbooks_by_playbook_name.assert_not_called()
        agg.storage.restore_archived_agent_playbooks_by_ids.assert_not_called()

    @patch.object(PlaybookAggregator, "get_clusters")
    @patch.object(PlaybookAggregator, "_generate_playbooks_with_source_clusters")
    def test_run_with_none_saved_playbooks_in_list(self, mock_gen, mock_clust):
        """saved_playbooks list containing None entries should not cause errors."""
        agg = self._make_runnable_aggregator()
        raws = [_raw(rid=1)]
        mock_clust.return_value = {0: raws}
        mock_gen.return_value = []

        with patch.object(PlaybookAggregator, "_create_state_manager") as mock_csm:
            mgr = MagicMock()
            mgr.get_cluster_fingerprints.return_value = {}
            mock_csm.return_value = mgr

            req = PlaybookAggregatorRequest(agent_version="v1")
            # Should not raise
            agg.run(req)

    @patch.object(PlaybookAggregator, "get_clusters")
    @patch.object(PlaybookAggregator, "_generate_playbooks_with_source_clusters")
    def test_multiple_saved_playbooks_assigned_to_multiple_fingerprints(
        self, mock_gen, mock_clust
    ):
        """Branch 580->579: second saved_fb skips first fp (already assigned) and finds second."""
        agg = self._make_runnable_aggregator()
        raws_a = [_raw(rid=1)]
        raws_b = [_raw(rid=2)]
        mock_clust.return_value = {0: raws_a, 1: raws_b}
        fb1 = _agent_playbook(fid=100, content="a")
        fb1.agent_playbook_id = 100
        fb2 = _agent_playbook(fid=200, content="b")
        fb2.agent_playbook_id = 200
        mock_gen.return_value = [(fb1, raws_a), (fb2, raws_b)]
        agg.storage.save_agent_playbook_with_aggregate_event.side_effect = [fb1, fb2]

        with patch.object(PlaybookAggregator, "_create_state_manager") as mock_csm:
            mgr = MagicMock()
            mgr.get_cluster_fingerprints.return_value = {}
            mock_csm.return_value = mgr

            req = PlaybookAggregatorRequest(agent_version="v1")
            agg.run(req)

        mgr.update_cluster_fingerprints.assert_called_once()
        call_kwargs = mgr.update_cluster_fingerprints.call_args
        new_fps = call_kwargs.kwargs.get("fingerprints") or call_kwargs[1].get(
            "fingerprints"
        )
        # Both fingerprints should have playbook_ids assigned
        assigned_ids = [
            v["agent_playbook_id"]
            for v in new_fps.values()
            if v["agent_playbook_id"] is not None
        ]
        assert len(assigned_ids) == 2
        assert set(assigned_ids) == {100, 200}

    @patch.object(PlaybookAggregator, "get_clusters")
    @patch.object(PlaybookAggregator, "_generate_playbooks_with_source_clusters")
    def test_generated_playbook_id_maps_to_exact_source_cluster(
        self, mock_gen, mock_clust
    ):
        """A generated playbook after a duplicate cluster keeps the correct fingerprint."""
        agg = self._make_runnable_aggregator()
        duplicate_cluster = [_raw(rid=1)]
        generated_cluster = [_raw(rid=2)]
        mock_clust.return_value = {0: duplicate_cluster, 1: generated_cluster}
        saved = _agent_playbook(fid=200, content="b")
        saved.agent_playbook_id = 200
        mock_gen.return_value = [(saved, generated_cluster)]
        agg.storage.save_agent_playbook_with_aggregate_event.return_value = saved

        with patch.object(PlaybookAggregator, "_create_state_manager") as mock_csm:
            mgr = MagicMock()
            mgr.get_cluster_fingerprints.return_value = {}
            mock_csm.return_value = mgr

            req = PlaybookAggregatorRequest(agent_version="v1")
            agg.run(req)

        mgr.update_cluster_fingerprints.assert_called_once()
        call_kwargs = mgr.update_cluster_fingerprints.call_args
        new_fps = call_kwargs.kwargs.get("fingerprints") or call_kwargs[1].get(
            "fingerprints"
        )
        duplicate_fp = PlaybookAggregator._compute_cluster_fingerprint(
            duplicate_cluster
        )
        generated_fp = PlaybookAggregator._compute_cluster_fingerprint(
            generated_cluster
        )
        assert new_fps[duplicate_fp]["agent_playbook_id"] is None
        assert new_fps[generated_fp]["agent_playbook_id"] == 200

    @patch.object(PlaybookAggregator, "get_clusters")
    @patch.object(PlaybookAggregator, "_generate_playbooks_with_source_clusters")
    def test_incremental_carries_forward_unchanged_fingerprints(
        self, mock_gen, mock_clust
    ):
        """Unchanged cluster fingerprints are carried forward in incremental mode."""
        agg = self._make_runnable_aggregator()
        # Two clusters: one unchanged, one new
        raws_unchanged = [_raw(rid=1)]
        raws_new = [_raw(rid=5), _raw(rid=6)]
        fp_unchanged = PlaybookAggregator._compute_cluster_fingerprint(raws_unchanged)

        all_raws = raws_unchanged + raws_new
        agg.storage.get_user_playbooks.return_value = all_raws
        mock_clust.return_value = {0: raws_unchanged, 1: raws_new}
        mock_gen.return_value = [(_agent_playbook(fid=200), raws_new)]
        agg.storage.save_agent_playbook_with_aggregate_event.return_value = (
            _agent_playbook(fid=200)
        )

        with patch.object(PlaybookAggregator, "_create_state_manager") as mock_csm:
            mgr = MagicMock()
            prev_fps = {
                fp_unchanged: {"agent_playbook_id": 10, "user_playbook_ids": [1]},
                "vanished_fp": {"agent_playbook_id": 20, "user_playbook_ids": [2]},
            }
            mgr.get_cluster_fingerprints.return_value = prev_fps
            mock_csm.return_value = mgr

            req = PlaybookAggregatorRequest(agent_version="v1")
            agg.run(req)

        mgr.update_cluster_fingerprints.assert_called_once()
        call_kwargs = mgr.update_cluster_fingerprints.call_args
        new_fps = call_kwargs.kwargs.get("fingerprints") or call_kwargs[1].get(
            "fingerprints"
        )
        # Unchanged fingerprint should be carried forward
        assert fp_unchanged in new_fps
        assert new_fps[fp_unchanged]["agent_playbook_id"] == 10


# ---------------------------------------------------------------------------
# _format_cluster_input
# ---------------------------------------------------------------------------


class TestFormatClusterInput:
    def test_all_fields_present(self):
        """Each playbook becomes a numbered block with Content and Trigger."""
        agg = _make_aggregator()
        raws = [
            _raw(rid=1, when="cond1"),
            _raw(rid=2, when="cond2"),
        ]

        result = agg._format_cluster_input(raws)

        assert "[1]" in result
        assert "[2]" in result
        assert 'Content: "content-1"' in result
        assert 'Content: "content-2"' in result
        assert 'Trigger: "cond1"' in result
        assert 'Trigger: "cond2"' in result

    def test_no_trigger_omits_trigger_line(self):
        agg = _make_aggregator()
        raws = [_raw(rid=1, when=None)]

        result = agg._format_cluster_input(raws)

        assert "Trigger:" not in result

    def test_empty_list_returns_placeholder(self):
        """Empty input returns a placeholder string."""
        agg = _make_aggregator()
        result = agg._format_cluster_input([])
        assert result == "(No playbook items)"

    def test_content_is_first_field_after_number(self):
        """Content line appears immediately after the numbered header."""
        agg = _make_aggregator()
        raws = [_raw(rid=1, when="cond")]

        result = agg._format_cluster_input(raws)

        lines = result.strip().split("\n")
        assert lines[0] == "[1]"
        assert lines[1].startswith("Content:")

    def test_multiple_playbooks_separated_by_blank_lines(self):
        """Multiple playbooks are separated by blank lines."""
        agg = _make_aggregator()
        raws = [_raw(rid=1, when="cond1"), _raw(rid=2, when="cond2")]

        result = agg._format_cluster_input(raws)

        # Two blocks separated by double newline
        assert "\n\n" in result
        assert "[1]" in result
        assert "[2]" in result


# ---------------------------------------------------------------------------
# get_clusters
# ---------------------------------------------------------------------------


class TestGetClusters:
    def test_no_config_returns_empty(self):
        agg = _make_aggregator()
        result = agg.get_clusters([_raw()], None)  # type: ignore[arg-type]
        assert result == {}

    def test_no_user_playbooks_returns_empty(self):
        agg = _make_aggregator()
        config = PlaybookAggregatorConfig(min_cluster_size=2)
        result = agg.get_clusters([], config)
        assert result == {}

    def test_fewer_than_min_returns_empty(self):
        agg = _make_aggregator()
        config = PlaybookAggregatorConfig(min_cluster_size=5)
        raws = [_raw(rid=i) for i in range(3)]
        # Need real embeddings for len check
        for r in raws:
            r.embedding = [0.0] * 10

        with patch.dict("os.environ", {"MOCK_LLM_RESPONSE": ""}):
            result = agg.get_clusters(raws, config)

        assert result == {}

    def test_mock_mode_clusters_by_when_condition(self):
        agg = _make_aggregator()
        config = PlaybookAggregatorConfig(min_cluster_size=2)
        raws = [
            _raw(rid=1, when="cond_a"),
            _raw(rid=2, when="cond_a"),
            _raw(rid=3, when="cond_b"),
        ]

        with patch.dict("os.environ", {"MOCK_LLM_RESPONSE": "true"}):
            result = agg.get_clusters(raws, config)

        # Only cond_a has 2 playbooks (meets threshold)
        assert len(result) == 1
        assert len(list(result.values())[0]) == 2


# ---------------------------------------------------------------------------
# _process_aggregation_response
# ---------------------------------------------------------------------------


class TestProcessAggregationResponse:
    def test_none_response_returns_none(self):
        agg = _make_aggregator()
        assert agg._process_aggregation_response(None, [_raw()]) is None  # type: ignore[arg-type]

    def test_null_playbook_returns_none(self):
        from reflexio.server.services.playbook.playbook_service_utils import (
            PlaybookAggregationOutput,
        )

        agg = _make_aggregator()
        response = PlaybookAggregationOutput(playbook=None)
        assert agg._process_aggregation_response(response, [_raw()]) is None

    def test_valid_response_returns_playbook(self):
        from reflexio.server.services.playbook.playbook_service_utils import (
            PlaybookAggregationOutput,
            StructuredPlaybookContent,
        )

        agg = _make_aggregator()
        structured = StructuredPlaybookContent(
            trigger="when testing",
            content="do something",
        )
        response = PlaybookAggregationOutput(playbook=structured)

        result = agg._process_aggregation_response(response, [_raw()])

        assert result is not None
        assert result.trigger == "when testing"
        assert result.content == "do something"
        assert result.playbook_status == PlaybookStatus.PENDING

    def test_does_not_postprocess_response_again(self):
        from reflexio.server.services.playbook.playbook_service_utils import (
            PlaybookAggregationOutput,
            StructuredPlaybookContent,
        )

        class CountingProcessor(_MappingAwareProcessor):
            def __init__(self) -> None:
                super().__init__()
                self.postprocess_calls = 0

            def postprocess_aggregation_output(
                self,
                value: object,
                *,
                context: AggregationPromptProcessingContext | None = None,
            ) -> PromptPostprocessResult:
                self.postprocess_calls += 1
                return super().postprocess_aggregation_output(value, context=context)

        processor = CountingProcessor()
        agg = _make_aggregator(aggregation_prompt_processor=processor)
        response = PlaybookAggregationOutput(
            playbook=StructuredPlaybookContent(
                trigger="when testing",
                content="do something",
            )
        )

        result = agg._process_aggregation_response(response, [_raw()])

        assert result is not None
        assert processor.postprocess_calls == 0

    def test_empty_structured_response_returns_none(self):
        from reflexio.server.services.playbook.playbook_service_utils import (
            PlaybookAggregationOutput,
            StructuredPlaybookContent,
        )

        agg = _make_aggregator()
        response = PlaybookAggregationOutput(
            playbook=StructuredPlaybookContent(
                trigger=None,
                content="   ",
                rationale=None,
            )
        )

        assert agg._process_aggregation_response(response, [_raw()]) is None


# ---------------------------------------------------------------------------
# _group_playbooks_by_direction — content-similarity grouping (no polarity gate)
# ---------------------------------------------------------------------------


def _make_pb(
    content: str,
    rid: int = 1,
    rationale: str | None = None,
) -> UserPlaybook:
    """Build a minimal UserPlaybook for grouping/aggregation tests.

    Grouping is now purely content-similarity based (Option B): whole-content
    polarity is no longer derived or gated. A skill may legitimately hold
    mixed-orientation rules for different sub-aspects; preserving distinct
    do/avoid rules when merging is the aggregation prompt's responsibility.
    """
    return UserPlaybook(
        user_playbook_id=rid,
        agent_version="v1",
        request_id=f"req-{rid}",
        playbook_name="test_fb",
        content=content,
        rationale=rationale,
    )


def test_aggregator_groups_by_content_similarity_not_polarity():
    """Grouping no longer gates on whole-content polarity.

    Two rows whose tokens overlap above the threshold but carry opposite
    orientations (a "do" rule and an "avoid" rule) MUST now land in the same
    similarity group — the retired mechanical whole-content polarity
    direction-split used to force them apart. Keeping the opposite-orientation
    rules distinct inside a
    merged skill is delegated to the aggregation prompt, not to a mechanical
    pre-LLM split.
    """
    positive = _make_pb(
        content="Always ask clarifying questions before proceeding",
        rid=1,
    )
    negative = _make_pb(
        content="Avoid asking clarifying questions before proceeding",
        rationale="user pushback observed",
        rid=2,
    )
    # Sanity: the two rows overlap above the grouping threshold.
    assert PlaybookAggregator._token_overlap(
        PlaybookAggregator._get_direction_key(positive),
        PlaybookAggregator._get_direction_key(negative),
        0.6,
    )
    groups = PlaybookAggregator._group_playbooks_by_direction(
        [positive, negative], threshold=0.6
    )
    # No polarity gate: high token overlap => a single similarity group.
    assert len(groups) == 1, (
        f"high-overlap content must group together (no polarity gate), got {groups}"
    )
    assert len(groups[0]) == 2


def test_aggregation_preserves_distinct_do_and_avoid_rules():
    """Prompt-preserved outcome: a do-rule and an avoid-rule survive
    aggregation as separate rules rather than being collapsed into one.

    The mechanical polarity-bucketing gate is gone; preserving distinct
    orientations is now the aggregation prompt's job. Here we drive the
    behavior through the mocked LLM aggregation output (the prompt's job, made
    deterministic) and assert that the resulting AgentPlaybook content keeps
    BOTH the do-rule and the avoid-rule as distinct bullets — the opposite of
    collapsing them into a single rule.
    """
    from reflexio.server.services.playbook.playbook_service_utils import (
        PlaybookAggregationOutput,
        StructuredPlaybookContent,
    )

    agg = _make_aggregator()

    # The cluster contains a do-rule and an avoid-rule on the same broad topic
    # (different sub-aspects). Under Option B these belong in one skill but as
    # two distinct rules.
    cluster = [
        _make_pb(content="Announce the deploy in the channel first", rid=1),
        _make_pb(
            content="Avoid deploying on Friday afternoons",
            rationale="late-Friday deploys caused weekend incidents",
            rid=2,
        ),
    ]

    # Mocked LLM aggregation output: the prompt is responsible for keeping the
    # two orientations as separate rules — assert that distinct-rule shape is
    # carried through into the generated playbook content (not collapsed).
    merged_content = (
        "- Announce the deploy in the channel first.\n"
        "- Avoid deploying on Friday afternoons."
    )
    response = PlaybookAggregationOutput(
        playbook=StructuredPlaybookContent(
            trigger="When deploying a service.",
            content=merged_content,
            rationale="Coordinated, well-timed deploys reduce incidents.",
        )
    )
    agg.client.generate_chat_response.return_value = response

    with patch.dict("os.environ", {"MOCK_LLM_RESPONSE": ""}):
        result = agg._generate_playbook_from_cluster(cluster, "None")

    assert result is not None
    # Both orientations survive as DISTINCT rules — not merged into one.
    assert "Announce the deploy in the channel first" in result.content
    assert "Avoid deploying on Friday afternoons" in result.content
    # Two separate bullets => the do-rule and the avoid-rule were not collapsed.
    bullet_lines = [
        line for line in result.content.splitlines() if line.strip().startswith("-")
    ]
    assert len(bullet_lines) == 2


def test_playbook_aggregation_prompt_specifies_structured_format():
    """Sanity (v2.2.0): aggregator prompt must carry the Agent-Skills
    formatting discipline — imperative conditional triggers, markdown bullet
    content, one-sentence rationale. Mirrors the extraction prompt v1.4.0
    so the downstream agent sees the same shape across per-user playbooks
    and aggregated ones. Guards against silent regression to prose shape."""
    from reflexio.server.prompt.prompt_manager import PromptManager

    pm = PromptManager()
    out = pm.render_prompt(
        "playbook_aggregation",
        variables={
            "user_playbooks": '[1]\nContent: "x"\nTrigger: "y"',
            "existing_approved_playbooks": "(none)",
            "aggregation_prompt_extra_instructions": "",
        },
    )
    # The Playbook format section must be present.
    assert "Playbook format" in out
    # Trigger guidance — imperative conditional phrasing + keyword coverage.
    assert "imperative conditional phrasing" in out
    # Content guidance — markdown bullet list for multi-action policies.
    assert "markdown bullet list" in out
    # Examples now show bullet-shaped content, not single-sentence prose.
    assert "- Ask for CLI preference" in out
    # Rationale guidance — one sentence WHY.
    assert "one sentence" in out.lower()


def test_playbook_aggregation_prompt_generalizes_direct_identifiers():
    """Aggregation prompt should generalize direct identifiers for shared playbooks."""
    from reflexio.server.prompt.prompt_manager import PromptManager

    out = PromptManager().render_prompt(
        "playbook_aggregation",
        {
            "existing_approved_playbooks": "[]",
            "user_playbooks": "TRIGGER conditions (to be consolidated):\n- when approving a deployment\nRATIONALE summaries:\n- direct approval details appeared in the source",
            "aggregation_prompt_extra_instructions": "",
        },
    )

    assert "Privacy and Identifier Generalization" in out
    assert "shared organization-wide rules" in out
    assert "all source fields shown to you" in out
    assert "triggers, rationales, and any freeform content" in out
    assert "Never carry user-specific or source-specific direct identifiers" in out
    assert "Secrets and credentials must not be copied" in out
    assert 'Return {"playbook": null}' in out


def test_playbook_aggregation_prompt_does_not_add_marker_guidance_by_default():
    """Default OSS prompt should stay generic because OSS does not create markers."""
    from reflexio.server.prompt.prompt_manager import PromptManager

    out = PromptManager().render_prompt(
        "playbook_aggregation",
        {
            "existing_approved_playbooks": "[]",
            "user_playbooks": "TRIGGER conditions:\n- When direct identifiers appear in source notes",
            "aggregation_prompt_extra_instructions": "",
        },
    )

    assert "behavior.\n\n- Preserve the reusable procedure" in out


def test_aggregation_prompt_extra_instructions_render_before_next_bullet():
    from reflexio.server.prompt.prompt_manager import PromptManager

    out = PromptManager().render_prompt(
        "playbook_aggregation",
        {
            "existing_approved_playbooks": "[]",
            "user_playbooks": "sample",
            "aggregation_prompt_extra_instructions": (
                "Extra guidance without trailing newline"
            ),
        },
    )

    assert (
        "Extra guidance without trailing newline\n- Preserve the reusable procedure"
        in out
    )
    assert "newline- Preserve the reusable procedure" not in out


def test_aggregation_prompt_extra_instructions_are_rendered_when_injected():
    class ProcessorWithPromptInstructions(_MappingAwareProcessor):
        prompt_extra_instructions = (
            "Processed markers from this processor represent transformed tokens."
        )

    agg = _make_aggregator(
        aggregation_prompt_processor=ProcessorWithPromptInstructions()
    )
    captured_variables: dict[str, str] = {}

    def render_prompt(_prompt_id: str, variables: dict[str, str]) -> str:
        captured_variables.update(variables)
        return variables["aggregation_prompt_extra_instructions"]

    agg.request_context.prompt_manager.render_prompt.side_effect = render_prompt
    agg.client.generate_chat_response.return_value = PlaybookAggregationOutput(
        playbook=StructuredPlaybookContent(
            content="Use generalized roles.",
            trigger="When access support is needed.",
        )
    )

    with patch.dict("os.environ", {"MOCK_LLM_RESPONSE": ""}):
        result = agg._generate_playbook_from_cluster(
            [_raw(rid=1)],
            "None",
            processing_context=AggregationPromptProcessingContext(changed=True),
        )

    assert result is not None
    assert captured_variables["aggregation_prompt_extra_instructions"].startswith(
        "Processed markers"
    )


def test_aggregation_prompt_extra_instructions_are_conditional_in_cluster_flow():
    class ProcessorWithPromptInstructions(_MappingAwareProcessor):
        prompt_extra_instructions = (
            "Processed markers from this processor represent transformed tokens."
        )

    agg = _make_aggregator(
        aggregation_prompt_processor=ProcessorWithPromptInstructions()
    )
    captured_variables: list[dict[str, str]] = []

    def render_prompt(_prompt_id: str, variables: dict[str, str]) -> str:
        captured_variables.append(dict(variables))
        return variables["aggregation_prompt_extra_instructions"]

    agg.request_context.prompt_manager.render_prompt.side_effect = render_prompt
    agg.client.generate_chat_response.return_value = PlaybookAggregationOutput(
        playbook=StructuredPlaybookContent(
            content="Use generalized roles.",
            trigger="When access support is needed.",
        )
    )

    with patch.dict("os.environ", {"MOCK_LLM_RESPONSE": ""}):
        agg._generate_playbooks_with_source_clusters({0: [_raw(rid=1)]}, [])
        agg._generate_playbooks_with_source_clusters(
            {
                0: [
                    UserPlaybook(
                        user_playbook_id=2,
                        agent_version="v1",
                        request_id="req-2",
                        playbook_name="test_fb",
                        content="Project Zephyr needs access support.",
                        trigger="When access support is needed.",
                    )
                ]
            },
            [],
        )

    assert captured_variables[0]["aggregation_prompt_extra_instructions"] == ""
    assert captured_variables[1]["aggregation_prompt_extra_instructions"].startswith(
        "Processed markers"
    )


def test_contextual_prompt_extra_instructions_receive_opaque_context():
    class ContextAwareProcessor(_MappingAwareProcessor):
        def preprocess_prompt_text(
            self,
            text: str,
            *,
            shared_state: dict[str, Any] | None = None,
            context: AggregationPromptProcessingContext | None = None,
        ) -> PromptPreprocessResult:
            result = super().preprocess_prompt_text(
                text,
                shared_state=shared_state,
                context=context,
            )
            if "special routing" in text and context is not None:
                context.data["instruction"] = "Preserve special routing."
                return PromptPreprocessResult(text=text.upper(), changed=True)
            return result

        def prompt_instructions(
            self,
            context: AggregationPromptProcessingContext,
        ) -> str | None:
            instruction = context.data.get("instruction")
            return instruction if isinstance(instruction, str) else None

    agg = _make_aggregator(aggregation_prompt_processor=ContextAwareProcessor())
    captured_variables: dict[str, str] = {}

    def render_prompt(_prompt_id: str, variables: dict[str, str]) -> str:
        captured_variables.update(variables)
        return variables["aggregation_prompt_extra_instructions"]

    agg.request_context.prompt_manager.render_prompt.side_effect = render_prompt
    agg.client.generate_chat_response.return_value = PlaybookAggregationOutput(
        playbook=StructuredPlaybookContent(
            content="Preserve Acme routing.",
            trigger="When Acme routing applies.",
        )
    )

    with patch.dict("os.environ", {"MOCK_LLM_RESPONSE": ""}):
        result = agg._generate_playbooks_with_source_clusters(
            {
                0: [
                    UserPlaybook(
                        user_playbook_id=1,
                        agent_version="v1",
                        request_id="req-1",
                        playbook_name="test_fb",
                        content="special routing matters.",
                        trigger="When special routing applies.",
                    )
                ]
            },
            [],
        )

    assert len(result) == 1
    assert captured_variables["aggregation_prompt_extra_instructions"].startswith(
        "Preserve special routing."
    )


def test_aggregation_prompt_extra_instructions_ignore_non_string_values():
    class ProcessorWithInvalidPromptInstructions(_MappingAwareProcessor):
        def prompt_instructions(
            self,
            context: AggregationPromptProcessingContext,  # noqa: ARG002
        ) -> Any:
            return object()

    agg = _make_aggregator(
        aggregation_prompt_processor=ProcessorWithInvalidPromptInstructions()
    )

    assert (
        agg._aggregation_prompt_extra_instructions_for_context(
            AggregationPromptProcessingContext(changed=True)
        )
        == ""
    )


def test_playbook_aggregation_prompt_has_privacy_self_check_before_output():
    """Privacy checklist should run after source sections and before final JSON output."""
    from reflexio.server.prompt.prompt_manager import PromptManager

    out = PromptManager().render_prompt(
        "playbook_aggregation",
        {
            "existing_approved_playbooks": "[]",
            "user_playbooks": "TRIGGER conditions (to be consolidated):\n- when handling account access",
            "aggregation_prompt_extra_instructions": "",
        },
    )

    checklist_index = out.index("Before returning JSON")
    output_index = out.index("## Output")
    assert checklist_index < output_index
    assert "`trigger`, `content`, and `rationale`" in out[checklist_index:output_index]
    assert (
        "direct identifiers, secrets, raw contact details, or exact IDs"
        in out[checklist_index:output_index]
    )
    assert (
        "grounded in the clustered source playbooks"
        in out[checklist_index:output_index]
    )


def test_playbook_aggregation_prompt_preserves_distinct_orientations():
    """v2.2.0: the aggregation prompt must carry the preserve-distinct-rules
    instruction that replaced the retired mechanical polarity-bucketing gate.

    When merging similar playbooks, the model must keep a do-rule and an
    avoid-rule (opposite orientations) as SEPARATE rules — never collapse them
    into one. This is the text-first replacement for the retired mechanical
    whole-content polarity direction-split in the aggregator."""
    from reflexio.server.prompt.prompt_manager import PromptManager

    pm = PromptManager()
    out = pm.render_prompt(
        "playbook_aggregation",
        variables={
            "user_playbooks": '[1]\nContent: "x"\nTrigger: "y"',
            "existing_approved_playbooks": "(none)",
            "aggregation_prompt_extra_instructions": "",
        },
    )
    # The preserve-distinct-orientations instruction must be present.
    assert "Preserve distinct orientations" in out
    # It must explicitly forbid collapsing a do-rule and an avoid-rule into one.
    # (Normalize whitespace so a line-wrapped phrase still matches.)
    normalized = " ".join(out.split())
    assert 'never collapse a "do" rule and an "avoid" rule into one' in normalized
    # Mixed-orientation rules for different sub-aspects are allowed in one skill.
    assert "separate bullets" in normalized
