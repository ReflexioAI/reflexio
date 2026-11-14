"""Saved child/parent TTL receipts retain evidence and attribution on replay."""

from dataclasses import replace
from unittest.mock import MagicMock

from reflexio.server.services.extraction.overflow_pipeline import (
    extract_overflow_window,
)
from reflexio.server.services.profile.committed_output import (
    read_saved_structured_output,
)
from reflexio.server.services.profile.profile_generation_service_utils import (
    StructuredProfilesOutput,
)
from tests.server.services.test_overflow_carried_evidence import (
    coordinator_fixture as coordinator_fixture,
)


def test_single_saved_legacy_child_and_parent_replay_without_provider(
    coordinator_fixture,
):
    coordinator, build, execute = coordinator_fixture
    prepared = coordinator.require_prepared()
    coordinator.require_prepared.return_value = replace(
        prepared, chunks=prepared.chunks[:1]
    )

    def saved_legacy_child(call):
        result = execute(call)
        raw = result.output.model_dump(mode="json")
        raw["profiles"][0].pop("explicit_storage_deadline", None)
        raw["profiles"][0].pop("explicit_forget_quote", None)
        raw["profiles"][0]["time_to_live"] = "one_week"
        coordinator.checkpoint["completed_steps"]["child-0"]["output"]["output"] = raw
        return replace(
            result, output=read_saved_structured_output(StructuredProfilesOutput, raw)
        )

    client = MagicMock()
    client.generate_chat_response_with_provenance.side_effect = AssertionError(
        "must reuse saved output"
    )
    first = extract_overflow_window(
        coordinator=coordinator,
        client=client,
        build_call=build,
        execute_call=saved_legacy_child,
        final_validator=None,
    )
    saved = coordinator.saved_run.committed_output
    keys = set(coordinator.checkpoint["candidate_models"])
    second = extract_overflow_window(
        coordinator=coordinator,
        client=client,
        build_call=build,
        execute_call=MagicMock(side_effect=AssertionError("must reuse saved children")),
        final_validator=None,
    )
    assert isinstance(first.output, StructuredProfilesOutput)
    assert isinstance(second.output, StructuredProfilesOutput)
    assert first.output.model_dump(mode="json") == second.output.model_dump(mode="json")
    assert second.output.profiles is not None
    assert second.output.profiles[0].model_dump()["time_to_live"] == "one_week"
    assert set(coordinator.checkpoint["candidate_models"]) == keys
    assert coordinator.saved_run.committed_output == saved
    client.generate_chat_response_with_provenance.assert_not_called()
