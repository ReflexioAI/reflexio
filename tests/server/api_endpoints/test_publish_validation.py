"""Boundary validation for publish payloads.

Regression suite for a class of silent data loss: a publish whose interaction
objects never bound was accepted with 200 and stored rows with ``content=''``,
producing no learnings and no error anywhere. Three defects combined to hide it:

1. Every ``InteractionData`` field has a default and unknown keys were ignored
   with no trace, so a mis-keyed ``content`` yielded a valid empty interaction.
2. The precondition guard meant to catch empty interactions was dead code
   (truthiness test against a truthy ``StrEnum`` member).
3. On the default async path the precondition result is discarded entirely.

The per-interaction rules therefore live on the request model, which is the
only layer that fires on both the sync and the background-task path. Unknown
keys are *warned*, not rejected -- see ``InteractionData.model_config`` for why
forbidding them was tried and reverted.
"""

from typing import Any

import pytest
from pydantic import ValidationError

from reflexio.models.api_schema.common import ToolUsed
from reflexio.models.api_schema.domain.entities import (
    CONTENT_BEARING_FIELD_NAMES,
    Citation,
    RetrievedLearning,
)
from reflexio.models.api_schema.domain.enums import UserActionType
from reflexio.models.api_schema.service_schemas import (
    InteractionData,
    PublishUserInteractionRequest,
)

# One populated sample per content-bearing field. Keyed by field name so the
# acceptance test below can be driven from CONTENT_BEARING_FIELD_NAMES itself:
# adding a field to the predicate without covering it here fails collection,
# which is how `citations` and `retrieved_learnings` previously went untested.
_CONTENT_SAMPLES: dict[str, Any] = {
    "content": "hello",
    "shadow_content": "shadow variant",
    "expert_content": "expert answer",
    "interacted_image_url": "https://example.com/i.png",
    "image_encoding": "base64data",
    "tools_used": [ToolUsed(tool_name="search", tool_data={"q": "x"})],
    "citations": [Citation(kind="profile", real_id="p-1")],
    "retrieved_learnings": [RetrievedLearning(kind="profile", learning_id="p-1")],
}


def _publish(
    interactions: list[InteractionData] | list[dict[str, Any]],
) -> PublishUserInteractionRequest:
    return PublishUserInteractionRequest.model_validate(
        {
            "user_id": "test-user",
            "session_id": "test-session",
            "interaction_data_list": interactions,
        }
    )


class TestUnknownFieldsWarned:
    """Unknown keys are dropped as before, but no longer silently.

    Forbidding them was tried and reverted: the first-party plugins build their
    wire payload with a denylist, so every turn carries request-level keys such
    as ``user_id``. Rejecting those wedged the plugins into a silent retry loop.
    """

    def test_miskeyed_content_is_recorded_not_silently_dropped(self):
        interaction = InteractionData.model_validate(
            {"content": "real text", "Content": "an important user preference"}
        )
        assert interaction.unknown_field_names() == ["Content"]

    def test_unknown_key_does_not_leak_into_model_dump(self):
        # ``extra="allow"`` is only a means of seeing the key; it must not
        # travel onward into storage or a re-serialised client request.
        interaction = InteractionData.model_validate(
            {"content": "real text", "Content": "x"}
        )
        assert "Content" not in interaction.model_dump()

    def test_request_surfaces_warning_naming_the_field(self):
        request = _publish([{"content": "real", "Content": "typo"}])
        assert request.payload_warnings() == [
            "interaction_data_list[0]: ignored unrecognised field(s) Content"
        ]

    def test_warning_names_the_key_but_never_its_value(self):
        # The value is caller payload -- potentially Customer Content -- and
        # must not reach a log or a response body (invariant from #377).
        request = _publish([{"content": "real", "Content": "SECRET-VALUE-XYZ"}])
        assert "SECRET-VALUE-XYZ" not in " ".join(request.payload_warnings())

    def test_clean_payload_produces_no_warnings(self):
        assert _publish([{"role": "User", "content": "hello"}]).payload_warnings() == []

    def test_plugin_wire_shape_is_accepted(self):
        # Regression for the P1 that reverted ``extra="forbid"``: both plugins
        # build the wire dict with a denylist, so every turn carries a
        # request-level ``user_id``. Rejecting it made the adapter swallow the
        # error and never advance its watermark -- publishing stopped forever.
        request = _publish(
            [{"content": "how do I do X?", "user_id": "proj-a", "role": "User"}]
        )
        assert request.interaction_data_list[0].content == "how do I do X?"

    def test_correctly_keyed_payload_still_binds(self):
        # Send a NON-default role: asserting role == "User" could not tell a
        # bound value from the schema default, which is the very tell-tale that
        # exposed the original bug.
        request = _publish([{"role": "Agent", "content": "hello"}])
        assert request.interaction_data_list[0].content == "hello"
        assert request.interaction_data_list[0].role == "Agent"


class TestEmptyInteractions:
    """Empty rows are skipped; only a wholly empty batch is fatal.

    Making a single empty row fatal was implemented and reverted: both
    first-party plugins append an empty ``Assistant`` placeholder
    unconditionally, so one empty row rejected the batch containing the real
    user turn, and their adapters swallow the error without advancing the
    publish watermark -- retrying the same doomed batch forever.
    """

    def test_all_empty_batch_is_rejected(self):
        # The actual production incident: 50 of 50 rows carried nothing.
        with pytest.raises(ValidationError, match="every interaction is empty"):
            _publish([{} for _ in range(50)])

    def test_single_empty_interaction_is_rejected_because_it_is_the_whole_batch(self):
        with pytest.raises(ValidationError, match="every interaction is empty"):
            _publish([InteractionData()])

    def test_whitespace_only_content_does_not_count(self):
        with pytest.raises(ValidationError, match="every interaction is empty"):
            _publish([{"content": "   \n\t "}])

    def test_user_action_none_does_not_count_as_content(self):
        # UserActionType.NONE is the truthy string "none"; treating it as
        # content is the dead-guard bug this suite exists for.
        with pytest.raises(ValidationError, match="every interaction is empty"):
            _publish([{"user_action": "none"}])

    def test_empty_row_beside_a_real_turn_is_skipped_not_fatal(self):
        # Regression for the plugin wedge: the real turn must still publish.
        request = _publish(
            [{"role": "User", "content": "REAL"}, {"role": "Assistant", "content": ""}]
        )
        assert [i.content for i in request.interaction_data_list] == ["REAL"]
        assert any("skipped 1 empty" in w for w in request.payload_warnings())

    def test_skip_is_reported_with_a_count(self):
        request = _publish([{"content": "a"}, {}, {}, {"content": "b"}])
        assert len(request.interaction_data_list) == 2
        assert any("skipped 2 empty" in w for w in request.payload_warnings())


class TestSiblingRulesEnforcedAtBoundary:
    """The other two per-interaction rules must 422 too.

    They previously lived only in ``precondition_checks``, whose result is
    discarded on the async path -- so they returned 200 "queued" and then
    silently refused the write.
    """

    def test_user_action_without_description_is_rejected(self):
        with pytest.raises(ValidationError, match="user_action_description"):
            _publish([{"content": "hi", "user_action": "click"}])

    def test_image_url_and_encoding_together_is_rejected(self):
        with pytest.raises(ValidationError, match="cannot both be set"):
            _publish(
                [
                    {
                        "interacted_image_url": "https://example.com/i.png",
                        "image_encoding": "base64data",
                    }
                ]
            )


class TestLegitimateTurnsStillAccepted:
    """Guards against over-rejecting: these all carry real information."""

    def test_content_bearing_field_set_is_pinned(self):
        """Pin the SET, not just iterate it.

        The parametrize below is driven by CONTENT_BEARING_FIELD_NAMES, so
        deleting a field from that tuple silently deletes its own coverage --
        a mutation dropping `citations` and `retrieved_learnings` killed zero
        tests. This assertion is the independent anchor: narrowing the
        predicate now fails here.
        """
        assert set(CONTENT_BEARING_FIELD_NAMES) == {
            "content",
            "shadow_content",
            "expert_content",
            "interacted_image_url",
            "image_encoding",
            "tools_used",
            "citations",
            "retrieved_learnings",
        }

    @pytest.mark.parametrize("field_name", CONTENT_BEARING_FIELD_NAMES)
    def test_each_content_bearing_field_alone_is_accepted(self, field_name: str):
        # Driven from the predicate's own field tuple, so a field can never be
        # added to `carries_content()` without acceptance coverage.
        assert field_name in _CONTENT_SAMPLES, (
            f"{field_name} is content-bearing but has no sample here"
        )
        request = _publish([{field_name: _CONTENT_SAMPLES[field_name]}])
        assert len(request.interaction_data_list) == 1

    def test_user_action_with_description_is_accepted(self):
        request = _publish(
            [
                {
                    "user_action": UserActionType.CLICK,
                    "user_action_description": "clicked submit",
                }
            ]
        )
        assert len(request.interaction_data_list) == 1

    def test_real_conversation_is_accepted(self):
        request = _publish(
            [
                {"role": "User", "content": "I prefer Postgres over MySQL."},
                {"role": "Agent", "content": "Noted, I'll target Postgres."},
            ]
        )
        assert len(request.interaction_data_list) == 2


class TestWarningsAreComputedBeforeFiltering:
    """Warnings must describe the payload the CALLER sent.

    Regression: warnings were originally computed after empty rows were
    filtered out, which silently defeated the feature in its primary case --
    a mis-keyed ``content`` yields an *empty* interaction, so the row carrying
    the typo was exactly the row removed, and its warning vanished. Surviving
    indices were renumbered too, pointing at a different row than was sent.
    """

    def test_miskeyed_content_on_an_otherwise_empty_row_is_still_reported(self):
        request = _publish([{"content": "real"}, {"Content": "IMPORTANT"}])
        assert any("Content" in w for w in request.payload_warnings()), (
            request.payload_warnings()
        )

    def test_reported_index_matches_what_the_caller_sent(self):
        # The caller put "Oops" at index 1; index 0 is empty and gets skipped.
        request = _publish([{}, {"content": "real", "Oops": "typo"}])
        assert any(
            "interaction_data_list[1]" in w and "Oops" in w
            for w in request.payload_warnings()
        ), request.payload_warnings()

    def test_skip_warning_names_the_original_indices(self):
        request = _publish([{"content": "a"}, {}, {"content": "b"}, {}])
        skip = next(w for w in request.payload_warnings() if "skipped" in w)
        assert "1, 3" in skip, skip


class TestWarningsAreBoundedAndSafe:
    """Unknown key names are caller data: bound them and strip control chars."""

    def test_control_characters_cannot_forge_log_lines(self):
        request = _publish([{"content": "x", "ev\nFAKE LOG LINE": "v"}])
        joined = " ".join(request.payload_warnings())
        assert "\n" not in joined
        assert "?" in joined

    def test_total_warning_volume_is_capped(self):
        # Per-name caps alone do not bound the total: interaction_data_list
        # permits 1000 entries, which totalled ~350 KB before this cap.
        request = _publish(
            [{"content": f"t{i}", ("k" * 300) + str(i): "v"} for i in range(1000)]
        )
        warnings = request.payload_warnings()
        assert len(warnings) <= 21, len(warnings)
        assert sum(len(w) for w in warnings) < 10_000
        assert "more interaction(s)" in warnings[-1]

    def test_long_key_names_are_truncated(self):
        request = _publish([{"content": "x", "k" * 500: "v"}])
        assert all(len(w) < 300 for w in request.payload_warnings())

    def test_skip_summary_survives_the_entry_cap(self):
        # The batch-level "N interactions were dropped" fact must never be the
        # thing the cap discards -- it is more important than the 20th
        # per-interaction field warning. Capping the combined list swallowed it
        # whenever there were >= 20 unknown-field warnings.
        request = _publish(
            [{f"Bad{i}": "v"} for i in range(25)] + [{"content": "real"}]
        )
        warnings = request.payload_warnings()
        assert any("skipped 25 empty" in w for w in warnings), warnings
        assert len(warnings) <= 22, len(warnings)


class TestToolStatusIsCaptured:
    """``ToolUsed.status`` is real signal, not an unknown key.

    Both first-party plugins derive "success"/"error" from the tool response and
    send it on every tool entry. It was silently discarded, which made it both
    lost data and the largest single source of unknown-field warnings once
    nested capture began reporting it.
    """

    def test_status_binds_and_produces_no_warning(self):
        request = _publish(
            [{"content": "x", "tools_used": [{"tool_name": "Bash", "status": "error"}]}]
        )
        assert request.interaction_data_list[0].tools_used[0].status == "error"
        assert request.payload_warnings() == []

    def test_genuinely_unknown_nested_key_still_warns(self):
        request = _publish(
            [{"content": "x", "tools_used": [{"tool_name": "Bash", "stat": "typo"}]}]
        )
        assert any("tools_used[0].stat" in w for w in request.payload_warnings()), (
            request.payload_warnings()
        )


class TestLogSafety:
    """Caller-controlled values must not be able to forge log lines."""

    def test_request_id_control_characters_are_neutralised(self):
        from reflexio.models.api_schema.common import sanitise_for_log

        forged = "ok\nERROR Background publish rejected for org 99: FAKE"
        assert "\n" not in sanitise_for_log(forged)

    def test_request_id_length_is_bounded(self):
        from reflexio.models.api_schema.common import sanitise_for_log

        # request_id is NonEmptyStr: no length cap on the model, so the log
        # site must impose one.
        assert len(sanitise_for_log("z" * 5000)) < 100

    @pytest.mark.parametrize(
        "bad_status",
        [1, None, True, {"a": 1}, "a" * 101],
        ids=["int", "none", "bool", "dict", "over-long"],
    )
    def test_out_of_contract_status_never_rejects_the_batch(self, bad_status):
        """Declaring a field must not smuggle in the strictness we rejected.

        A strictly-typed `status` turned five previously-harmless values into a
        422 for the WHOLE publish. The plugin adapters swallow that and never
        advance their watermark, so the same batch retries forever — the exact
        failure `extra="allow"` exists to avoid.
        """
        request = _publish(
            [{"content": "x", "tools_used": [{"tool_name": "B", "status": bad_status}]}]
        )
        stored = request.interaction_data_list[0].tools_used[0].status
        assert isinstance(stored, str)
        assert len(stored) <= 100
