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
only layer that fires on both the sync and the background-task path.

Unknown keys stay silently ignored here, deliberately. Rejecting them was
implemented and reverted -- it broke every first-party plugin publish -- and
*reporting* them to the caller is a separate change; this one fixes the
incident.
"""

from typing import Any

import pytest
from pydantic import ValidationError

from reflexio.models.api_schema.common import ToolUsed, sanitise_for_log
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
        assert "skipped 1 empty" in (request.skipped_empty_summary() or "")

    def test_skip_is_summarised_with_original_indices(self):
        request = _publish([{"content": "a"}, {}, {}, {"content": "b"}])
        assert len(request.interaction_data_list) == 2
        summary = request.skipped_empty_summary()
        assert summary is not None
        assert "skipped 2 empty" in summary
        # Indices are the caller's, not the filtered list's.
        assert "1, 2" in summary, summary


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


class TestLogSafety:
    """`sanitise_for_log` is a security control, so pin it.

    Any caller-supplied value reaching a log record needs it: a newline forges a
    line in a shared multi-tenant stream that Sentry ingests, and an unbounded
    value bloats the record. `request_id` qualifies — it is a `NonEmptyStr` with
    no length cap and no character restrictions.
    """

    def test_newline_cannot_forge_a_log_line(self):
        forged = "ok\nERROR Background publish rejected for org 99: FAKE"
        cleaned = sanitise_for_log(forged)
        assert "\n" not in cleaned
        assert "?" in cleaned

    def test_other_control_characters_are_replaced(self):
        assert "\r" not in sanitise_for_log("a\rb")
        assert "\t" not in sanitise_for_log("a\tb")

    def test_length_is_bounded(self):
        # No cap on the model, so the log site must impose one.
        assert len(sanitise_for_log("z" * 5000)) < 100

    def test_ordinary_values_pass_through_unchanged(self):
        for value in ("req-123", "a b c", "sess/42"):
            assert sanitise_for_log(value) == value

    def test_no_log_call_interpolates_a_raw_request_id(self):
        """Fix by class: no logger call may pass a bare `request_id`.

        Three sites log it -- two in generation_service, one in the publish
        route -- and the first pass fixed only the route. Scans the argument
        lines of every logger call for an unsanitised `request_id`.
        """
        from pathlib import Path

        root = Path(__file__).resolve().parents[3] / "reflexio"
        offenders: list[str] = []
        for path in root.rglob("*.py"):
            if "test" in path.name:
                continue
            lines = path.read_text().splitlines()
            in_call = False
            for number, line in enumerate(lines, 1):
                # error/warning/exception are the Sentry-ingested levels, and
                # info still lands in the shared stream.
                if any(
                    f"logger.{level}(" in line
                    for level in ("error", "warning", "exception", "info")
                ):
                    in_call = ")" not in line.split("logger.", 1)[1]
                    continue
                if not in_call:
                    continue
                # Match attribute access too: `plan.request_id,` is the same
                # caller-supplied value, and three logger.exception calls were
                # passing it raw while an exact-equality check passed clean.
                stripped = line.strip()
                if stripped == "request_id," or stripped.endswith(".request_id,"):
                    offenders.append(f"{path.relative_to(root)}:{number}")
                if ")" in line:
                    in_call = False
        # Two modules outside this change's blast radius carry the same
        # pre-existing pattern (10 sites at the time of writing). They are
        # allowlisted BY FILE rather than by line so the guard still fails for
        # any NEW file, and so the debt stays visible and countable instead of
        # being hidden by narrowing the scan. Sanitising them is a separate
        # change: this PR is the minimal publish-validation fix, and widening it
        # to unrelated modules is how a small fix turns into a large one.
        known_pending = {
            "server/services/publish_learning_worker.py",
            "server/services/extraction/resumable_agent.py",
        }
        new_offenders = [
            offender
            for offender in offenders
            if offender.rsplit(":", 1)[0] not in known_pending
        ]
        assert new_offenders == [], (
            "these logger calls pass an unsanitised caller-supplied request_id;"
            f" wrap with sanitise_for_log(): {new_offenders}"
        )
        # And the known set must not grow silently either.
        assert len(offenders) <= 10, (
            f"pre-existing unsanitised request_id sites grew to {len(offenders)}:"
            f" {offenders}"
        )
