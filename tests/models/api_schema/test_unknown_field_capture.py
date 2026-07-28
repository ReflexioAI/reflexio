"""The capture foundation for reporting unrecognised publish fields.

A publish whose interaction objects never bound was accepted with 200 and stored
rows with ``content=''``. The incident fix (#384) makes a wholly empty batch a
422; this layer tells the caller *which field* they mis-keyed, which is what
turns "no learnings appeared" from a mystery into a one-line correction.

Everything here is bounded and content-free by construction, because the inputs
are caller-controlled and reach both an HTTP response and a shared log stream.
"""

import pytest
from pydantic import ValidationError

from reflexio.models.api_schema.common import (
    CapturesUnknownFields,
    ToolUsed,
)
from reflexio.models.api_schema.domain.entities import (
    _MAX_WARNING_ENTRIES,
    Citation,
    InteractionData,
    PublishUserInteractionRequest,
    RetrievedLearning,
    _cap_warning_list,
    _summarise_unknown_names,
)


class _Sample(CapturesUnknownFields):
    known: str = ""


def _request(interactions: list[dict], **extra) -> PublishUserInteractionRequest:
    """Build a publish request around the given raw interaction dicts."""
    return PublishUserInteractionRequest(
        user_id="u",
        session_id="s",
        interaction_data_list=interactions,  # type: ignore[arg-type]
        **extra,
    )


class TestCaptureAndStrip:
    def test_unknown_key_is_recorded(self):
        assert _Sample.model_validate(
            {"known": "x", "Bogus": 1}
        ).unknown_field_names() == ["Bogus"]

    def test_unknown_key_does_not_survive_into_model_dump(self):
        # extra="allow" is only a means of SEEING the key. If it travelled
        # onward it would reach storage and a re-serialised client request.
        assert "Bogus" not in _Sample.model_validate({"Bogus": 1}).model_dump()

    def test_names_are_sorted_for_stable_output(self):
        captured = _Sample.model_validate({"b": 1, "a": 2}).unknown_field_names()
        assert captured == ["a", "b"]

    def test_clean_payload_records_nothing(self):
        assert _Sample.model_validate({"known": "x"}).unknown_field_names() == []

    def test_unknown_keys_are_never_rejected(self):
        """Forbidding them wedged every first-party plugin publish."""
        assert _Sample.model_validate({"user_id": "p", "known": "x"}).known == "x"


class TestToolUsedStatus:
    """`status` is real signal both plugins send, previously discarded."""

    def test_status_binds(self):
        assert ToolUsed.model_validate(
            {"tool_name": "B", "status": "error"}
        ).status == ("error")

    @pytest.mark.parametrize(
        "bad",
        [1, None, True, {"a": 1}, "a" * 200],
        ids=["int", "none", "bool", "dict", "long"],
    )
    def test_out_of_contract_status_is_coerced_never_rejected(self, bad):
        # A strict declaration turned these into a 422 for the WHOLE batch,
        # which the plugin adapters swallow while never advancing their
        # watermark -- the batch then retries forever.
        stored = ToolUsed.model_validate({"tool_name": "B", "status": bad}).status
        assert isinstance(stored, str)
        assert len(stored) <= 100

    def test_genuinely_unknown_nested_key_is_still_captured(self):
        tool = ToolUsed.model_validate({"tool_name": "B", "stat": "typo"})
        assert tool.unknown_field_names() == ["stat"]


# Every nested model an ``InteractionData`` can carry. All three inherit the
# capture mixin, so all three must capture-and-strip identically -- testing only
# ``ToolUsed`` let the other two regress silently.
_MIXIN_SUBCLASS_PAYLOADS = [
    (ToolUsed, {"tool_name": "B"}),
    (Citation, {"kind": "profile", "real_id": "1"}),
    (RetrievedLearning, {"kind": "profile", "learning_id": "1"}),
]


class TestEveryMixinSubclassCapturesAndStrips:
    @pytest.mark.parametrize(
        ("model", "payload"),
        _MIXIN_SUBCLASS_PAYLOADS,
        ids=[model.__name__ for model, _ in _MIXIN_SUBCLASS_PAYLOADS],
    )
    def test_unknown_key_is_captured_and_stripped(self, model, payload):
        instance = model.model_validate({**payload, "zzz": "typo"})
        assert instance.unknown_field_names() == ["zzz"]
        assert "zzz" not in instance.model_dump()


class TestBoundedRendering:
    def test_names_per_entry_are_capped_with_a_suffix(self):
        rendered = _summarise_unknown_names([f"k{i}" for i in range(9)])
        assert rendered.endswith("+4 more")
        assert rendered.count(",") == 5

    def test_control_characters_cannot_forge_a_log_line(self):
        assert "\n" not in _summarise_unknown_names(["a\nERROR forged"])

    def test_long_names_are_truncated(self):
        assert len(_summarise_unknown_names(["z" * 500])) < 100

    def test_entry_count_is_capped_with_an_overflow_entry(self):
        capped = _cap_warning_list([f"w{i}" for i in range(30)])
        assert len(capped) == 21
        assert "10 more interaction(s)" in capped[-1]

    def test_under_the_cap_is_unchanged(self):
        assert _cap_warning_list(["a", "b"]) == ["a", "b"]

    def test_returns_a_copy_so_callers_can_append(self):
        original = ["a"]
        _cap_warning_list(original).append("x")
        assert original == ["a"]


class TestReportableUnknownFields:
    """``InteractionData.reportable_unknown_fields`` -- what is worth saying."""

    def test_nested_paths_are_composed_with_the_callers_index(self):
        """The path, literally, and from a non-zero index.

        A caller matching only bare names would not recognise the format, and
        an index taken from anything but ``enumerate`` points at the wrong row.
        """
        interaction = InteractionData.model_validate(
            {
                "content": "real",
                "tools_used": [{"tool_name": "t", "zzz": 1}],
                "citations": [
                    {"kind": "profile", "real_id": "1"},
                    {"kind": "profile", "real_id": "2", "zzz": 1},
                ],
            }
        )
        assert interaction.reportable_unknown_fields() == [
            "citations[1].zzz",
            "tools_used[0].zzz",
        ]

    def test_benign_keys_are_suppressed_but_the_filter_stays_selective(self):
        """``user_id``/``session_id`` are silent; anything else is not.

        Both plugins duplicate the request-level identifiers onto every turn, so
        warning about them would emit an entry per interaction on every correct
        publish. Suppressing them must not degrade into suppressing everything,
        at either level -- hence a reported key sitting beside each benign one.
        """
        interaction = InteractionData.model_validate(
            {
                "content": "real",
                "user_id": "p",
                "session_id": "s",
                "zzz": 1,
                "tools_used": [{"tool_name": "t", "user_id": "p", "zzz": 1}],
            }
        )
        assert interaction.reportable_unknown_fields() == ["tools_used[0].zzz", "zzz"]


class TestWarningsUseTheCallersIndices:
    def test_index_is_the_one_the_caller_sent_not_the_filtered_one(self):
        """Warnings are computed BEFORE empty rows are dropped.

        Computed afterwards, index 1 renumbers to 0 and the warning points the
        caller at a row they did not mis-key.
        """
        request = _request(
            [
                {"role": "Agent"},
                {"role": "User", "content": "hi", "zzz": 1},
            ]
        )
        field_warnings = [
            warning
            for warning in request.payload_warnings()
            if "unrecognised" in warning
        ]
        assert len(field_warnings) == 1
        assert field_warnings[0].startswith("interaction_data_list[1]: ")


class TestComposeOrder:
    def test_skip_summary_survives_a_capped_field_warning_list(self):
        """The cap is applied BEFORE the batch-level entries are appended.

        Capping the combined list swallowed "N interactions were dropped"
        whenever there were >= 20 field warnings -- the single most important
        fact about such a batch.
        """
        request = _request(
            [
                *(
                    {"role": "User", "content": f"turn {index}", "zzz": 1}
                    for index in range(25)
                ),
                {"role": "Agent"},
            ]
        )
        warnings = request.payload_warnings()
        # 20 field warnings + 1 overflow entry + 1 skip summary.
        assert len(warnings) == _MAX_WARNING_ENTRIES + 2
        assert warnings[-1].startswith("skipped 1 empty interaction(s)")
        assert "5 more interaction(s)" in warnings[_MAX_WARNING_ENTRIES]


class TestAllEmptyBatchNamesTheMisKeyedField:
    """The motivating incident: 50 rows keyed ``Content`` bind nothing."""

    @staticmethod
    def _all_empty_error_message(rows: int) -> str:
        with pytest.raises(ValidationError) as exc_info:
            _request([{"Content": f"turn {index}"} for index in range(rows)])
        # The rendered ``msg``, NOT ``str(exc)``: the latter echoes the whole
        # request body in ``input_value``, so a substring check against it
        # passes even with the naming removed entirely.
        messages = [error["msg"] for error in exc_info.value.errors()]
        assert len(messages) == 1, messages
        return messages[0]

    def test_error_names_the_field_that_bound_nothing(self):
        message = self._all_empty_error_message(1)
        assert "every interaction is empty" in message
        assert "Content" in message

    @pytest.mark.parametrize("rows", [50, 1_000], ids=["incident", "max_batch"])
    def test_error_stays_bounded_for_a_large_batch(self, rows):
        """One entry per row would make the 422 body grow with the batch.

        50 is the incident; 1000 is the largest batch the schema permits, and
        the size that shows the cap is doing the work — uncapped it renders
        roughly 60 KB into an error body.
        """
        assert len(self._all_empty_error_message(rows)) < 8_192


class TestRequestLevelUnknownKeys:
    """A top-level typo changes what the server DOES, not just what it stores."""

    def test_top_level_typo_is_reported(self):
        request = _request([{"content": "real"}], **{"forceExtraction": True})
        assert any(
            warning.startswith("publish request: ") and "forceExtraction" in warning
            for warning in request.payload_warnings()
        ), request.payload_warnings()

    def test_top_level_typo_survives_a_capped_field_warning_list(self):
        """Appended outside the cap, so per-interaction noise cannot bury it."""
        request = _request(
            [
                {"role": "User", "content": f"turn {index}", "zzz": 1}
                for index in range(25)
            ],
            **{"forceExtraction": True},
        )
        assert any(
            warning.startswith("publish request: ")
            for warning in request.payload_warnings()
        ), request.payload_warnings()

    def test_top_level_typo_does_not_survive_into_model_dump(self):
        """Otherwise the SDK re-sends it and the server double-reports it."""
        request = _request([{"content": "real"}], **{"forceExtraction": True})
        assert "forceExtraction" not in request.model_dump()
