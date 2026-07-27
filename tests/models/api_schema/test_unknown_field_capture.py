"""The capture foundation for reporting unrecognised publish fields.

A publish whose interaction objects never bound was accepted with 200 and stored
rows with ``content=''``. The incident fix (#384) makes a wholly empty batch a
422; this layer tells the caller *which field* they mis-keyed, which is what
turns "no learnings appeared" from a mystery into a one-line correction.

Everything here is bounded and content-free by construction, because the inputs
are caller-controlled and reach both an HTTP response and a shared log stream.
"""

import pytest

from reflexio.models.api_schema.common import (
    CapturesUnknownFields,
    ToolUsed,
    cap_warning_list,
    summarise_unknown_names,
)


class _Sample(CapturesUnknownFields):
    known: str = ""


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


class TestBoundedRendering:
    def test_names_per_entry_are_capped_with_a_suffix(self):
        rendered = summarise_unknown_names([f"k{i}" for i in range(9)])
        assert rendered.endswith("+4 more")
        assert rendered.count(",") == 5

    def test_control_characters_cannot_forge_a_log_line(self):
        assert "\n" not in summarise_unknown_names(["a\nERROR forged"])

    def test_long_names_are_truncated(self):
        assert len(summarise_unknown_names(["z" * 500])) < 100

    def test_entry_count_is_capped_with_an_overflow_entry(self):
        capped = cap_warning_list([f"w{i}" for i in range(30)])
        assert len(capped) == 21
        assert "10 more interaction(s)" in capped[-1]

    def test_under_the_cap_is_unchanged(self):
        assert cap_warning_list(["a", "b"]) == ["a", "b"]

    def test_returns_a_copy_so_callers_can_append(self):
        original = ["a"]
        cap_warning_list(original).append("x")
        assert original == ["a"]
