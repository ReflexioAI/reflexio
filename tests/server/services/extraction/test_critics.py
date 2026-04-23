"""Unit tests for critics + reconciler + summarize helper."""

from unittest.mock import MagicMock, patch

import pytest

from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.services.extraction.critics import (
    CriticCtx,
    CrossEntityFlag,
    MergeArgs,
    PlaybookCritic,
    ProfileCritic,
    Reconciler,
    ReconcilerCtx,
    VettedPlaybook,
    VettedProfile,
    summarize,
)
from reflexio.server.services.playbook.playbook_service_utils import (
    StructuredPlaybookContent,
)
from reflexio.server.services.profile.profile_generation_service_utils import (
    ProfileAddItem,
)


@pytest.fixture
def real_client(monkeypatch):
    """Real LiteLLMClient with anthropic creds — matches test_tools.py pattern."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)
    return LiteLLMClient(LiteLLMConfig(model="claude-sonnet-4-6"))


def _pm(render_return: str = "critic prompt") -> MagicMock:
    pm = MagicMock()
    pm.render_prompt.return_value = render_return
    return pm


# ---------------- summarize ---------------- #


def test_summarize_empty_returns_sentinel():
    assert summarize([]) == "(none)"


def test_summarize_caps_and_marks_truncated():
    items = [
        ProfileAddItem(content=f"c{i}", time_to_live="infinity") for i in range(30)
    ]
    s = summarize(items, limit=5)
    # 5 rendered lines + 1 truncation marker = 6 lines → 5 newlines
    assert s.count("\n") == 5
    assert "c0" in s
    assert "truncated" in s.lower()


def test_summarize_renders_source_span():
    items = [
        ProfileAddItem(
            content="User likes polars",
            time_to_live="infinity",
            source_span="I use polars not pandas",
        )
    ]
    s = summarize(items)
    assert "src=I use polars" in s


def test_summarize_falls_back_to_trigger_when_content_missing():
    items = [StructuredPlaybookContent(trigger="ship", content=None)]
    s = summarize(items)
    assert "ship" in s


# ---------------- ProfileCritic ---------------- #


def test_profile_critic_accept_and_flag(real_client, tool_call_completion):
    """Critic accepts one candidate and flags a cross-entity conflict."""
    make_tc, _ = tool_call_completion
    cand = ProfileAddItem(content="User uses polars.", time_to_live="infinity")
    responses = [
        make_tc("accept", {"candidate_index": 0}),
        make_tc(
            "flag_cross_entity_conflict",
            {"candidate_index": 0, "reason": "contradicts playbook #2"},
        ),
        make_tc("finish", {}),
    ]
    critic = ProfileCritic(client=real_client, prompt_manager=_pm())
    with patch("litellm.completion", side_effect=responses):
        vetted, flags = critic.review([cand], other_lane_summary="- b0\n- b1")

    assert len(vetted) == 1
    assert isinstance(vetted[0], VettedProfile)
    assert vetted[0].content == "User uses polars."
    assert len(flags) == 1
    assert flags[0].reason.startswith("contradicts")
    assert flags[0].lane == "profile"


def test_profile_critic_refine_edits_and_accepts(real_client, tool_call_completion):
    """Refine tool edits content + time_to_live, producing a vetted item."""
    make_tc, _ = tool_call_completion
    cand = ProfileAddItem(content="User uses polars.", time_to_live="one_day")
    responses = [
        make_tc(
            "refine",
            {
                "candidate_index": 0,
                "content": "User prefers polars over pandas.",
                "time_to_live": "infinity",
                "notes": "confidence=0.9",
            },
        ),
        make_tc("finish", {}),
    ]
    critic = ProfileCritic(client=real_client, prompt_manager=_pm())
    with patch("litellm.completion", side_effect=responses):
        vetted, flags = critic.review([cand], other_lane_summary="(none)")

    assert vetted[0].content == "User prefers polars over pandas."
    assert vetted[0].time_to_live == "infinity"
    assert vetted[0].notes == "confidence=0.9"
    assert flags == []


def test_profile_critic_reject_does_not_vet(real_client, tool_call_completion):
    make_tc, _ = tool_call_completion
    cand = ProfileAddItem(content="User might use pandas.", time_to_live="infinity")
    responses = [
        make_tc("reject", {"candidate_index": 0, "reason": "speculative"}),
        make_tc("finish", {}),
    ]
    critic = ProfileCritic(client=real_client, prompt_manager=_pm())
    with patch("litellm.completion", side_effect=responses):
        vetted, flags = critic.review([cand], other_lane_summary="(none)")

    assert vetted == []
    assert flags == []


def test_profile_critic_handles_out_of_range_index(real_client, tool_call_completion):
    make_tc, _ = tool_call_completion
    cand = ProfileAddItem(content="a", time_to_live="infinity")
    responses = [
        make_tc("accept", {"candidate_index": 99}),  # out of range
        make_tc("accept", {"candidate_index": 0}),
        make_tc("finish", {}),
    ]
    critic = ProfileCritic(client=real_client, prompt_manager=_pm())
    with patch("litellm.completion", side_effect=responses):
        vetted, _ = critic.review([cand], other_lane_summary="(none)")

    # Out-of-range is reported as an error to the model but doesn't crash.
    assert len(vetted) == 1


# ---------------- PlaybookCritic ---------------- #


def test_playbook_critic_refine_and_finish(real_client, tool_call_completion):
    make_tc, _ = tool_call_completion
    cand = StructuredPlaybookContent(trigger="user says 'ship'", content="skip tests")
    responses = [
        make_tc(
            "refine",
            {
                "candidate_index": 0,
                "trigger": "user types 'ship'",
                "content": "skip integration tests only",
                "rationale": "unit tests remain valuable",
            },
        ),
        make_tc("finish", {}),
    ]
    critic = PlaybookCritic(client=real_client, prompt_manager=_pm())
    with patch("litellm.completion", side_effect=responses):
        vetted, flags = critic.review([cand], other_lane_summary="(none)")

    assert len(vetted) == 1
    assert isinstance(vetted[0], VettedPlaybook)
    assert vetted[0].trigger == "user types 'ship'"
    assert vetted[0].rationale == "unit tests remain valuable"
    assert flags == []


# ---------------- Reconciler ---------------- #


def test_reconciler_no_flags_is_noop(real_client):
    """With zero flags, the reconciler returns inputs without calling the LLM."""
    profs = [VettedProfile(content="a", time_to_live="infinity")]
    pbs = [VettedPlaybook(trigger="t", content="c")]
    rec = Reconciler(client=real_client, prompt_manager=_pm())
    out_p, out_b = rec.resolve(profs, pbs, flags=[])
    assert out_p == profs
    assert out_b == pbs


def test_reconciler_supersede_drops_profile(real_client, tool_call_completion):
    make_tc, _ = tool_call_completion
    profs = [VettedProfile(content="old", time_to_live="infinity")]
    pbs = [VettedPlaybook(trigger="t", content="c", rationale="r")]
    flags = [
        CrossEntityFlag(
            candidate_index=0, reason="pb contradicts profile", lane="profile"
        )
    ]
    responses = [
        make_tc(
            "supersede",
            {
                "keep_lane": "playbook",
                "keep_index": 0,
                "drop_lane": "profile",
                "drop_index": 0,
            },
        ),
        make_tc("finish", {}),
    ]
    rec = Reconciler(client=real_client, prompt_manager=_pm())
    with patch("litellm.completion", side_effect=responses):
        out_p, out_b = rec.resolve(profs, pbs, flags)
    assert out_p == []
    assert len(out_b) == 1


def test_reconciler_merge_updates_kept_content(real_client, tool_call_completion):
    make_tc, _ = tool_call_completion
    profs = [VettedProfile(content="User likes polars.", time_to_live="infinity")]
    pbs = [VettedPlaybook(trigger="choose dataframe lib", content="prefer pandas")]
    flags = [
        CrossEntityFlag(
            candidate_index=0, reason="overlapping guidance", lane="playbook"
        )
    ]
    responses = [
        make_tc(
            "merge",
            {
                "keep_lane": "playbook",
                "keep_index": 0,
                "drop_lane": "profile",
                "drop_index": 0,
                "merged_content": "use polars — user prefers it",
            },
        ),
        make_tc("finish", {}),
    ]
    rec = Reconciler(client=real_client, prompt_manager=_pm())
    with patch("litellm.completion", side_effect=responses):
        out_p, out_b = rec.resolve(profs, pbs, flags)
    assert out_p == []  # profile side was dropped by the merge
    assert out_b[0].content == "use polars — user prefers it"


def test_reconciler_keep_both_preserves_both_lanes(real_client, tool_call_completion):
    make_tc, _ = tool_call_completion
    profs = [VettedProfile(content="a", time_to_live="infinity")]
    pbs = [VettedPlaybook(trigger="t", content="c")]
    flags = [CrossEntityFlag(candidate_index=0, reason="false alarm", lane="profile")]
    responses = [
        make_tc("keep_both", {"reason": "not actually contradictory"}),
        make_tc("finish", {}),
    ]
    rec = Reconciler(client=real_client, prompt_manager=_pm())
    with patch("litellm.completion", side_effect=responses):
        out_p, out_b = rec.resolve(profs, pbs, flags)
    assert len(out_p) == 1
    assert len(out_b) == 1


# ---------------- MergeArgs validator ---------------- #


def test_merge_args_rejects_same_lane():
    """MergeArgs must raise ValidationError when keep_lane == drop_lane."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="keep_lane and drop_lane must differ"):
        MergeArgs(
            keep_lane="profile",
            keep_index=0,
            drop_lane="profile",
            drop_index=1,
            merged_content="merged text",
        )


def test_merge_args_accepts_different_lanes():
    """MergeArgs with distinct lanes should construct without error."""
    args = MergeArgs(
        keep_lane="profile",
        keep_index=0,
        drop_lane="playbook",
        drop_index=1,
        merged_content="merged text",
    )
    assert args.keep_lane == "profile"
    assert args.drop_lane == "playbook"


# ---------------- ctx defaults ---------------- #


def test_critic_ctx_defaults():
    ctx = CriticCtx(candidates=[], lane="profile")
    assert ctx.vetted == []
    assert ctx.flags == []
    assert ctx.finished is False


def test_reconciler_ctx_default_not_finished():
    ctx = ReconcilerCtx(profiles=[], playbooks=[])
    assert ctx.finished is False
