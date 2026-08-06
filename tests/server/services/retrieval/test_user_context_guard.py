"""Tests for the explicit personalization opt-out guard."""

import pytest

from reflexio.server.services.retrieval.user_context_guard import (
    should_suppress_user_context,
)


@pytest.mark.parametrize(
    "query",
    [
        "Draft a troubleshooting guide. Do not personalize it.",
        "Keep the lesson generic, without personalization.",
        "DON'T USE MY PROFILE for this recommendation.",
        "Please don’t use my saved preferences in the response.",
        "I don't want you to use my conversation history for this task.",
        "Answer without using any of my previous context.",
        "Ignore prior user context and create a neutral overview.",
        "Avoid using my memory when writing the summary.",
        "Disregard my stored memories for this task.",
        "Never reference historical user information when answering.",
        "Do not use any personalization context; generate it for everyone.",
        "Skip profiles and answer only from the current request.",
        "Please\n  exclude   my profile from this one.",
        "Do not use my profile; personalize only from the current request.",
        "不要使用我的个人资料来回答。",
        "請勿參考我的偏好。",
        "无需个性化，直接回答问题。",
        "請直接回答，別使用我的歷史記錄。",
        "先说明背景。不要使用我的个人资料。",
    ],
)
def test_explicit_opt_outs_suppress_user_context(query: str) -> None:
    assert should_suppress_user_context(query) is True


@pytest.mark.parametrize(
    "query",
    [
        None,
        "",
        "   ",
        "Don't ignore my preferences.",
        "Use my profile to personalize this.",
        "Explain why personalization can be harmful.",
        "Compare recommendations with and without personalization.",
        "Explain the tradeoffs of no personalization.",
        "Do not use jargon.",
        "Create this for a public audience.",
        "This response is not without personalization.",
        "Never disregard my saved preferences.",
        "Do not exclude my profile.",
        "Do not, under any circumstances, ignore my preferences.",
        "Never again disregard my saved preferences.",
        "Create a course about why recommendation systems should not personalize ads.",
        "Explain why some agents do not personalize recommendations.",
        "Explain how agents operate without personalization.",
        "Do not personalize the introduction; use my profile for the rest.",
        "I never said 'don't use my profile'; please personalize this.",
        "解释为什么有些系统无需个性化。",
        "比較使用個人資料與不用個人資料的差異。",
    ],
)
def test_non_opt_outs_preserve_user_context(query: str | None) -> None:
    assert should_suppress_user_context(query) is False
