"""The session LLM mock must be suspendable per test, not per command line.

``configure_llm_mock`` decides whether to patch from the invocation path
(``_is_e2e_test_run`` scans ``config.args`` for ``e2e_tests``), so before
``unpatched_litellm`` existed a live-provider test was mocked or not depending
on how the command was typed. ``pytest -m e2e`` carries no path, so the patch
stayed on and the test asserted against canned mock text.
"""

from __future__ import annotations

import os

import pytest

from reflexio.test_support import llm_mock
from reflexio.test_support.llm_mock import litellm_is_patched, unpatched_litellm


@pytest.fixture(autouse=True)
def _requires_the_session_patch() -> None:
    """These assertions only mean anything while the session mock is active."""
    if not litellm_is_patched():
        pytest.skip("session mock is not active in this invocation")


def test_suspends_the_patch_then_restores_it() -> None:
    assert litellm_is_patched()

    with unpatched_litellm():
        assert not litellm_is_patched(), (
            "the real litellm.completion must be in place inside the context"
        )

    assert litellm_is_patched(), "the session patch must be restored on exit"


def test_restores_the_patch_when_the_body_raises() -> None:
    """A failing live test must not leave the rest of the session unmocked."""
    with pytest.raises(RuntimeError, match="boom"), unpatched_litellm():
        raise RuntimeError("boom")

    assert litellm_is_patched()


def test_clears_and_restores_the_mock_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """Service code branches on MOCK_LLM_RESPONSE to skip LLM work entirely."""
    monkeypatch.setenv("MOCK_LLM_RESPONSE", "true")

    with unpatched_litellm():
        assert os.environ.get("MOCK_LLM_RESPONSE") is None

    assert os.environ.get("MOCK_LLM_RESPONSE") == "true"


def test_is_a_noop_when_no_session_patcher_is_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An e2e-path invocation never starts one; suspending must still be safe."""
    monkeypatch.setattr(llm_mock, "_litellm_patcher", None)

    with unpatched_litellm():
        pass
