"""D7: torch intra-op thread pinning is dormant unless explicitly configured."""

from __future__ import annotations

import pytest

from reflexio.server.llm.providers import nomic_embedding_provider
from reflexio.server.llm.providers.nomic_embedding_provider import (
    _maybe_pin_torch_threads,
)


@pytest.fixture(autouse=True)
def _reset_pinned_flag() -> None:
    """Reset the once-per-process pin guard so each test starts fresh."""
    nomic_embedding_provider._torch_threads_pinned = False


def _patch_torch(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Patch ``torch.set_num_threads`` and return the list it records into."""
    import torch

    calls: list[int] = []
    monkeypatch.setattr(torch, "set_num_threads", lambda n: calls.append(n))
    return calls


def test_unset_does_not_touch_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default (unset) → torch.set_num_threads is never called."""
    monkeypatch.delenv("REFLEXIO_EMBED_TORCH_THREADS", raising=False)
    calls = _patch_torch(monkeypatch)
    _maybe_pin_torch_threads()
    assert calls == []


def test_set_pins_threads_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """A positive value pins exactly once, even across repeated calls."""
    monkeypatch.setenv("REFLEXIO_EMBED_TORCH_THREADS", "1")
    calls = _patch_torch(monkeypatch)
    _maybe_pin_torch_threads()
    _maybe_pin_torch_threads()  # second call must be a no-op (already pinned)
    assert calls == [1]


def test_set_two_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tested alternative (2) is applied verbatim."""
    monkeypatch.setenv("REFLEXIO_EMBED_TORCH_THREADS", "2")
    calls = _patch_torch(monkeypatch)
    _maybe_pin_torch_threads()
    assert calls == [2]


def test_invalid_value_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-integer value leaves torch untouched (no crash)."""
    monkeypatch.setenv("REFLEXIO_EMBED_TORCH_THREADS", "not-a-number")
    calls = _patch_torch(monkeypatch)
    _maybe_pin_torch_threads()
    assert calls == []


def test_non_positive_value_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-positive value leaves torch untouched."""
    monkeypatch.setenv("REFLEXIO_EMBED_TORCH_THREADS", "0")
    calls = _patch_torch(monkeypatch)
    _maybe_pin_torch_threads()
    assert calls == []
