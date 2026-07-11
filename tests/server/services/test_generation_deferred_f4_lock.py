"""Focused F4 same-user guard test for the durable-learning compute split.

Exercises the per-user in-progress lock primitive that
``GenerationService.compute_deferred_learning`` acquires FIRST (before any LLM)
and ``emit_deferred_learning_side_effects`` releases post-commit:

    acquire(req1)                 -> True   (first holder)
    acquire(req2, same user)      -> False  (contention — must not run compute)
    release(req1)                 -> lock free
    acquire(req2)                 -> True   (reclaimable after release)

This is the F4 mechanism that keeps two durable workers processing two jobs for
the SAME user from racing a duplicate write. It uses the atomic DB-backed
``try_acquire_in_progress_lock`` / ``clear_in_progress_lock_if_owner`` storage
terminals (distinct from the learning-write terminals the compute purity
contract forbids), on a dedicated ``durable_learning`` per-user key. No LLM /
extraction / scheduler is involved — the guard is tested at the primitive level.
"""

from __future__ import annotations

import tempfile

from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.services.generation_service import (
    _DURABLE_LEARNING_LOCK_SERVICE,
    GenerationService,
)


def _make_gen(org_id: str, tmp_dir: str) -> GenerationService:
    ctx = RequestContext(org_id=org_id, storage_base_dir=tmp_dir)
    return GenerationService(
        llm_client=LiteLLMClient(LiteLLMConfig(model="gpt-4o-mini")),
        request_context=ctx,
    )


def test_deferred_learning_same_user_lock_serializes():
    """acquire → same-user re-acquire denied → release → re-acquire (F4)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        gen = _make_gen("org_f4_lock", tmp_dir)
        user_id = "u_f4"

        # First holder acquires.
        assert (
            gen._acquire_durable_learning_lock(user_id=user_id, request_id="req1")
            is True
        )

        # A second, DIFFERENT request for the SAME user is denied while held.
        assert (
            gen._acquire_durable_learning_lock(user_id=user_id, request_id="req2")
            is False
        )

        # The holder's own re-acquire is idempotent (still True).
        assert (
            gen._acquire_durable_learning_lock(user_id=user_id, request_id="req1")
            is True
        )

        # A DIFFERENT user is never blocked by this user's lock.
        assert (
            gen._acquire_durable_learning_lock(user_id="other_user", request_id="reqX")
            is True
        )

        # Release by the holder frees the lock; the contender can now acquire.
        gen._release_durable_learning_lock(user_id=user_id, request_id="req1")
        assert (
            gen._acquire_durable_learning_lock(user_id=user_id, request_id="req2")
            is True
        )


def test_release_by_non_owner_does_not_free_lock():
    """A non-owner release must not steal the lock from the current holder."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        gen = _make_gen("org_f4_lock2", tmp_dir)
        user_id = "u_f4b"

        assert (
            gen._acquire_durable_learning_lock(user_id=user_id, request_id="owner")
            is True
        )

        # A stale/other request releasing does NOT clear the owner's lock.
        gen._release_durable_learning_lock(user_id=user_id, request_id="not_owner")
        assert (
            gen._acquire_durable_learning_lock(user_id=user_id, request_id="contender")
            is False
        ), "non-owner release must leave the owner's lock intact"


def test_durable_lock_key_shape():
    """The per-user key is on a dedicated durable_learning service prefix."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        gen = _make_gen("org_key", tmp_dir)
        key = gen._durable_learning_lock_key("u1")
        assert key == f"{_DURABLE_LEARNING_LOCK_SERVICE}::org_key::u1::lock"
