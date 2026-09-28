import time

import pytest

import reflexio.server.llm._provider_concurrency as pc


def _hold(model, started, release, results, idx):
    with pc.provider_slot(model):
        started.set()
        results[idx] = True
        release.wait(timeout=5)


def test_caps_concurrent_holders_per_provider(monkeypatch):
    # Patch values rather than reloading: reload changes exception class
    # identity while the generation module still references the original.
    monkeypatch.setattr(pc, "REFLEXIO_LLM_PROVIDER_MAX_CONCURRENCY", 2)
    monkeypatch.setattr(pc, "_per_provider_cap", {})
    monkeypatch.setattr(pc, "_semaphores", {})
    monkeypatch.setattr(pc, "_ACQUIRE_TIMEOUT_SECONDS", 0.3)
    # Force a deterministic provider key (avoid network/model lookups).
    monkeypatch.setattr(pc, "_provider_key", lambda _m: "openai")

    sem = pc._get_semaphore("openai")
    assert sem._value == 2  # BoundedSemaphore initial permits
    # Hold both permits.
    with pc.provider_slot("gpt-x"), pc.provider_slot("gpt-x"):
        assert sem._value == 0
        # A third acquire must FAIL OPEN after the bounded timeout (not block forever).
        t0 = time.monotonic()
        with pc.provider_slot("gpt-x"):
            waited = time.monotonic() - t0
        assert waited >= 0.3  # waited the bounded timeout, then proceeded


def test_second_provider_independent(monkeypatch):
    monkeypatch.setattr(pc, "_provider_key", lambda m: m)  # model name == provider
    a = pc._get_semaphore("prov_a")
    b = pc._get_semaphore("prov_b")
    assert a is not b


def test_generation_seam_imports_provider_slot():
    import reflexio.server.llm._litellm_text_generation as tg

    assert hasattr(tg, "provider_slot")


def test_embedding_seam_imports_provider_slot():
    import reflexio.server.llm._litellm_embedding as emb

    assert hasattr(emb, "provider_slot")


def test_unknown_provider_not_capped(monkeypatch):
    monkeypatch.setattr(pc, "_provider_key", lambda _m: None)
    # Should be a no-op context (never blocks, never raises).
    with pc.provider_slot("whatever"):
        pass


def test_fail_open_emits_log(monkeypatch, caplog):
    monkeypatch.setattr(pc, "_provider_key", lambda _m: "openai")
    monkeypatch.setattr(pc, "_ACQUIRE_TIMEOUT_SECONDS", 0.2)
    monkeypatch.setattr(pc, "REFLEXIO_LLM_PROVIDER_MAX_CONCURRENCY", 1)
    # rebuild the semaphore registry to pick up cap=1
    pc._semaphores.clear()
    # Enter left-to-right: the first slot takes the lone permit, then the
    # second slot saturates and fails open (emitting the WARNING).
    with (
        pc.provider_slot("gpt-x"),
        caplog.at_level("WARNING"),
        pc.provider_slot("gpt-x"),  # saturated → fail open
    ):
        pass
    assert any(
        "provider_cap_saturated" in r.message or "saturat" in r.message.lower()
        for r in caplog.records
    )
    pc._semaphores.clear()


def _reset_registry():
    with pc._registry_lock:
        pc._semaphores.clear()


def test_fail_open_provider_proceeds_on_saturation(monkeypatch, caplog):
    _reset_registry()
    monkeypatch.setattr(pc, "REFLEXIO_LLM_PROVIDER_MAX_CONCURRENCY", 1)
    monkeypatch.setattr(pc, "_ACQUIRE_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(pc, "_fail_closed_providers", frozenset())
    monkeypatch.setattr(pc, "_per_provider_cap", {})
    monkeypatch.setattr(pc, "_provider_key", lambda _m: "openai")
    # Holds the only permit, then a second acquire saturates → fail-open
    # (proceeds, no raise).
    with pc.provider_slot("openai/gpt-4o"), pc.provider_slot("openai/gpt-4o"):
        pass


def test_fail_closed_provider_raises_on_saturation(monkeypatch):
    _reset_registry()
    monkeypatch.setattr(pc, "REFLEXIO_LLM_PROVIDER_MAX_CONCURRENCY", 1)
    monkeypatch.setattr(pc, "_ACQUIRE_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(pc, "_fail_closed_providers", frozenset({"zai"}))
    monkeypatch.setattr(pc, "_per_provider_cap", {})
    monkeypatch.setattr(pc, "_provider_key", lambda _m: "zai")
    with (
        pc.provider_slot("zai/glm-5.2"),
        pytest.raises(pc.ProviderCapSaturatedError),
        pc.provider_slot("zai/glm-5.2"),
    ):
        pass


def test_per_provider_cap_override(monkeypatch):
    _reset_registry()
    monkeypatch.setattr(pc, "REFLEXIO_LLM_PROVIDER_MAX_CONCURRENCY", 8)
    monkeypatch.setattr(pc, "_per_provider_cap", {"zai": 2})
    assert pc._max_concurrency_for_provider("zai") == 2
    assert pc._max_concurrency_for_provider("openai") == 8


@pytest.mark.parametrize("fail_closed", [False, True])
def test_search_deadline_bounds_saturated_provider_wait(monkeypatch, fail_closed):
    import threading

    from reflexio.server import search_runtime

    sem = threading.BoundedSemaphore(1)
    sem.acquire()
    monkeypatch.setattr(pc, "_provider_key", lambda _model: "test")
    monkeypatch.setattr(pc, "_get_semaphore", lambda _provider: sem)
    monkeypatch.setattr(pc, "_ACQUIRE_TIMEOUT_SECONDS", 0.8)
    monkeypatch.setattr(
        pc,
        "_fail_closed_providers",
        frozenset({"test"}) if fail_closed else frozenset(),
    )
    scope = search_runtime.SearchScope(deadline=time.monotonic() + 0.04)
    token = search_runtime._scope.set(scope)
    started = time.monotonic()
    try:
        with (
            pytest.raises(search_runtime.SearchDeadlineError),
            pc.provider_slot("test"),
        ):
            pytest.fail("expired request started provider work")
        assert time.monotonic() - started < 0.5
        assert sem._value == 0  # The waiter must not release another caller's permit.
        assert scope.phase_counts["llm.provider_queue"] == 1
    finally:
        search_runtime._scope.reset(token)
        sem.release()


def test_provider_acquired_at_deadline_releases_its_permit(monkeypatch):
    from reflexio.server import search_runtime

    scope = search_runtime.SearchScope(deadline=time.monotonic() + 10)
    releases = []

    class Semaphore:
        def acquire(self, *, timeout):
            scope.deadline = time.monotonic() - 1
            return True

        def release(self):
            releases.append(True)

    monkeypatch.setattr(pc, "_provider_key", lambda _model: "test")
    monkeypatch.setattr(pc, "_get_semaphore", lambda _provider: Semaphore())
    token = search_runtime._scope.set(scope)
    try:
        with (
            pytest.raises(search_runtime.SearchDeadlineError),
            pc.provider_slot("test"),
        ):
            pytest.fail("provider entered after acquisition consumed the deadline")
        assert releases == [True]
    finally:
        search_runtime._scope.reset(token)


@pytest.mark.parametrize("scoped", [False, True])
def test_provider_wait_preserves_unscoped_and_disabled_behavior(monkeypatch, scoped):
    from reflexio.server import search_runtime

    waits = []

    class Semaphore:
        def acquire(self, *, timeout):
            waits.append(timeout)
            return False

    monkeypatch.setattr(pc, "_provider_key", lambda _model: "test")
    monkeypatch.setattr(pc, "_get_semaphore", lambda _provider: Semaphore())
    monkeypatch.setattr(pc, "_fail_closed_providers", frozenset())
    scope = search_runtime.SearchScope() if scoped else None
    token = search_runtime._scope.set(scope)
    try:
        with pc.provider_slot("test"):
            pass
        assert waits == [pc._ACQUIRE_TIMEOUT_SECONDS]
    finally:
        search_runtime._scope.reset(token)
