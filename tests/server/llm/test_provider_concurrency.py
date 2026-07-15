import time

import reflexio.server.llm._provider_concurrency as pc


def _hold(model, started, release, results, idx):
    with pc.provider_slot(model):
        started.set()
        results[idx] = True
        release.wait(timeout=5)


def test_caps_concurrent_holders_per_provider(monkeypatch):
    # Force a small cap and a short acquire timeout via reload.
    monkeypatch.setenv("REFLEXIO_LLM_PROVIDER_MAX_CONCURRENCY", "2")
    import importlib

    importlib.reload(pc)
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
    importlib.reload(pc)


def test_second_provider_independent(monkeypatch):
    monkeypatch.setattr(pc, "_provider_key", lambda m: m)  # model name == provider
    a = pc._get_semaphore("prov_a")
    b = pc._get_semaphore("prov_b")
    assert a is not b


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
