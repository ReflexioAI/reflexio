import time

import pytest

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
