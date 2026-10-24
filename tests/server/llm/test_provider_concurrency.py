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
        # A third caller cannot dispatch without an available permit.
        t0 = time.monotonic()
        with pytest.raises(pc.ProviderCapSaturatedError), pc.provider_slot("gpt-x"):
            pytest.fail("saturated provider dispatched")
        waited = time.monotonic() - t0
        assert waited >= 0.3  # admission remains bounded


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


def test_saturation_is_logged_without_dispatch(monkeypatch, caplog):
    monkeypatch.setattr(pc, "_provider_key", lambda _m: "openai")
    monkeypatch.setattr(pc, "_ACQUIRE_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(pc, "REFLEXIO_LLM_PROVIDER_MAX_CONCURRENCY", 1)
    monkeypatch.setattr(pc, "_semaphores", {})
    with (
        pc.provider_slot("gpt-x"),
        caplog.at_level("WARNING"),
        pytest.raises(pc.ProviderCapSaturatedError),
        pc.provider_slot("gpt-x"),
    ):
        pytest.fail("saturated provider dispatched")
    assert "provider_cap_saturated" in caplog.text
    assert "request not admitted" in caplog.text


def _reset_registry():
    with pc._registry_lock:
        pc._semaphores.clear()


@pytest.mark.parametrize("provider", ["openai", "zai"])
def test_every_resolved_provider_requires_a_permit(monkeypatch, provider):
    _reset_registry()
    monkeypatch.setattr(pc, "REFLEXIO_LLM_PROVIDER_MAX_CONCURRENCY", 1)
    monkeypatch.setattr(pc, "_ACQUIRE_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(pc, "_per_provider_cap", {})
    monkeypatch.setattr(pc, "_provider_key", lambda _m: provider)
    with (
        pc.provider_slot("example"),
        pytest.raises(pc.ProviderCapSaturatedError),
        pc.provider_slot("example"),
    ):
        pytest.fail("saturated provider dispatched")


def test_per_provider_cap_override(monkeypatch):
    _reset_registry()
    monkeypatch.setattr(pc, "REFLEXIO_LLM_PROVIDER_MAX_CONCURRENCY", 8)
    monkeypatch.setattr(pc, "_per_provider_cap", {"zai": 2})
    assert pc._max_concurrency_for_provider("zai") == 2
    assert pc._max_concurrency_for_provider("openai") == 8


def test_search_deadline_bounds_saturated_provider_wait(monkeypatch):
    import threading

    from reflexio.server import search_runtime

    sem = threading.BoundedSemaphore(1)
    sem.acquire()
    monkeypatch.setattr(pc, "_provider_key", lambda _model: "test")
    monkeypatch.setattr(pc, "_get_semaphore", lambda _provider: sem)
    monkeypatch.setattr(pc, "_ACQUIRE_TIMEOUT_SECONDS", 0.8)
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
def test_provider_wait_is_bounded_without_a_search_deadline(monkeypatch, scoped):
    from reflexio.server import search_runtime

    waits = []

    class Semaphore:
        def acquire(self, *, timeout):
            waits.append(timeout)
            return False

    monkeypatch.setattr(pc, "_provider_key", lambda _model: "test")
    monkeypatch.setattr(pc, "_get_semaphore", lambda _provider: Semaphore())
    scope = search_runtime.SearchScope() if scoped else None
    token = search_runtime._scope.set(scope)
    try:
        with pytest.raises(pc.ProviderCapSaturatedError), pc.provider_slot("test"):
            pytest.fail("saturated provider dispatched")
        assert waits == [pc._ACQUIRE_TIMEOUT_SECONDS]
    finally:
        search_runtime._scope.reset(token)


# ---------------------------------------------------------------------------
# Custom providers must be visible to the cap (claude-smart#162)
# ---------------------------------------------------------------------------


def _declared_custom_provider_keys() -> set[str]:
    """Derive every PROVIDER_KEY declared by a custom-provider module.

    Derived rather than hand-listed so a third CLI bridge is covered the day it
    lands -- a list here would be silent about provider N+1, which is the exact
    failure this guard exists to stop.
    """
    import ast
    import pathlib

    providers_dir = pathlib.Path(pc.__file__).resolve().parent / "providers"
    keys: set[str] = set()
    for path in sorted(providers_dir.glob("*.py")):
        source = path.read_text()
        # Only modules that actually register into litellm carry a cap-relevant key.
        if "custom_provider_map" not in source:
            continue
        for node in ast.parse(source).body:
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == "PROVIDER_KEY"
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)
                ):
                    keys.add(node.value.value)
    return keys


def test_declared_custom_providers_are_discoverable() -> None:
    """The scan must find something, or the guard below proves nothing."""
    keys = _declared_custom_provider_keys()
    assert keys, "found no custom-provider PROVIDER_KEY; the guard is measuring nothing"
    assert "claude-code" in keys


def test_registered_custom_provider_is_visible_to_the_cap(monkeypatch) -> None:
    """Every registered custom provider must resolve to a cap key.

    Regression for claude-smart#162: litellm.get_llm_provider RAISES for a
    custom provider even once it is in custom_provider_map, so _provider_key
    returned None and provider_slot yielded UNCAPPED. The providers affected are
    both CLI bridges -- the ones least able to tolerate concurrency.
    """
    import litellm

    for key in sorted(_declared_custom_provider_keys()):
        monkeypatch.setattr(
            litellm,
            "custom_provider_map",
            [{"provider": key, "custom_handler": object()}],
        )
        resolved = pc._provider_key(f"{key}/some-model")
        assert resolved == key, (
            f"registered custom provider {key!r} is invisible to the concurrency "
            f"cap (_provider_key returned {resolved!r}); provider_slot would yield "
            f"uncapped -- claude-smart#162"
        )


def test_unregistered_prefix_stays_uncapped(monkeypatch) -> None:
    """A slash in a model name must not invent a provider.

    The fallback accepts only prefixes present in custom_provider_map, so an
    unknown vendor-style name still returns None and is left uncapped, matching
    the documented 'unknown provider must not be capped or raise' contract.
    """
    import litellm

    monkeypatch.setattr(litellm, "custom_provider_map", [])
    assert pc._provider_key("not-a-provider/whatever") is None


def test_saturated_primary_advances_real_ladder_without_dispatch(monkeypatch):
    """The real semaphore and fallback walk share the dispatch admission contract."""
    from unittest.mock import MagicMock

    import litellm

    from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig

    monkeypatch.setattr(pc, "REFLEXIO_LLM_PROVIDER_MAX_CONCURRENCY", 1)
    monkeypatch.setattr(pc, "_per_provider_cap", {})
    monkeypatch.setattr(pc, "_semaphores", {})
    monkeypatch.setattr(pc, "_ACQUIRE_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(pc, "_provider_key", lambda model: model.split("/")[0])
    calls = []

    def completion(**params):
        calls.append(params["model"])
        response = MagicMock()
        response.choices[0].message.content = "admitted"
        response.choices[0].finish_reason = "stop"
        response.usage = None
        return response

    monkeypatch.setattr(litellm, "completion", completion)
    client = LiteLLMClient(
        LiteLLMConfig(model="primary/a", fallback_models=["backup/b"])
    )
    with pc.provider_slot("primary/a"):
        assert (
            client.generate_chat_response([{"role": "user", "content": "hello"}])
            == "admitted"
        )
        assert calls == ["backup/b"]
        assert pc._get_semaphore("primary")._value == 0
    assert pc._get_semaphore("primary")._value == 1
    assert pc._get_semaphore("backup")._value == 1


def test_saturated_embedding_never_dispatches_and_preserves_held_permit(monkeypatch):
    from unittest.mock import MagicMock

    import litellm

    from reflexio.server.llm import _litellm_embedding as embedding
    from reflexio.server.llm.litellm_client import (
        LiteLLMClient,
        LiteLLMClientError,
        LiteLLMConfig,
    )

    monkeypatch.setattr(pc, "REFLEXIO_LLM_PROVIDER_MAX_CONCURRENCY", 1)
    monkeypatch.setattr(pc, "_per_provider_cap", {})
    monkeypatch.setattr(pc, "_semaphores", {})
    monkeypatch.setattr(pc, "_ACQUIRE_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(pc, "_provider_key", lambda _model: "openai")
    monkeypatch.setattr(embedding, "embedding_provider_mode", lambda _model: "cloud")
    monkeypatch.setattr(embedding, "should_use_embedding_service", lambda _model: False)
    monkeypatch.setattr(embedding, "_truncate_for_embedding", lambda text, _model: text)
    provider = MagicMock(
        return_value=MagicMock(data=[{"index": 0, "embedding": [1.0]}])
    )
    monkeypatch.setattr(litellm, "embedding", provider)
    client = LiteLLMClient(LiteLLMConfig(model="openai/example"))
    with pc.provider_slot("openai/text-embedding-3-small"):
        with pytest.raises(LiteLLMClientError) as failure:
            client.get_embedding("hello", model="openai/text-embedding-3-small")
        assert isinstance(failure.value.__cause__, pc.ProviderCapSaturatedError)
        provider.assert_not_called()
        assert pc._get_semaphore("openai")._value == 0
    assert pc._get_semaphore("openai")._value == 1


def test_live_holders_keep_capacity_until_their_work_finishes(monkeypatch):
    import threading

    monkeypatch.setattr(pc, "REFLEXIO_LLM_PROVIDER_MAX_CONCURRENCY", 2)
    monkeypatch.setattr(pc, "_per_provider_cap", {})
    monkeypatch.setattr(pc, "_semaphores", {})
    monkeypatch.setattr(pc, "_ACQUIRE_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(pc, "_provider_key", lambda _model: "provider")
    release = threading.Event()
    started = [threading.Event(), threading.Event()]
    results = {}
    threads = [
        threading.Thread(target=_hold, args=("model", started[i], release, results, i))
        for i in range(2)
    ]
    for thread in threads:
        thread.start()
    try:
        assert all(event.wait(timeout=2) for event in started)
        assert results == {0: True, 1: True}
        with pytest.raises(pc.ProviderCapSaturatedError), pc.provider_slot("model"):
            pytest.fail("a third caller bypassed the live holders")
        assert pc._get_semaphore("provider")._value == 0
    finally:
        release.set()
        for thread in threads:
            thread.join(timeout=2)
    assert not any(thread.is_alive() for thread in threads)
    assert pc._get_semaphore("provider")._value == 2


@pytest.mark.parametrize("legacy", ["", "openai", "private-provider-list"])
def test_direct_provider_policy_rejects_the_retired_selector(monkeypatch, legacy):
    import runpy

    monkeypatch.setenv("REFLEXIO_LLM_FAIL_CLOSED_PROVIDERS", legacy)
    monkeypatch.setenv("REFLEXIO_LLM_PROVIDER_MAX_CONCURRENCY", "16")
    with pytest.raises(
        ValueError, match="REFLEXIO_LLM_FAIL_CLOSED_PROVIDERS"
    ) as failure:
        runpy.run_path(pc.__file__)
    assert "every resolved provider" in str(failure.value)
    assert "REFLEXIO_LLM_PROVIDER_MAX_CONCURRENCY_OVERRIDES" in str(failure.value)
    assert "private-provider-list" not in str(failure.value)


@pytest.mark.parametrize("fails", [False, True])
def test_provider_timings_separate_admission_execution_and_retirement(
    monkeypatch, fails
):
    import threading
    from types import SimpleNamespace

    now = [0.0]
    measurements = []
    sem = threading.BoundedSemaphore(1)
    monkeypatch.setattr(pc, "time", SimpleNamespace(monotonic=lambda: now[0]))
    monkeypatch.setattr(pc, "_provider_key", lambda _: "test")
    monkeypatch.setattr(pc, "_get_semaphore", lambda _: sem)
    monkeypatch.setattr(
        pc,
        "record_health",
        lambda name, value, **attrs: measurements.append((name, value, attrs)),
    )
    try:
        with pc.provider_slot("test"):
            # A timed-out caller transfers its worker reference before unwinding.
            release_worker = pc.retain_current_provider_slot()
            now[0] = 7.0
            if fails:
                raise RuntimeError("fake timed-out caller")
    except RuntimeError:
        pass
    assert not sem.acquire(blocking=False), (
        "caller exit must not release live worker capacity"
    )
    phases = {attrs["phase"]: value for _, value, attrs in measurements}
    assert phases == {"admission": 0, "execution": 7}
    now[0] = 12.0
    release_worker()
    assert sem.acquire(blocking=False)
    phases = {attrs["phase"]: value for _, value, attrs in measurements}
    assert phases == {"admission": 0, "execution": 7, "retirement": 5}
    assert {name for name, _, _ in measurements} == {"llm.provider.duration"}
    assert all(
        set(attrs) <= {"kind", "unit", "phase", "outcome"}
        for _, _, attrs in measurements
    )


def test_provider_timeout_reports_admission_without_execution(monkeypatch):
    import threading

    measurements = []
    sem = threading.BoundedSemaphore(1)
    sem.acquire()
    monkeypatch.setattr(pc, "_provider_key", lambda _: "test")
    monkeypatch.setattr(pc, "_get_semaphore", lambda _: sem)
    monkeypatch.setattr(
        pc,
        "record_health",
        lambda _name, value, **attrs: measurements.append((value, attrs)),
    )
    with (
        pytest.raises(pc.ProviderCapSaturatedError),
        pc.provider_slot("test", max_wait=0.02),
    ):
        pytest.fail("full owner dispatched work")
    assert len(measurements) == 1
    duration, attrs = measurements[0]
    assert duration >= 0.02
    assert attrs["phase"] == "admission" and attrs["outcome"] == "timeout"


def test_contended_provider_records_queue_before_execution(monkeypatch):
    import threading

    sem = threading.BoundedSemaphore(1)
    sem.acquire()
    queued = threading.Event()
    entered = threading.Event()
    finish = threading.Event()
    measurements = []
    monkeypatch.setattr(pc, "_provider_key", lambda _: "test")
    monkeypatch.setattr(pc, "_get_semaphore", lambda _: sem)
    monkeypatch.setattr(
        pc,
        "record_health",
        lambda _name, value, **attrs: measurements.append((value, attrs)),
    )

    def call():
        queued.set()
        with pc.provider_slot("test"):
            entered.set()
            finish.wait(2)

    thread = threading.Thread(target=call)
    thread.start()
    assert queued.wait(2)
    assert not entered.wait(0.03)
    sem.release()
    try:
        assert entered.wait(2)
        assert len(measurements) == 1
        duration, attrs = measurements[0]
        assert attrs["phase"] == "admission" and attrs["outcome"] == "admitted"
        assert duration >= 0.02
    finally:
        finish.set()
        thread.join(2)
    assert not thread.is_alive()
    assert {attrs["phase"] for _, attrs in measurements} == {
        "admission",
        "execution",
        "retirement",
    }
    assert sem.acquire(blocking=False)
