"""Fair bounded discovery and library restart recovery use the same engine."""

import time

from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.services.durable_learning.scheduler import DurableLearningScheduler


def _unused_context(_: str) -> RequestContext:
    raise AssertionError("This test must not construct a request context")


def test_scheduler_always_starts_and_stops():
    scheduler = DurableLearningScheduler(
        request_context_factory=_unused_context, org_ids_provider=lambda: []
    )
    scheduler.start()
    try:
        assert scheduler.is_running()
    finally:
        scheduler.stop()
    assert not scheduler.is_running()


def test_round_robin_orgs_and_no_claim_when_budget_busy(monkeypatch):
    monkeypatch.setattr(
        "reflexio.server.services.durable_learning.scheduler.worker_count", lambda: 1
    )
    scheduler = DurableLearningScheduler(
        request_context_factory=_unused_context,
        org_ids_provider=lambda: ["b", "a", "c"],
    )
    seen = []
    monkeypatch.setattr(
        scheduler._worker, "start_org", lambda org, _lease: seen.append(org) or True
    )
    for _ in range(5):
        scheduler._run_once()
    assert seen == ["a", "b", "c", "a", "b"]
    monkeypatch.setattr(scheduler._worker, "start_org", lambda *_: False)
    scheduler._run_once()
    assert scheduler._last_org == "b"


def test_library_recovers_persisted_backlog_without_new_publish(tmp_path, monkeypatch):
    from reflexio.lib.reflexio_lib import Reflexio
    from reflexio.models.config_schema import (
        Config,
        ProfileExtractorConfig,
        StorageConfigSQLite,
    )
    from reflexio.server.services.configurator.configurator import DefaultConfigurator
    from reflexio.server.services.durable_learning import local
    from reflexio.server.services.profile.service import ProfileGenerationService

    org = "restart"
    config = DefaultConfigurator(org_id=org, base_dir=str(tmp_path))
    config.set_config(
        Config(
            storage_config=StorageConfigSQLite(db_path=str(tmp_path / "restart.db")),
            window_size=1,
            stride_size=1,
            profile_extractor_config=ProfileExtractorConfig(
                extraction_definition_prompt="Preferences"
            ),
            user_playbook_extractor_config=None,
        )
    )
    with monkeypatch.context() as paused:
        paused.setattr(local, "ensure_local_extraction", lambda _: None)
        first = Reflexio(
            org_id=org, storage_base_dir=str(tmp_path), configurator=config
        )
        response = first.publish_interaction(
            {
                "request_id": "r",
                "user_id": "u",
                "session_id": "s",
                "interaction_data_list": [{"content": "I prefer concise answers"}],
            },
            defer_learning=True,
        )
        assert response.success
        assert first.get_storage().extraction_status("u", "r")["status"] == "pending"
    monkeypatch.setattr(
        ProfileGenerationService, "_should_run_before_extraction", lambda *_: False
    )
    monkeypatch.setenv("REFLEXIO_DURABLE_LEARNING_POLL_SECONDS", ".05")
    # A new facade opens the existing DB and discovers its work at construction.
    second = Reflexio(org_id=org, storage_base_dir=str(tmp_path), configurator=config)
    try:
        deadline = time.monotonic() + 5
        while (
            second.get_storage().extraction_status("u", "r")["status"] != "done"
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)
        assert second.get_storage().extraction_status("u", "r")["status"] == "done"
    finally:
        with local._lock:
            scheduler = local._schedulers.pop(str(tmp_path), None)
            # `_contexts` is a WeakSet of live contexts, not a dict keyed by
            # (org, dir) -- two handles here share that key, which is exactly
            # why the key was removed. Clear it; this is teardown.
            local._contexts.clear()
            local._live.clear()
        if scheduler:
            scheduler.stop()
