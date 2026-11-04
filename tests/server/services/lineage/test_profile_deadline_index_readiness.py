"""Deferred explicit-retention index readiness does not stop ordinary maintenance."""

from reflexio.server.services.storage.error import ReadinessUnavailableError

from .test_profile_deadline_scheduler import (
    _PROJECT,
    MemoryRetention,
    clock,  # noqa: F401 -- imported pytest fixture
    isolate,  # noqa: F401 -- imported autouse pytest fixture
    scheduler,
)


def test_delayed_index_is_scope_local_and_preserves_ordinary_maintenance(clock):  # noqa: F811
    class DelayedIndex(MemoryRetention):
        def next_profile_deadline(self, *, user_id=None):
            if _PROJECT.get() == "blocked":
                raise ReadinessUnavailableError(
                    "profile_explicit_retention_index_unavailable"
                )
            return super().next_profile_deadline(user_id=user_id)

    storage = DelayedIndex({"healthy": 1005})
    owner, maintenance = scheduler(storage, projects=("blocked", "healthy"))
    visited = []

    def record_project(org_id, storage, cfg, project_id=None):
        del org_id, storage, cfg
        visited.append(project_id)

    owner._sweep_project_data = record_project
    assert owner._run_once() == 5
    assert owner._tick_had_failure
    assert visited == ["blocked", "healthy"]
    assert "org" in maintenance and "global" in maintenance
    before = list(maintenance)
    clock.advance(5)
    owner._run_once()
    assert any(call[0] == "purge" and call[1] == "healthy" for call in storage.calls)
    assert maintenance == before
    assert visited == ["blocked", "healthy"]
