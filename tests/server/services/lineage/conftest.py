"""Isolate process-owned maintenance composition between scheduler tests."""

import pytest

from reflexio.server.services.lineage import gc_scheduler


@pytest.fixture(autouse=True)
def isolated_maintenance_composition(monkeypatch):
    # create_app() in other suites registers backfill/governance sweeps. Those
    # registrations must not make a no-work scheduler fixture start a daemon.
    for name in (
        "_per_org_sweep_hooks",
        "_global_sweep_hooks",
        "_always_global_sweep_hooks",
    ):
        monkeypatch.setattr(gc_scheduler, name, [])
    for name in (
        "_maintenance_schedule_hook",
        "_org_id_provider_hook",
        "_project_id_provider_hook",
    ):
        monkeypatch.setattr(gc_scheduler, name, None)
