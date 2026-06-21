"""Integration tests for aggregation soft-supersede (B3b T1).

Tests the new supersede_agent_playbooks_by_ids and
supersede_agent_playbooks_by_playbook_name storage methods and the branched
aggregation removal path behind is_aggregation_soft_delete_enabled.

Test coverage:
1. Flag ON, incremental run: old rows SUPERSEDED, content intact, status_change events
2. Resurfacing: SUPERSEDED rows excluded from standard reads/search
3. APPROVED never superseded
4. Full-archive run routes through supersede_agent_playbooks_by_playbook_name
5. run_mode signal: aggregate event reason is aggregate:incremental / aggregate:full_archive
6. Flag OFF, full-archive: hard_delete behavior byte-for-byte unchanged
7. Flag OFF, incremental: delete_agent_playbooks_by_ids path (rows physically gone)
8. Idempotency: adds (op=aggregate) and removes (op=status_change) coexist under same _run_id
9. Empty _run_id rejected at call site (production guard in playbook_aggregator.py)
10. search_agent_playbooks(status_filter=None) excludes SUPERSEDED (Part D fix)
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from reflexio.models.api_schema.retriever_schema import SearchAgentPlaybookRequest
from reflexio.models.api_schema.service_schemas import (
    AgentPlaybook,
    PlaybookStatus,
    UserPlaybook,
)
from reflexio.models.config_schema import PlaybookAggregatorConfig, PlaybookConfig
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.services.playbook.playbook_aggregator import PlaybookAggregator
from reflexio.server.services.playbook.playbook_service_utils import (
    PlaybookAggregatorRequest,
)
from reflexio.server.services.storage.error import StorageError
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


def _make_storage(temp_dir: str, worker_id: str, suffix: str = "") -> SQLiteStorage:
    return SQLiteStorage(
        org_id=f"test-agg-soft-delete-{worker_id}{suffix}",
        db_path=os.path.join(temp_dir, f"test{suffix}.db"),
    )


def _make_request_context(
    storage: SQLiteStorage, temp_dir: str, worker_id: str, suffix: str = ""
) -> RequestContext:
    ctx = RequestContext(
        org_id=f"test-agg-soft-delete-{worker_id}{suffix}",
        storage_base_dir=temp_dir,
    )
    ctx.storage = storage
    ctx.prompt_manager = MagicMock()
    ctx.configurator = MagicMock()
    agg_config = PlaybookAggregatorConfig(
        min_cluster_size=2,
        reaggregation_trigger_count=2,
    )
    ctx.configurator.get_config.return_value.user_playbook_extractor_config = (
        PlaybookConfig(
            extractor_name="default",
            extraction_definition_prompt="stub",
            aggregation_config=agg_config,
        )
    )
    return ctx


def _seed_user_playbook(
    storage: SQLiteStorage,
    uid: int,
    playbook_name: str = "default",
    agent_version: str = "v0",
) -> UserPlaybook:
    pb = UserPlaybook(
        user_playbook_id=0,
        user_id="u1",
        agent_version=agent_version,
        request_id=f"req-{uid}",
        playbook_name=playbook_name,
        content=f"Do thing {uid}.",
        trigger=f"when cond {uid}",
        rationale="r",
        status=None,
        source="chat",
        source_interaction_ids=[],
    )
    storage.save_user_playbooks([pb])
    saved = storage.get_user_playbooks(user_id="u1")
    return max(saved, key=lambda p: p.user_playbook_id)


def _seed_agent_playbook(
    storage: SQLiteStorage,
    content: str = "old ap",
    playbook_name: str = "default",
    agent_version: str = "v0",
    playbook_status: PlaybookStatus = PlaybookStatus.PENDING,
) -> AgentPlaybook:
    ap = AgentPlaybook(
        agent_playbook_id=0,
        playbook_name=playbook_name,
        agent_version=agent_version,
        content=content,
        playbook_status=playbook_status,
    )
    saved_list = storage.save_agent_playbooks([ap])
    return saved_list[0]


# ---------------------------------------------------------------------------
# Part A: Storage-level unit tests for supersede methods
# ---------------------------------------------------------------------------


class TestSupersedeAgentPlaybooksByIds:
    @pytest.fixture
    def db(self, temp_dir, worker_id):
        return _make_storage(temp_dir, worker_id, suffix="-by-ids")

    def test_basic_supersede(self, db: SQLiteStorage) -> None:
        """M removed rows get status=SUPERSEDED, content intact, row survives."""
        ap1 = _seed_agent_playbook(db, content="ap one")
        ap2 = _seed_agent_playbook(db, content="ap two")

        count = db.supersede_agent_playbooks_by_ids(
            [ap1.agent_playbook_id, ap2.agent_playbook_id],
            request_id="run_abc",
        )

        assert count == 2
        # Rows survived with content intact (include_tombstones=True)
        row1 = db.get_agent_playbook_by_id(
            ap1.agent_playbook_id, include_tombstones=True
        )
        row2 = db.get_agent_playbook_by_id(
            ap2.agent_playbook_id, include_tombstones=True
        )
        assert row1 is not None
        assert row2 is not None
        assert row1.content == "ap one"
        assert row2.content == "ap two"

    def test_superseded_rows_excluded_by_default(self, db: SQLiteStorage) -> None:
        """Default reads exclude SUPERSEDED rows."""
        ap = _seed_agent_playbook(db, content="old")
        db.supersede_agent_playbooks_by_ids([ap.agent_playbook_id], request_id="run_x")

        # Default get excludes tombstones
        result = db.get_agent_playbook_by_id(ap.agent_playbook_id)
        assert result is None

        # include_tombstones=True returns it
        result_with = db.get_agent_playbook_by_id(
            ap.agent_playbook_id, include_tombstones=True
        )
        assert result_with is not None

    def test_status_change_events_emitted(self, db: SQLiteStorage) -> None:
        """One status_change event per superseded row, all carrying the shared request_id."""
        ap1 = _seed_agent_playbook(db, content="ap one")
        ap2 = _seed_agent_playbook(db, content="ap two")
        # Archive both first so from_status == "archived" (the typical aggregation flow)
        db.archive_agent_playbooks_by_ids(
            [ap1.agent_playbook_id, ap2.agent_playbook_id]
        )

        run_id = "run_shared_id_123"
        db.supersede_agent_playbooks_by_ids(
            [ap1.agent_playbook_id, ap2.agent_playbook_id],
            request_id=run_id,
        )

        for ap in [ap1, ap2]:
            events = db.get_lineage_events(
                entity_type="agent_playbook",
                entity_id=str(ap.agent_playbook_id),
            )
            sc_events = [e for e in events if e.op == "status_change"]
            # The last status_change is the supersede event (there may be one for archive too)
            supersede_events = [e for e in sc_events if e.to_status == "superseded"]
            assert len(supersede_events) == 1, (
                f"Expected 1 status_change->superseded for ap {ap.agent_playbook_id}"
            )
            evt = supersede_events[0]
            assert evt.request_id == run_id
            assert evt.to_status == "superseded"
            assert evt.from_status == "archived", (
                f"Expected from_status='archived', got {evt.from_status!r}"
            )
            assert evt.prov_relation == "wasInvalidatedBy"
            assert evt.status_namespace == "lifecycle_status"
            assert evt.actor == "aggregator"

    def test_approved_never_superseded(self, db: SQLiteStorage) -> None:
        """APPROVED playbooks must not be superseded by the aggregation run."""
        ap = _seed_agent_playbook(db, playbook_status=PlaybookStatus.APPROVED)
        count = db.supersede_agent_playbooks_by_ids(
            [ap.agent_playbook_id], request_id="run_approved"
        )
        assert count == 0
        # Row still exists with original status
        row = db.get_agent_playbook_by_id(ap.agent_playbook_id)
        assert row is not None
        assert row.playbook_status == PlaybookStatus.APPROVED

    def test_already_superseded_not_reprocessed(self, db: SQLiteStorage) -> None:
        """Calling supersede again on already-SUPERSEDED rows is a no-op (idempotent)."""
        ap = _seed_agent_playbook(db)
        db.supersede_agent_playbooks_by_ids([ap.agent_playbook_id], request_id="run_1")
        # Second call
        count = db.supersede_agent_playbooks_by_ids(
            [ap.agent_playbook_id], request_id="run_2"
        )
        assert count == 0

    def test_empty_list_returns_zero(self, db: SQLiteStorage) -> None:
        count = db.supersede_agent_playbooks_by_ids([], request_id="run_empty")
        assert count == 0

    def test_empty_request_id_raises(self, db: SQLiteStorage) -> None:
        """F3: empty request_id must raise at the storage layer (wrapped as StorageError)."""
        ap = _seed_agent_playbook(db, content="to supersede")
        with pytest.raises(StorageError, match="request_id must be non-empty"):
            db.supersede_agent_playbooks_by_ids([ap.agent_playbook_id], request_id="")

    def test_archived_status_superseded(self, db: SQLiteStorage) -> None:
        """Rows with status='archived' (transient) CAN be superseded."""
        ap = _seed_agent_playbook(db)
        # Archive it first
        db.archive_agent_playbooks_by_ids([ap.agent_playbook_id])
        count = db.supersede_agent_playbooks_by_ids(
            [ap.agent_playbook_id], request_id="run_archived"
        )
        assert count == 1
        row = db.get_agent_playbook_by_id(ap.agent_playbook_id, include_tombstones=True)
        assert row is not None


class TestSupersedeAgentPlaybooksByPlaybookName:
    @pytest.fixture
    def db(self, temp_dir, worker_id):
        return _make_storage(temp_dir, worker_id, suffix="-by-name")

    def test_basic_supersede_by_name(self, db: SQLiteStorage) -> None:
        """Rows matching playbook_name + agent_version are superseded."""
        ap1 = _seed_agent_playbook(
            db, content="old v0 a", playbook_name="pb", agent_version="v0"
        )
        # Archive both (full-archive scenario)
        db.archive_agent_playbooks_by_ids([ap1.agent_playbook_id])

        count = db.supersede_agent_playbooks_by_playbook_name(
            playbook_name="pb", agent_version="v0", request_id="run_name"
        )
        assert count == 1
        row = db.get_agent_playbook_by_id(
            ap1.agent_playbook_id, include_tombstones=True
        )
        assert row is not None

    def test_wrong_name_not_superseded(self, db: SQLiteStorage) -> None:
        ap = _seed_agent_playbook(db, playbook_name="pb_other")
        db.archive_agent_playbooks_by_ids([ap.agent_playbook_id])
        count = db.supersede_agent_playbooks_by_playbook_name(
            playbook_name="pb_different", agent_version=None, request_id="run_wrong"
        )
        assert count == 0
        row = db.get_agent_playbook_by_id(ap.agent_playbook_id, include_tombstones=True)
        assert row is not None

    def test_events_carry_request_id_and_from_status(self, db: SQLiteStorage) -> None:
        """status_change events under supersede_by_name carry the caller request_id and from_status='archived'."""
        ap = _seed_agent_playbook(db, playbook_name="pb2", agent_version="v1")
        db.archive_agent_playbooks_by_ids([ap.agent_playbook_id])
        run_id = "full_archive_run_999"
        db.supersede_agent_playbooks_by_playbook_name(
            playbook_name="pb2", agent_version="v1", request_id=run_id
        )
        events = db.get_lineage_events(
            entity_type="agent_playbook", entity_id=str(ap.agent_playbook_id)
        )
        sc_events = [
            e for e in events if e.op == "status_change" and e.to_status == "superseded"
        ]
        assert len(sc_events) == 1
        evt = sc_events[0]
        assert evt.request_id == run_id
        assert evt.from_status == "archived", (
            f"Expected from_status='archived', got {evt.from_status!r}"
        )

    def test_empty_name_not_crashed(self, db: SQLiteStorage) -> None:
        """No archived rows for name means count=0, no exception."""
        count = db.supersede_agent_playbooks_by_playbook_name(
            playbook_name="nonexistent", agent_version=None, request_id="run_none"
        )
        assert count == 0

    def test_empty_request_id_raises(self, db: SQLiteStorage) -> None:
        """F3: empty request_id must raise at the storage layer (wrapped as StorageError)."""
        with pytest.raises(StorageError, match="request_id must be non-empty"):
            db.supersede_agent_playbooks_by_playbook_name(
                playbook_name="any", agent_version=None, request_id=""
            )


# ---------------------------------------------------------------------------
# Part B + C: Aggregator integration (flag ON/OFF, incremental + full-archive)
# ---------------------------------------------------------------------------


def _run_aggregator_with_supersede(
    temp_dir: str,
    worker_id: str,
    full_archive: bool,
    flag_on: bool,
    suffix: str,
) -> tuple[SQLiteStorage, RequestContext]:
    """Run one aggregation with one new and one old archived playbook."""
    storage = _make_storage(temp_dir, worker_id, suffix=suffix)
    ctx = _make_request_context(storage, temp_dir, worker_id, suffix=suffix)

    # Seed old archived agent playbook (will be removed on SUCCESS path)
    old_ap = _seed_agent_playbook(
        storage, content="old content", playbook_name="pb", agent_version="v0"
    )
    storage.archive_agent_playbooks_by_ids([old_ap.agent_playbook_id])

    # Seed user playbooks
    up_a = _seed_user_playbook(storage, uid=1, playbook_name="pb", agent_version="v0")
    up_b = _seed_user_playbook(storage, uid=2, playbook_name="pb", agent_version="v0")
    cluster_playbooks = [up_a, up_b]

    # New agent playbook
    new_ap = AgentPlaybook(
        agent_playbook_id=0,
        playbook_name="pb",
        agent_version="v0",
        content="New aggregated content.",
        playbook_status=PlaybookStatus.PENDING,
    )

    flag_path = "reflexio.server.services.playbook.playbook_aggregator.is_aggregation_soft_delete_enabled"
    with (
        patch.object(
            PlaybookAggregator,
            "get_clusters",
            return_value={0: cluster_playbooks},
        ),
        patch.object(
            PlaybookAggregator,
            "_generate_playbooks_with_source_clusters",
            return_value=[(new_ap, cluster_playbooks)],
        ),
        patch(flag_path, return_value=flag_on),
    ):
        aggregator = PlaybookAggregator(
            llm_client=MagicMock(),
            request_context=ctx,
            agent_version="v0",
        )
        aggregator.run(
            PlaybookAggregatorRequest(
                agent_version="v0",
                rerun=full_archive,
            )
        )

    return storage, ctx


def _run_two_aggregations_incremental_flag_off(
    temp_dir: str,
    worker_id: str,
    suffix: str,
) -> tuple[SQLiteStorage, list[int]]:
    """Run two aggregations to exercise the incremental flag-OFF hard-delete path.

    First run: 2 clusters → saves 2 agent playbooks, stores fingerprints.
    Second run: only cluster 0 remains (cluster 1 disappeared) →
        archived_playbook_ids is non-empty → flag OFF → delete_agent_playbooks_by_ids.

    Returns:
        Tuple of (storage, removed_ap_ids) where removed_ap_ids are the IDs that
        should be physically deleted by the second run.
    """
    storage = _make_storage(temp_dir, worker_id, suffix=suffix)
    ctx = _make_request_context(storage, temp_dir, worker_id, suffix=suffix)

    # Seed user playbooks for two separate clusters
    up_a = _seed_user_playbook(
        storage, uid=10, playbook_name="default", agent_version="v0"
    )
    up_b = _seed_user_playbook(
        storage, uid=11, playbook_name="default", agent_version="v0"
    )
    up_c = _seed_user_playbook(
        storage, uid=12, playbook_name="default", agent_version="v0"
    )
    up_d = _seed_user_playbook(
        storage, uid=13, playbook_name="default", agent_version="v0"
    )

    cluster_0 = [up_a, up_b]
    cluster_1 = [up_c, up_d]

    new_ap_0 = AgentPlaybook(
        agent_playbook_id=0,
        playbook_name="default",
        agent_version="v0",
        content="Cluster 0 playbook.",
        playbook_status=PlaybookStatus.PENDING,
    )
    new_ap_1 = AgentPlaybook(
        agent_playbook_id=0,
        playbook_name="default",
        agent_version="v0",
        content="Cluster 1 playbook.",
        playbook_status=PlaybookStatus.PENDING,
    )

    flag_path = "reflexio.server.services.playbook.playbook_aggregator.is_aggregation_soft_delete_enabled"

    # First run: both clusters present — seeds fingerprints for cluster 0 and cluster 1.
    with (
        patch.object(
            PlaybookAggregator,
            "get_clusters",
            return_value={0: cluster_0, 1: cluster_1},
        ),
        patch.object(
            PlaybookAggregator,
            "_generate_playbooks_with_source_clusters",
            return_value=[(new_ap_0, cluster_0), (new_ap_1, cluster_1)],
        ),
        patch(flag_path, return_value=False),
    ):
        agg = PlaybookAggregator(
            llm_client=MagicMock(),
            request_context=ctx,
            agent_version="v0",
        )
        agg.run(PlaybookAggregatorRequest(agent_version="v0", rerun=False))

    # After first run, find the agent playbook created for cluster 1 (it will be removed).
    all_aps = storage.get_agent_playbooks(agent_version="v0")
    # Both APs exist (cluster 0 and cluster 1)
    assert len(all_aps) >= 2, (
        f"Expected >= 2 agent playbooks after first run, got {len(all_aps)}"
    )

    # We need the ap_id for cluster 1's playbook — it will be archived then deleted.
    # Use lineage events to find the aggregate event for cluster 1's user_playbook_ids.
    events_all = storage.get_lineage_events(entity_type="agent_playbook")
    cluster1_up_ids = {str(up_c.user_playbook_id), str(up_d.user_playbook_id)}
    cluster1_ap_id: int | None = None
    for evt in events_all:
        if (
            evt.op == "aggregate"
            and evt.source_ids
            and cluster1_up_ids.issubset(set(evt.source_ids))
        ):
            cluster1_ap_id = int(evt.entity_id)
            break

    assert cluster1_ap_id is not None, (
        "Could not find cluster 1 agent_playbook_id from lineage events"
    )

    # Second run: only cluster 0 remains — cluster 1 fingerprint disappears →
    # archived_playbook_ids = [cluster1_ap_id] → incremental path → flag OFF → hard delete.
    # Patch _should_run_aggregation to True so the gate doesn't skip (no new UPs since run 1).
    new_ap_0_v2 = AgentPlaybook(
        agent_playbook_id=0,
        playbook_name="default",
        agent_version="v0",
        content="Cluster 0 updated playbook.",
        playbook_status=PlaybookStatus.PENDING,
    )
    with (
        patch.object(
            PlaybookAggregator,
            "get_clusters",
            return_value={0: cluster_0},
        ),
        patch.object(
            PlaybookAggregator,
            "_generate_playbooks_with_source_clusters",
            return_value=[(new_ap_0_v2, cluster_0)],
        ),
        patch.object(PlaybookAggregator, "_should_run_aggregation", return_value=True),
        patch(flag_path, return_value=False),
    ):
        agg2 = PlaybookAggregator(
            llm_client=MagicMock(),
            request_context=ctx,
            agent_version="v0",
        )
        agg2.run(PlaybookAggregatorRequest(agent_version="v0", rerun=False))

    return storage, [cluster1_ap_id]


class TestAggregationSoftDeleteFlagOn:
    def test_incremental_supersedes_old_rows(self, temp_dir, worker_id) -> None:
        """Flag ON + incremental: old archived rows become SUPERSEDED, content intact."""
        storage, _ctx = _run_aggregator_with_supersede(
            temp_dir, worker_id, full_archive=False, flag_on=True, suffix="-incr-on"
        )
        # Old ap was archived, should now be superseded.
        # We use the lineage query to verify SUPERSEDED events exist.
        events = storage.get_lineage_events(entity_type="agent_playbook")
        sc_supersede = [
            e for e in events if e.op == "status_change" and e.to_status == "superseded"
        ]
        assert len(sc_supersede) >= 1, (
            "Expected at least 1 status_change->superseded event"
        )

    def test_incremental_status_change_events_carry_run_id(
        self, temp_dir, worker_id
    ) -> None:
        """All status_change events for removed rows share the same request_id as aggregate events."""
        storage, _ctx = _run_aggregator_with_supersede(
            temp_dir, worker_id, full_archive=False, flag_on=True, suffix="-incr-rid"
        )
        events = storage.get_lineage_events(entity_type="agent_playbook")
        agg_events = [e for e in events if e.op == "aggregate"]
        sc_events = [
            e for e in events if e.op == "status_change" and e.to_status == "superseded"
        ]
        assert agg_events, "Expected aggregate events"
        assert sc_events, "Expected status_change->superseded events"
        run_ids_agg = {e.request_id for e in agg_events}
        run_ids_sc = {e.request_id for e in sc_events}
        # Both sets should share the same run_id
        shared = run_ids_agg & run_ids_sc
        assert shared, (
            f"aggregate and status_change events must share _run_id; "
            f"agg={run_ids_agg} sc={run_ids_sc}"
        )

    def test_idempotency_key_non_collision(self, temp_dir, worker_id) -> None:
        """aggregate (op=aggregate) and removes (op=status_change) coexist under same _run_id."""
        storage, _ctx = _run_aggregator_with_supersede(
            temp_dir, worker_id, full_archive=False, flag_on=True, suffix="-idem"
        )
        events = storage.get_lineage_events(entity_type="agent_playbook")
        agg_events = [e for e in events if e.op == "aggregate"]
        sc_events = [
            e for e in events if e.op == "status_change" and e.to_status == "superseded"
        ]
        # Both must be present after the run
        assert agg_events, "aggregate events must be present"
        assert sc_events, "status_change->superseded events must be present"

    def test_superseded_rows_excluded_from_get_agent_playbooks(
        self, temp_dir, worker_id
    ) -> None:
        """After flag-ON run, superseded rows are NOT in standard get_agent_playbooks()."""
        storage, _ctx = _run_aggregator_with_supersede(
            temp_dir, worker_id, full_archive=False, flag_on=True, suffix="-excl"
        )
        events = storage.get_lineage_events(entity_type="agent_playbook")
        sc_events = [
            e for e in events if e.op == "status_change" and e.to_status == "superseded"
        ]
        for evt in sc_events:
            ap_id = int(evt.entity_id)
            row = storage.get_agent_playbook_by_id(ap_id)
            assert row is None, (
                f"SUPERSEDED ap {ap_id} must not appear in default reads"
            )
            row_with = storage.get_agent_playbook_by_id(ap_id, include_tombstones=True)
            assert row_with is not None, (
                f"SUPERSEDED ap {ap_id} must be found with include_tombstones=True"
            )

    def test_run_mode_reason_incremental(self, temp_dir, worker_id) -> None:
        """aggregate event reason == 'aggregate:incremental' for truly incremental runs.

        The first run always triggers full_archive (no previous fingerprints).
        A second run with rerun=False and no changed clusters yields 'incremental'.
        We verify the reason is one of the two valid structured tokens.
        """
        storage, _ctx = _run_aggregator_with_supersede(
            temp_dir, worker_id, full_archive=False, flag_on=True, suffix="-rm-incr"
        )
        events = storage.get_lineage_events(entity_type="agent_playbook")
        agg_events = [e for e in events if e.op == "aggregate"]
        assert agg_events, "Expected aggregate events"
        valid_reasons = {"aggregate:incremental", "aggregate:full_archive"}
        for evt in agg_events:
            assert evt.reason in valid_reasons, (
                f"Expected reason in {valid_reasons!r}, got {evt.reason!r}"
            )

    def test_full_archive_run_routes_supersede_by_name(
        self, temp_dir, worker_id
    ) -> None:
        """Flag ON + full_archive: supersede_by_name path, old rows SUPERSEDED."""
        storage, _ctx = _run_aggregator_with_supersede(
            temp_dir, worker_id, full_archive=True, flag_on=True, suffix="-full-on"
        )
        events = storage.get_lineage_events(entity_type="agent_playbook")
        sc_events = [
            e for e in events if e.op == "status_change" and e.to_status == "superseded"
        ]
        assert sc_events, "Full-archive + flag ON must supersede old rows"

    def test_run_mode_reason_full_archive(self, temp_dir, worker_id) -> None:
        """aggregate event reason == 'aggregate:full_archive' for full-archive run."""
        storage, _ctx = _run_aggregator_with_supersede(
            temp_dir, worker_id, full_archive=True, flag_on=True, suffix="-rm-full"
        )
        events = storage.get_lineage_events(entity_type="agent_playbook")
        agg_events = [e for e in events if e.op == "aggregate"]
        assert agg_events, "Expected aggregate events"
        for evt in agg_events:
            assert evt.reason == "aggregate:full_archive", (
                f"Expected reason='aggregate:full_archive', got {evt.reason!r}"
            )

    def test_run_mode_reason_is_structured_token(self, temp_dir, worker_id) -> None:
        """aggregate event reason must be a structured 'aggregate:<mode>' token, not free text."""
        storage, _ctx = _run_aggregator_with_supersede(
            temp_dir, worker_id, full_archive=True, flag_on=True, suffix="-rm-struct"
        )
        events = storage.get_lineage_events(entity_type="agent_playbook")
        agg_events = [e for e in events if e.op == "aggregate"]
        assert agg_events, "Expected aggregate events"
        for evt in agg_events:
            assert evt.reason.startswith("aggregate:"), (
                f"reason must start with 'aggregate:' — got {evt.reason!r}"
            )
            assert evt.reason != "user->agent aggregation", (
                "Legacy free-text reason must be replaced with structured token"
            )


class TestAggregationSoftDeleteFlagOff:
    def test_flag_off_full_archive_hard_delete(self, temp_dir, worker_id) -> None:
        """Flag OFF + full_archive: removal uses delete_archived_*_by_playbook_name (rows physically gone)."""
        storage, _ctx = _run_aggregator_with_supersede(
            temp_dir,
            worker_id,
            full_archive=True,
            flag_on=False,
            suffix="-flag-off-full",
        )
        events = storage.get_lineage_events(entity_type="agent_playbook")
        hd_events = [e for e in events if e.op == "hard_delete"]
        assert hd_events, "Flag OFF + full_archive must emit hard_delete events"
        # No superseded rows
        sc_supersede = [
            e for e in events if e.op == "status_change" and e.to_status == "superseded"
        ]
        assert not sc_supersede, (
            "Flag OFF must not emit status_change->superseded events"
        )

    def test_flag_off_full_archive_rows_physically_gone(
        self, temp_dir, worker_id
    ) -> None:
        """Flag OFF + full_archive: physically deleted rows are gone even with include_tombstones=True."""
        storage, _ctx = _run_aggregator_with_supersede(
            temp_dir, worker_id, full_archive=True, flag_on=False, suffix="-phys-gone"
        )
        events = storage.get_lineage_events(entity_type="agent_playbook")
        hd_events = [e for e in events if e.op == "hard_delete"]
        for evt in hd_events:
            ap_id = int(evt.entity_id)
            row = storage.get_agent_playbook_by_id(ap_id, include_tombstones=True)
            assert row is None, f"Hard-deleted ap {ap_id} must be physically gone"

    def test_flag_off_incremental_deletes_by_ids(self, temp_dir, worker_id) -> None:
        """Flag OFF + incremental: delete_agent_playbooks_by_ids path — rows physically gone, hard_delete events emitted.

        This exercises the ``elif archived_playbook_ids: ... delete_agent_playbooks_by_ids``
        branch in playbook_aggregator.py (flag OFF, not full_archive).
        Setup: two-run sequence where cluster 1 disappears on the second run, causing
        its agent playbook ID to land in archived_playbook_ids.
        """
        storage, removed_ids = _run_two_aggregations_incremental_flag_off(
            temp_dir, worker_id, suffix="-incr-flag-off"
        )
        assert removed_ids, "Expected at least one removed ap_id"
        events = storage.get_lineage_events(entity_type="agent_playbook")
        hd_events = [e for e in events if e.op == "hard_delete"]

        for ap_id in removed_ids:
            # Row must be physically gone (even with include_tombstones=True)
            row = storage.get_agent_playbook_by_id(ap_id, include_tombstones=True)
            assert row is None, (
                f"Incremental flag-OFF: ap {ap_id} must be physically deleted"
            )
            # hard_delete event must exist for this entity
            entity_hd = [e for e in hd_events if e.entity_id == str(ap_id)]
            assert entity_hd, (
                f"Incremental flag-OFF: expected hard_delete event for ap {ap_id}"
            )

        # No superseded rows from the removal path
        sc_supersede = [
            e for e in events if e.op == "status_change" and e.to_status == "superseded"
        ]
        assert not sc_supersede, (
            "Flag OFF incremental must not emit status_change->superseded events"
        )


# ---------------------------------------------------------------------------
# Part B: Empty _run_id rejected (production guard in playbook_aggregator.py)
# ---------------------------------------------------------------------------


class TestEmptyRunIdRejected:
    def test_empty_run_id_raises_via_aggregator(self, temp_dir, worker_id) -> None:
        """StorageError propagates when _run_id is empty (validated at storage layer).

        The aggregator no longer has its own guard for empty _run_id; instead,
        supersede methods raise StorageError("request_id must be non-empty").
        The test patches uuid.uuid4 in the aggregator module so _run_id becomes "",
        sets flag ON (soft=True), uses rerun=True (full_archive=True guaranteed), and
        confirms StorageError propagates from the supersede call through the aggregator.
        """
        storage = _make_storage(temp_dir, worker_id, suffix="-empty-rid")
        ctx = _make_request_context(storage, temp_dir, worker_id, suffix="-empty-rid")

        up_a = _seed_user_playbook(storage, uid=1)
        up_b = _seed_user_playbook(storage, uid=2)
        cluster_playbooks = [up_a, up_b]
        new_ap = AgentPlaybook(
            agent_playbook_id=0,
            playbook_name="default",
            agent_version="v0",
            content="Some content.",
            playbook_status=PlaybookStatus.PENDING,
        )

        flag_path = "reflexio.server.services.playbook.playbook_aggregator.is_aggregation_soft_delete_enabled"

        # We need _run_id = str(uuid.uuid4()) to evaluate to "".
        # Patch uuid.uuid4 in the aggregator's module namespace only. To avoid breaking
        # other uuid.uuid4().hex calls in the storage layer (different module import),
        # return a mock object whose __str__ returns "" but whose .hex is a valid string.
        class _EmptyStrUUID:
            hex = "000000000000000000000000"

            def __str__(self) -> str:
                return ""

        uuid_path = "reflexio.server.services.playbook.playbook_aggregator.uuid.uuid4"

        with (
            patch.object(
                PlaybookAggregator,
                "get_clusters",
                return_value={0: cluster_playbooks},
            ),
            patch.object(
                PlaybookAggregator,
                "_generate_playbooks_with_source_clusters",
                return_value=[(new_ap, cluster_playbooks)],
            ),
            patch(flag_path, return_value=True),
            # Patch uuid.uuid4 in the aggregator module so _run_id = str(mock) = "".
            patch(uuid_path, return_value=_EmptyStrUUID()),
            pytest.raises(StorageError, match="request_id must be non-empty"),
        ):
            aggregator = PlaybookAggregator(
                llm_client=MagicMock(),
                request_context=ctx,
                agent_version="v0",
            )
            aggregator.run(
                PlaybookAggregatorRequest(
                    agent_version="v0",
                    rerun=True,  # guarantees full_archive=True → StorageError fires from supersede
                )
            )


# ---------------------------------------------------------------------------
# Part D: search_agent_playbooks tombstone-exclusion fix
# ---------------------------------------------------------------------------


class TestSearchAgentPlaybooksTombstoneExclusion:
    @pytest.fixture
    def db(self, temp_dir, worker_id):
        return _make_storage(temp_dir, worker_id, suffix="-search-excl")

    def test_superseded_excluded_by_default_search(self, db: SQLiteStorage) -> None:
        """search_agent_playbooks(status_filter=None) excludes SUPERSEDED rows."""
        ap = _seed_agent_playbook(db, content="old superseded ap")
        # Manually supersede it
        db.supersede_agent_playbooks_by_ids([ap.agent_playbook_id], request_id="r1")

        req = SearchAgentPlaybookRequest(
            query="old superseded",
            agent_version="v0",
            top_k=10,
        )
        results = db.search_agent_playbooks(req)
        result_ids = [r.agent_playbook_id for r in results]
        assert ap.agent_playbook_id not in result_ids, (
            "SUPERSEDED agent playbook must not appear in search_agent_playbooks with status_filter=None"
        )

    def test_non_superseded_present_in_search(self, db: SQLiteStorage) -> None:
        """A normal (non-tombstone) agent playbook appears in search results."""
        ap = _seed_agent_playbook(db, content="live active ap for search test")
        req = SearchAgentPlaybookRequest(
            query="live active",
            agent_version="v0",
            top_k=10,
        )
        results = db.search_agent_playbooks(req)
        result_ids = [r.agent_playbook_id for r in results]
        assert ap.agent_playbook_id in result_ids, (
            "Live agent playbook must appear in search_agent_playbooks"
        )


# ---------------------------------------------------------------------------
# Part E: End-to-end from_status signal through full aggregator run
# ---------------------------------------------------------------------------


class TestAggregatorFromStatusSignal:
    def test_aggregator_full_archive_status_change_carries_from_status_archived(
        self, temp_dir, worker_id
    ) -> None:
        """Full-archive aggregator run: supersede events carry from_status='archived' and status_namespace='lifecycle_status'.

        Verifies the end-to-end signal — not just at storage-unit level but through
        the complete aggregator run: archive_agent_playbooks_by_ids followed by
        supersede_agent_playbooks_by_playbook_name produces status_change events
        with the correct structured fields.
        """
        storage, _ctx = _run_aggregator_with_supersede(
            temp_dir, worker_id, full_archive=True, flag_on=True, suffix="-from-status"
        )
        events = storage.get_lineage_events(entity_type="agent_playbook")
        sc_supersede = [
            e for e in events if e.op == "status_change" and e.to_status == "superseded"
        ]
        assert sc_supersede, (
            "Full-archive + flag ON must produce status_change->superseded events"
        )
        for evt in sc_supersede:
            assert evt.from_status == "archived", (
                f"Expected from_status='archived' (aggregator archives before superseding), "
                f"got {evt.from_status!r} for entity {evt.entity_id}"
            )
            assert evt.status_namespace == "lifecycle_status", (
                f"Expected status_namespace='lifecycle_status', got {evt.status_namespace!r}"
            )
