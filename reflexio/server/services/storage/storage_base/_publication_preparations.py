# ruff: noqa: S608 -- Only fixed preparation/run table and column identifiers.
"""Owned original preparation and its interrupted compute entry, across SQL adapters."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from typing import TYPE_CHECKING, Any, cast

from reflexio.server.services.extraction.publication_preparation import (
    PendingPreparationAttempt,
    SavedPublicationPreparation,
    preparation_claim_digest,
)
from reflexio.server.services.extraction.publication_receipts import require_output_id
from reflexio.server.work_scope import current_project_id

from ._agent_run import AgentRunRecord
from ._extraction_stream import LeaseLostError, StreamSQL, decoded

if TYPE_CHECKING:
    from . import BaseStorage


class PublicationPreparationStore:
    @contextmanager
    def _owned_preparation_writer(
        self, run: AgentRunRecord
    ) -> Iterator[tuple[BaseStorage, AgentRunRecord, StreamSQL]]:
        from reflexio.server.services.durable_learning.user_lease import (
            owned_agent_run_write,
        )

        from . import BaseStorage

        storage = cast(BaseStorage, self)
        retained = deepcopy(run)
        if not isinstance(retained.committed_output, dict):
            raise ValueError("Preparation requires retained output")
        with owned_agent_run_write(storage, retained), storage._stream_sql() as db:
            storage._require_current_agent_output(
                db, retained, retained.committed_output
            )
            yield storage, retained, db

    @staticmethod
    def _write_preparation_row(
        storage: BaseStorage,
        run: AgentRunRecord,
        db: StreamSQL,
        rows: list[dict[str, Any]],
        preparation: dict[str, Any] | None,
        active: PendingPreparationAttempt | None,
    ) -> None:
        values = (
            json.dumps(preparation, allow_nan=False)
            if preparation is not None
            else None,
            json.dumps(active.payload(), allow_nan=False)
            if active is not None
            else None,
        )
        if rows:
            db.query(
                f"UPDATE {db.table('_agent_output_preparations')} AS r SET preparation=?,active_attempt=? "
                f"WHERE r.org_id=? AND r.run_id=? AND {db.project_filter()}",
                (*values, storage.org_id, run.id, current_project_id() or ""),
            )
        else:
            db.query(
                f"INSERT INTO {db.table('_agent_output_preparations')} (preparation,active_attempt,org_id,run_id,project_id) VALUES (?,?,?,?,?)",
                (*values, storage.org_id, run.id, current_project_id() or ""),
            )

    @staticmethod
    def _check_previous_preparation(
        storage: BaseStorage, run: AgentRunRecord, db: StreamSQL, cached: Any
    ) -> dict[str, Any] | None:
        if not isinstance(cached, dict):
            raise ValueError("Saved publication preparation invalid")
        previous_id = require_output_id(cached.get("output_id"))
        output = run.committed_output
        publication = output.get("publication") if isinstance(output, dict) else None
        output_id = require_output_id(
            publication.get("id") if isinstance(publication, dict) else None
        )
        if previous_id != output_id:
            if storage._get_output_publication_receipt(db, run.id, previous_id) is None:
                raise ValueError("Unpublished preparation cannot be replaced")
            return None
        SavedPublicationPreparation.from_payload(run, cached)
        return cached

    def agent_output_preparation(
        self,
        run: AgentRunRecord,
        *,
        save: dict[str, Any] | None = None,
        attempt_id: str | None = None,
    ) -> dict[str, Any] | None:
        requested = deepcopy(save)
        with self._owned_preparation_writer(run) as (storage, retained, db):
            rows = db.query(
                f"SELECT r.preparation,r.active_attempt FROM {db.table('_agent_output_preparations')} r "
                f"WHERE r.org_id=? AND r.run_id=? AND {db.project_filter()}",
                (storage.org_id, retained.id, current_project_id() or ""),
            )
            cached = decoded(rows[0]["preparation"]) if rows else None
            pending = decoded(rows[0]["active_attempt"]) if rows else None
            entry = (
                PendingPreparationAttempt.from_payload(retained, pending)
                if pending is not None
                else None
            )
            if cached is not None:
                cached = self._check_previous_preparation(storage, retained, db, cached)
            if requested is None:
                return deepcopy(cached)
            completed = SavedPublicationPreparation.from_payload(retained, requested)
            if entry is not None:
                if (
                    entry.identity != attempt_id
                    or entry.claim_digest != preparation_claim_digest(retained)
                ):
                    raise LeaseLostError(
                        "Preparation completion lost its admitted entry"
                    )
            elif attempt_id is not None:
                raise LeaseLostError("Preparation entry disappeared")
            if cached is not None:
                if cached != requested:
                    raise ValueError("Original publication preparation is immutable")
                return deepcopy(cached)
            # The store owns raw completion admission under strict input locks,
            # then commits completion and clearing its compute entry together.
            # A refusal rolls back the writer; the live owner retains its result.
            deadline = storage.validate_agent_preparation_inputs(
                retained,
                completed.history_inputs.profiles,
                completed.history_inputs.playbooks,
            )
            self._write_preparation_row(storage, retained, db, rows, requested, None)
            result = deepcopy(requested)
            if deadline is not None and db.now() >= deadline:
                from reflexio.server.llm.input_budget import (
                    InputBudgetConfigurationError,
                )

                raise InputBudgetConfigurationError(
                    "Preparation input expired during completion"
                )
            return result

    def begin_agent_preparation_attempt(
        self, run: AgentRunRecord
    ) -> PendingPreparationAttempt:
        from reflexio.server.services.durable_learning.user_lease import (
            _require_user_work_owner,
        )

        with self._owned_preparation_writer(run) as (storage, retained, db):
            storage.lock_extraction_stream(_require_user_work_owner(storage).user_id)
            rows = db.query(
                f"SELECT r.preparation,r.active_attempt FROM {db.table('_agent_output_preparations')} r "
                f"WHERE r.org_id=? AND r.run_id=? AND {db.project_filter()}",
                (storage.org_id, retained.id, current_project_id() or ""),
            )
            if rows and rows[0]["active_attempt"] is not None:
                PendingPreparationAttempt.from_payload(
                    retained, decoded(rows[0]["active_attempt"])
                )
                raise LeaseLostError("Preparation already has an active entry")
            if (
                rows
                and self._check_previous_preparation(
                    storage, retained, db, decoded(rows[0]["preparation"])
                )
                is not None
            ):
                raise ValueError("Completed preparation cannot compute again")
            counters = db.query(
                f"SELECT r.finalization_attempts FROM {db.table('_agent_runs')} r WHERE r.org_id=? AND r.id=? AND {db.project_filter()}",
                (storage.org_id, retained.id, current_project_id() or ""),
            )
            if counters[0]["finalization_attempts"] != retained.finalization_attempts:
                raise LeaseLostError("Preparation failure budget changed")
            entry = PendingPreparationAttempt.begin(retained)
            self._write_preparation_row(storage, retained, db, rows, None, entry)
            return entry

    def pending_agent_preparation_attempt(
        self, run: AgentRunRecord
    ) -> PendingPreparationAttempt | None:
        with self._owned_preparation_writer(run) as (storage, retained, db):
            rows = db.query(
                f"SELECT r.active_attempt FROM {db.table('_agent_output_preparations')} r "
                f"WHERE r.org_id=? AND r.run_id=? AND {db.project_filter()}",
                (storage.org_id, retained.id, current_project_id() or ""),
            )
            pending = decoded(rows[0]["active_attempt"]) if rows else None
            return (
                PendingPreparationAttempt.from_payload(retained, pending)
                if pending is not None
                else None
            )

    def settle_agent_preparation_failure(
        self, run: AgentRunRecord, identity: str
    ) -> None:
        """Remove pending data only after its failure was durably charged in this TX."""
        from . import BaseStorage

        storage = cast(BaseStorage, self)
        if not storage.in_commit_scope:
            raise ValueError("Preparation settlement requires the failure writer")
        with self._owned_preparation_writer(run) as (storage, retained, db):
            rows = db.query(
                f"SELECT r.active_attempt FROM {db.table('_agent_output_preparations')} r "
                f"WHERE r.org_id=? AND r.run_id=? AND {db.project_filter()}",
                (storage.org_id, retained.id, current_project_id() or ""),
            )
            pending = decoded(rows[0]["active_attempt"]) if rows else None
            if pending is None:
                raise LeaseLostError("Preparation entry disappeared before settlement")
            entry = PendingPreparationAttempt.from_payload(retained, pending)
            counters = db.query(
                f"SELECT r.finalization_attempts FROM {db.table('_agent_runs')} r WHERE r.org_id=? AND r.id=? AND {db.project_filter()}",
                (storage.org_id, retained.id, current_project_id() or ""),
            )
            if (
                entry.identity != identity
                or counters[0]["finalization_attempts"] != entry.failures_at_entry + 1
            ):
                raise LeaseLostError("Preparation failure charge is not current")
            db.query(
                f"DELETE FROM {db.table('_agent_output_preparations')} AS r WHERE r.org_id=? AND r.run_id=? AND {db.project_filter()}",
                (storage.org_id, retained.id, current_project_id() or ""),
            )
