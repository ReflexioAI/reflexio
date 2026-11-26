# ruff: noqa: S608 -- Only native fixed table names are interpolated.
"""Owned partial procedural execution; never a committed learning/output."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import TYPE_CHECKING, Any, cast

from reflexio.server.llm.input_budget import ExtractionInputIntegrityError
from reflexio.server.work_scope import current_project_id

from ._agent_run import AgentRunRecord
from ._extraction_stream import LeaseLostError, decoded

if TYPE_CHECKING:
    from . import BaseStorage

KEY = "procedural_execution"


class ProceduralExecutionStore:
    def checkpoint_procedural_execution(
        self,
        run: AgentRunRecord,
        *,
        expected: dict[str, Any] | None,
        save: dict[str, Any],
    ) -> dict[str, Any]:
        """Compare-and-swap one source-bound checkpoint under the actual run owner."""
        from reflexio.server.services.durable_learning.user_lease import (
            owned_agent_run_write,
        )

        storage = cast("BaseStorage", self)
        requested = deepcopy(save)
        if (
            run.binding.extractor_kind != "playbook"
            or run.generation_request_snapshot.get("output_schema_name")
            != "ProceduralOutput"
            or run.committed_output is not None
            or requested.get("version") != 1
            or type(requested.get("revision")) is not int
            or type(requested.get("physical_calls")) is not int
            or not 0 <= requested["physical_calls"] <= 6
            or not isinstance(requested.get("contract"), dict)
            or not isinstance(requested.get("completed"), dict)
            or not isinstance(requested.get("failed"), dict)
        ):
            raise ExtractionInputIntegrityError(
                "Invalid procedural execution checkpoint"
            )
        usage = requested.get("observed_usage")
        fields = {
            "prompt_tokens",
            "completion_tokens",
            "cache_read_input_tokens",
            "cache_write_input_tokens",
        }
        if usage is not None and (
            not isinstance(usage, dict)
            or set(usage) != fields
            or any(type(value) is not int or value < 0 for value in usage.values())
        ):
            raise ExtractionInputIntegrityError("Invalid procedural observed usage")
        previous = deepcopy(expected)
        earlier_usage = previous.get("observed_usage") if previous is not None else None
        if earlier_usage is not None and (
            usage is None
            or any(usage[field] < earlier_usage[field] for field in fields)
        ):
            raise ExtractionInputIntegrityError(
                "Procedural observed usage cannot decrease"
            )
        if requested["revision"] != (previous["revision"] + 1 if previous else 0):
            raise ExtractionInputIntegrityError(
                "Procedural checkpoint revision changed"
            )
        if previous is None and (
            requested["physical_calls"] != 0
            or requested["completed"]
            or requested["failed"]
        ):
            raise ExtractionInputIntegrityError(
                "Initial procedural checkpoint must precede dispatch"
            )
        if previous is not None and (
            requested["contract"] != previous["contract"]
            or not previous["physical_calls"]
            <= requested["physical_calls"]
            <= previous["physical_calls"] + 1
            or any(
                requested["completed"].get(key) != value
                for key, value in previous["completed"].items()
            )
            or any(
                requested["failed"].get(key) != value
                for key, value in previous["failed"].items()
            )
            or requested.get("trace", {}).get("turns", [])[
                : len(previous.get("trace", {}).get("turns", []))
            ]
            != previous.get("trace", {}).get("turns", [])
        ):
            raise ExtractionInputIntegrityError(
                "Procedural execution cannot rewrite retained work"
            )
        with owned_agent_run_write(storage, run), storage._stream_sql() as db:
            storage._require_current_agent_output(db, run, None)
            rows = db.query(
                f"SELECT r.generation_request_snapshot,r.status FROM {db.table('_agent_runs')} r "
                f"WHERE r.org_id=? AND r.id=? AND {db.project_filter()}"
                + db.validation_lock(),
                (storage.org_id, run.id, current_project_id() or ""),
            )
            if not rows or rows[0]["status"] not in {"running", "resuming"}:
                raise LeaseLostError("Procedural execution is no longer computing")
            snapshot = decoded(rows[0]["generation_request_snapshot"])
            if (
                not isinstance(snapshot, dict)
                or snapshot.get("output_schema_name") != "ProceduralOutput"
                or snapshot.get(KEY) != previous
            ):
                raise LeaseLostError("Procedural checkpoint revision was superseded")
            contract = requested["contract"]
            # Reuse original-source and historical native locks; this temporary
            # envelope is only validator input and is never retained as output.
            storage._lock_agent_source_inputs(
                db, run, {"publication": {"source_inputs": contract["source_inputs"]}}
            )
            storage._lock_historical_rows(
                db,
                user_id=run.binding.user_id or "",
                project_id=current_project_id() or "",
                profiles=contract["history"]["profiles"],
                playbooks=contract["history"]["playbooks"],
            )
            snapshot[KEY] = requested
            db.query(
                f"UPDATE {db.table('_agent_runs')} AS r SET generation_request_snapshot=? "
                f"WHERE r.org_id=? AND r.id=? AND {db.project_filter()}",
                (
                    json.dumps(snapshot, allow_nan=False),
                    storage.org_id,
                    run.id,
                    current_project_id() or "",
                ),
            )
            return deepcopy(requested)
