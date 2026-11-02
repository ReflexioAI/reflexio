"""Source-backed ordinary outputs authored by the actual owned agent writer."""

import uuid
from unittest.mock import MagicMock, patch

from pydantic import BaseModel

from reflexio.server.llm.tools import ToolLoopResult, ToolLoopTrace
from reflexio.server.services.durable_learning.user_lease import (
    create_owned_agent_run,
    user_extraction_lease,
)
from reflexio.server.services.extraction.answer_inputs import answer_input_scope
from reflexio.server.services.extraction.history_inputs import HistoricalInputs
from reflexio.server.services.extraction.publication_receipts import PublicationReceipt
from reflexio.server.services.extraction.resumable_agent import ResumableExtractionAgent
from reflexio.server.services.extraction.resume_worker import (
    _request_interaction_models_from_ids,
)
from reflexio.server.services.extraction.source_inputs import capture_source_inputs
from reflexio.server.services.storage.storage_base import (
    AgentRunRecord,
    AgentRunStatus,
    BaseStorage,
)


def observed_financial_receipt(run, learning_ids, *, entity_type):
    """Delivery-unit fixture: validated observed facts, never execution authority.

    Real writer/store integration tests separately prove how these immutable
    facts are created. Delivery-only tests explicitly stub external acceptance
    or the row acknowledgement; they do not claim that this row was persisted.
    """
    clock = run.created_at or run.agent_completed_at
    return PublicationReceipt.from_row(
        {
            "receipt_id": uuid.uuid4().hex,
            "org_id": run.binding.org_id,
            "project_id": run.project_id or "",
            "run_id": run.id,
            "output_id": uuid.uuid4().hex,
            "row_kind": "output",
            "entity_type": entity_type,
            "learning_ids": learning_ids,
            "delivery_owner": "receipt",
            "delivery_state": "pending" if learning_ids else "accepted",
            "delivery_key": None,
            "delivery_facts": {
                "pipeline": run.binding.extractor_kind,
                "source": "resumable_extraction",
                "created_at": clock.timestamp() if clock is not None else None,
            }
            if learning_ids
            else None,
        }
    )


def write_completed_output(
    storage: BaseStorage, run: AgentRunRecord, output: BaseModel
) -> AgentRunRecord:
    """Script only the provider result, never a UUID or original-input proof.

    These fresh-output fixtures have no supplied human answers. Callers which
    need dependencies use the real resume worker instead of manufacturing hashes.
    """
    assert run.status == AgentRunStatus.RUNNING and run.committed_output is None
    assert run.binding.user_id
    assert (
        run.generation_request_snapshot.get("output_schema_name", type(output).__name__)
        == type(output).__name__
    )
    with user_extraction_lease(storage, run.binding.user_id, max_wait_seconds=0):
        run = create_owned_agent_run(storage, run)
        sources = capture_source_inputs(
            storage,
            run,
            _request_interaction_models_from_ids(
                storage,
                run.binding.source_interaction_ids,
                fallback_source=run.binding.source,
                fallback_agent_version=run.binding.agent_version,
            ),
        )
        with (
            answer_input_scope(storage, run.id),
            patch(
                "reflexio.server.services.extraction.resumable_agent.run_tool_loop",
                return_value=ToolLoopResult(
                    ctx=None,
                    trace=ToolLoopTrace(finished=True),
                    finished_reason="structured_output",
                    structured_output=output,
                    max_steps_remaining=8,
                ),
            ),
        ):
            result = ResumableExtractionAgent(client=MagicMock(), storage=storage)._run(
                run=run,
                publication_origin="resume",
                messages=[],
                history_inputs=HistoricalInputs(),
                output_schema=type(output),
                source_inputs=sources,
                dependency_inputs={},
            )
        assert result.execution_run is not None
        return result.execution_run
