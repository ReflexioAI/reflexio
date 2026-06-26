from __future__ import annotations

from pathlib import Path

_MODULE_DIR = (
    Path(__file__).resolve().parents[4]
    / "reflexio"
    / "server"
    / "services"
    / "extraction"
)
_EXPECTED_FILES = {
    "__init__.py",
    "agent_run_records.py",
    "outcome.py",
    "pending_tool_call_dispatch.py",
    "prior_answer_search.py",
    "resumable_agent.py",
    "resume_scheduler.py",
    "resume_worker.py",
    "README.md",
}
_REMOVED_FILES = {"tools.py", "plan.py", "invariants.py"}


def test_extraction_canonical_imports_work() -> None:
    from reflexio.server.services.extraction.agent_run_records import (
        build_extractor_agent_run_record,
    )
    from reflexio.server.services.extraction.outcome import ExtractionOutcome
    from reflexio.server.services.extraction.pending_tool_call_dispatch import (
        PendingToolCallToolContext,
        create_ask_human_tool,
        create_attach_pending_info_request_tool,
    )
    from reflexio.server.services.extraction.prior_answer_search import (
        append_prior_knowledge_context,
    )
    from reflexio.server.services.extraction.resumable_agent import (
        AgentRunResult,
        ResumableExtractionAgent,
        run_resumable_extraction_agent,
    )
    from reflexio.server.services.extraction.resume_scheduler import (
        ExtractionResumeScheduler,
        maybe_start_resume_scheduler,
    )
    from reflexio.server.services.extraction.resume_worker import (
        ExtractionResumeWorker,
        ResumeWorkerError,
    )

    assert ExtractionOutcome.__name__ == "ExtractionOutcome"
    assert build_extractor_agent_run_record.__name__ == (
        "build_extractor_agent_run_record"
    )
    assert PendingToolCallToolContext.__name__ == "PendingToolCallToolContext"
    assert create_ask_human_tool.__name__ == "create_ask_human_tool"
    assert create_attach_pending_info_request_tool.__name__ == (
        "create_attach_pending_info_request_tool"
    )
    assert append_prior_knowledge_context.__name__ == "append_prior_knowledge_context"
    assert AgentRunResult.__name__ == "AgentRunResult"
    assert ResumableExtractionAgent.__name__ == "ResumableExtractionAgent"
    assert run_resumable_extraction_agent.__name__ == "run_resumable_extraction_agent"
    assert ExtractionResumeScheduler.__name__ == "ExtractionResumeScheduler"
    assert maybe_start_resume_scheduler.__name__ == "maybe_start_resume_scheduler"
    assert ExtractionResumeWorker.__name__ == "ExtractionResumeWorker"
    assert ResumeWorkerError.__name__ == "ResumeWorkerError"


def test_extraction_file_layout_stays_compact_and_current() -> None:
    source_files = {path.name for path in _MODULE_DIR.iterdir() if path.is_file()}

    assert source_files >= _EXPECTED_FILES
    assert source_files.isdisjoint(_REMOVED_FILES)
