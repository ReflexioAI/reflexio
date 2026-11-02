"""Answer digest capture is isolated data, not an execution permission."""

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import asdict, replace

import pytest

from reflexio.server.llm.input_budget import InputBudgetConfigurationError
from reflexio.server.services.extraction import answer_inputs
from reflexio.server.services.extraction.overflow_answers import (
    answer_witness,
    pin_extraction_answer,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.storage_base import build_scope_hash
from tests.server.services.extraction import test_prior_answer_search as prior

storage = prior.storage
pytestmark = pytest.mark.integration


def test_capture_keeps_only_detached_digests_across_matching_nested_calls(storage):
    record = prior._capture_record(storage)
    expected = {record.id: build_scope_hash(answer_witness(asdict(record)))}
    with answer_inputs.answer_input_scope(storage, "run"):
        pin_extraction_answer(record)
        with answer_inputs.answer_input_scope(storage, "run"):
            assert answer_inputs.snapshot_answer_inputs(storage, "run") == expected
        detached = answer_inputs.snapshot_answer_inputs(storage, "run")
        detached.clear()
        assert answer_inputs.snapshot_answer_inputs(storage, "run") == expected
    assert not answer_inputs.has_answer_input_capture(storage)
    with pytest.raises(InputBudgetConfigurationError, match="unavailable"):
        answer_inputs.snapshot_answer_inputs(storage, "run")


def test_other_run_capture_is_separate_and_outer_capture_is_restored(storage):
    record = prior._capture_record(storage)
    with answer_inputs.answer_input_scope(storage, "outer"):
        pin_extraction_answer(record)
        original = answer_inputs.snapshot_answer_inputs(storage, "outer")
        with answer_inputs.answer_input_scope(storage, "inner"):
            assert answer_inputs.snapshot_answer_inputs(storage, "inner") == {}
            pin_extraction_answer(replace(record, result={"answer": "other input"}))
        assert answer_inputs.snapshot_answer_inputs(storage, "outer") == original


def test_capture_rejects_another_storage_run_or_project(storage, tmp_path, monkeypatch):
    other = SQLiteStorage(org_id=storage.org_id, db_path=str(tmp_path / "other.db"))
    monkeypatch.setattr(answer_inputs, "current_project_id", lambda: "project-one")
    with answer_inputs.answer_input_scope(storage, "run"):
        with pytest.raises(InputBudgetConfigurationError, match="target changed"):
            answer_inputs.snapshot_answer_inputs(other, "run")
        with pytest.raises(InputBudgetConfigurationError, match="target changed"):
            answer_inputs.snapshot_answer_inputs(storage, "different-run")
        monkeypatch.setattr(answer_inputs, "current_project_id", lambda: "project-two")
        with pytest.raises(InputBudgetConfigurationError, match="target changed"):
            answer_inputs.snapshot_answer_inputs(storage, "run")


def test_answer_change_within_one_input_is_refused_without_overwriting(storage):
    record = prior._capture_record(storage)
    with answer_inputs.answer_input_scope(storage, "run"):
        pin_extraction_answer(record)
        before = answer_inputs.snapshot_answer_inputs(storage, "run")
        with pytest.raises(InputBudgetConfigurationError, match="Answer changed"):
            pin_extraction_answer(replace(record, result={"answer": "changed"}))
        assert answer_inputs.snapshot_answer_inputs(storage, "run") == before


def test_capture_reset_and_copied_context_cannot_reopen_finished_run(storage):
    record = prior._capture_record(storage)
    with (
        pytest.raises(RuntimeError, match="controlled failure"),
        answer_inputs.answer_input_scope(storage, "run"),
    ):
        copied = copy_context()
        raise RuntimeError("controlled failure")
    assert not answer_inputs.has_answer_input_capture(storage)
    with pytest.raises(InputBudgetConfigurationError, match="capture ended"):
        copied.run(pin_extraction_answer, record)

    def reopen():
        with answer_inputs.answer_input_scope(storage, "run"):
            pass

    with pytest.raises(InputBudgetConfigurationError, match="capture ended"):
        copied.run(reopen)
    with answer_inputs.answer_input_scope(storage, "run"):
        assert answer_inputs.snapshot_answer_inputs(storage, "run") == {}


def test_concurrent_conflicting_answer_inputs_never_overwrite_each_other(storage):
    record = prior._capture_record(storage)
    changed = replace(record, result={"answer": "a competing input"})

    def remember(value):
        try:
            pin_extraction_answer(value)
            return "accepted"
        except InputBudgetConfigurationError:
            return "refused"

    with answer_inputs.answer_input_scope(storage, "run"):
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(copy_context().run, remember, value)
                for value in (record, changed)
            ]
            assert sorted(f.result(timeout=5) for f in futures) == [
                "accepted",
                "refused",
            ]
        snapshot = answer_inputs.snapshot_answer_inputs(storage, "run")
        assert snapshot[record.id] in {
            build_scope_hash(answer_witness(asdict(value)))
            for value in (record, changed)
        }


def test_foreign_answer_is_refused_without_creating_input_evidence(storage):
    record = prior._capture_record(storage)
    with answer_inputs.answer_input_scope(storage, "run"):
        with pytest.raises(InputBudgetConfigurationError, match="unavailable"):
            pin_extraction_answer(replace(record, org_id="foreign-org"))
        assert answer_inputs.snapshot_answer_inputs(storage, "run") == {}


def test_prior_lookup_identity_mismatch_is_refused_before_rendering(
    storage, monkeypatch
):
    record = prior._capture_record(storage)
    monkeypatch.setattr(
        storage, "get_pending_tool_call", lambda _id: replace(record, id="wrong-answer")
    )
    with answer_inputs.answer_input_scope(storage, "run"):
        with pytest.raises(
            InputBudgetConfigurationError, match="changed during capture"
        ):
            prior.append_prior_knowledge_context(
                messages=[{"role": "user", "content": "extract"}],
                storage=storage,
                org_id=storage.org_id,
                extractor_kind="profile",
                extractor_config=prior._ExtractorConfig(),
                source="api",
                agent_version="v1",
            )
        assert answer_inputs.snapshot_answer_inputs(storage, "run") == {}
