"""Replay committed v1 outputs; hold unfinished work across instruction changes."""

import copy
import json
import sqlite3
import time
from types import MappingProxyType, SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from reflexio.lib.reflexio_lib import Reflexio
from reflexio.models.config_schema import (
    Config,
    ProfileExtractorConfig,
    StorageConfigSQLite,
    UserPlaybookExtractorConfig,
)
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.metering_events import record_window_extraction_finances
from reflexio.server.services.configurator.configurator import DefaultConfigurator
from reflexio.server.services.durable_learning.worker import DurableLearningWorker
from reflexio.server.services.extraction.recovery import inspect_run
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.sqlite_storage._extraction_stream import (
    SQLiteStreamSQL,
)
from reflexio.server.services.storage.sqlite_storage._publication_receipts import (
    migrate_publication_receipts,
)
from reflexio.server.services.storage.storage_base import (
    AgentBinding,
    AgentRunRecord,
    AgentRunStatus,
)
from reflexio.server.usage_metrics import (
    UsageEventDeliveryError,
    UsageEventDeliveryStatus,
    configure_usage_event_recorder,
)
from reflexio.test_support.legacy_publication import restore_legacy_publication_fixture

pytestmark = pytest.mark.integration


def test_fresh_schema_needs_no_additive_column_upgrades(tmp_path, monkeypatch):
    alterations = []
    original_connect = sqlite3.connect

    def connect(*args, **kwargs):
        conn = original_connect(*args, **kwargs)

        def authorize(action, database, table, _column, _trigger):
            if action == sqlite3.SQLITE_ALTER_TABLE:
                alterations.append((database, table))
            return sqlite3.SQLITE_OK

        conn.set_authorizer(authorize)
        return conn

    monkeypatch.setattr(sqlite3, "connect", connect)
    storage = SQLiteStorage(org_id="fresh_schema", db_path=str(tmp_path / "fresh.db"))
    try:
        assert not storage.conn.in_transaction
        assert not alterations, f"Fresh initialization altered tables: {alterations!r}"
    finally:
        storage.conn.close()


def _row(conn):
    conn.row_factory = sqlite3.Row
    return dict(conn.execute("SELECT * FROM _agent_runs").fetchone())


@pytest.mark.parametrize(
    "phase",
    [
        "running",
        "completed_effects",
        "completed_projection_no_body",
        "completed_before_ack",
        "completed_acked",
        "completed_erased_acked",
    ],
)
def test_original_v1_cutover_replays_completed_and_holds_changed_instructions(
    tmp_path, monkeypatch, caplog, phase
):
    target = restore_legacy_publication_fixture(phase, tmp_path / "predecessor.sqlite")
    with sqlite3.connect(target) as destination:
        before = _row(destination)
        before_manifest = json.loads(before["generation_request_snapshot"])["overflow"]
        original_receipts = [
            tuple(row)
            for row in destination.execute(
                "SELECT * FROM _agent_run_finalization_receipts"
            ).fetchall()
        ]
        due = destination.execute(
            "SELECT max(effects_retry_at) FROM extraction_windows"
        ).fetchone()[0]
    monkeypatch.setenv("MOCK_LLM_RESPONSE", "false")
    monkeypatch.setenv(
        "REFLEXIO_LLM_CONTEXT_LIMITS_JSON",
        '{"minimax/MiniMax-M3":1000000,"gpt-4o":65536}',
    )
    monkeypatch.setattr(
        "reflexio.server.services.durable_learning.local.ensure_local_extraction",
        lambda _: None,
    )
    monkeypatch.setattr(
        "reflexio.server.services.tagging.tagging_scheduler.schedule_tagging",
        lambda **_: None,
    )
    monkeypatch.setattr(
        "reflexio.server.llm.input_budget.litellm.token_counter", lambda **_: 100
    )
    now = max(time.time(), (due or 0) + 1)
    monkeypatch.setattr(SQLiteStreamSQL, "now", lambda _: now)
    calls = []

    def respond(params, hard_timeout):
        del hard_timeout
        calls.append(1)
        schema = getattr(params.get("response_format"), "__name__", "") or json.dumps(
            params.get("response_format"), default=str
        )
        content = {
            "profiles": [
                {
                    "content": "The user prefers concise answers.",
                    "explicit_storage_deadline": None,
                    "source_span": "I prefer concise answers with details 0",
                }
            ]
        }
        if "ProfileDeduplicationOutput" in schema or any(
            "unique_ids" in str(m.get("content", "")) for m in params["messages"]
        ):
            content = {"duplicate_groups": [], "unique_ids": ["NEW-0"], "deletions": []}
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(content), tool_calls=None, refusal=None
                    ),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=20,
                total_tokens=120,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            ),
            model="minimax/MiniMax-M3",
            _hidden_params={"custom_llm_provider": "minimax"},
        )

    monkeypatch.setattr(
        LiteLLMClient, "_completion_with_hard_timeout", Mock(side_effect=respond)
    )
    config = DefaultConfigurator(org_id="stream-test", base_dir=str(tmp_path))
    config.set_config(
        Config(
            storage_config=StorageConfigSQLite(db_path=str(target)),
            window_size=10,
            stride_size=5,
            profile_extractor_config=None
            if phase == "completed_projection_no_body"
            else ProfileExtractorConfig(
                extraction_definition_prompt="Extract user preferences"
            ),
            user_playbook_extractor_config=UserPlaybookExtractorConfig(
                extraction_definition_prompt="Learn deployment instructions"
            )
            if phase == "completed_projection_no_body"
            else None,
            agent_success_config=None,
        )
    )
    engine = Reflexio(
        org_id="stream-test", storage_base_dir=str(tmp_path), configurator=config
    )
    storage = engine.get_storage()
    with sqlite3.connect(target) as conn:
        after_upgrade = _row(conn)
        assert all(after_upgrade[key] == value for key, value in before.items())
        assert [
            tuple(row)
            for row in conn.execute(
                "SELECT * FROM _agent_run_finalization_receipts"
            ).fetchall()
        ] == original_receipts
    accepted = {}
    attempts = []

    def record(event):
        if event.event_name not in ("extraction_tokens", "learnings_generated"):
            return UsageEventDeliveryStatus.EXEMPT
        attempts.append(event)
        if event.event_key in accepted:
            assert accepted[event.event_key] == event
            return UsageEventDeliveryStatus.DUPLICATE
        accepted[event.event_key] = event
        return UsageEventDeliveryStatus.APPENDED

    configure_usage_event_recorder(record)
    try:
        worker = DurableLearningWorker(lambda _: engine.request_context)
        if phase == "running":
            # The original policy includes the historical TTL authoring schema.
            # Finish this work under its original publisher before cutover; a
            # new image must not dispatch with a different instruction witness.
            worker.drain_org("stream-test", 3, 300)
            assert "Extraction input held kind=profile" in caplog.text
            assert "reason=input_budget_configuration_error" in caplog.text
            held = storage.get_agent_run(before["id"])
            assert held.status == AgentRunStatus.FAILED
            assert held.committed_output is None
            assert held.generation_request_snapshot == json.loads(
                before["generation_request_snapshot"]
            )
            assert calls == [] and accepted == {}
            assert storage.count_all_profiles() == 0
            assert storage.extraction_status("u", "original-v1")["status"] != "done"
            with sqlite3.connect(target) as conn:
                assert [
                    tuple(row)
                    for row in conn.execute(
                        "SELECT * FROM _agent_run_finalization_receipts"
                    ).fetchall()
                ] == original_receipts
            return
        worker.drain_org("stream-test", 3, 300)
        if phase != "completed_erased_acked":
            assert (
                storage.extraction_status(
                    "u",
                    "original-v1-projection"
                    if phase == "completed_projection_no_body"
                    else "original-v1",
                )["status"]
                == "done"
            )
        assert storage.count_all_profiles() == (
            0 if phase == "completed_projection_no_body" else 1
        )
        if phase == "completed_projection_no_body":
            assert len(storage.get_user_playbooks(user_id="u")) == 1
            assert before["committed_output"] is None
        assert storage.pending_extraction_effects() == []
        final = storage.get_agent_run(before["id"])
        assert final.status == (
            AgentRunStatus.AGENT_COMPLETED
            if phase == "completed_erased_acked"
            else AgentRunStatus.FINALIZED
        )
        final_manifest = final.generation_request_snapshot["overflow"]
        assert final_manifest["attempt_limit"] == before_manifest["attempt_limit"]
        assert (
            final_manifest["attempts"][: len(before_manifest["attempts"])]
            == before_manifest["attempts"]
        )
        if phase.startswith("completed"):
            assert calls == [] and final_manifest == before_manifest
        assert len(accepted) == (
            0 if phase in ("completed_acked", "completed_erased_acked") else 2
        )
        observed_calls = list(calls)
        observed_events = dict(accepted)
        worker.drain_org("stream-test", 2, 300)
        assert calls == observed_calls and accepted == observed_events
        with sqlite3.connect(target) as conn:
            if phase.startswith("completed"):
                assert [
                    tuple(row)
                    for row in conn.execute(
                        "SELECT * FROM _agent_run_finalization_receipts"
                    ).fetchall()
                ] == original_receipts
            assert (
                conn.execute("SELECT count(*) FROM _agent_output_receipts").fetchone()[
                    0
                ]
                == 0
            )
        print(
            "DRAFT_ORIGINAL_V1_FULL_OWNER_PROOF="
            + json.dumps(
                {
                    "phase": phase,
                    "startup_preserved_original_run_and_receipt": True,
                    "same_parent_finalized": final.status == AgentRunStatus.FINALIZED,
                    "acknowledged_erased_parent_unchanged": phase
                    == "completed_erased_acked",
                    "original_attempts_preserved": True,
                    "original_attempt_limit_preserved": True,
                    "profiles": 0 if phase == "completed_projection_no_body" else 1,
                    "playbooks": 1 if phase == "completed_projection_no_body" else 0,
                    "scripted_model_calls": len(calls),
                    "unique_financial_events": len(accepted),
                    "new_output_receipts": 0,
                    "second_turn_no_replay": True,
                    "outbound_inference_calls": 0,
                },
                sort_keys=True,
            )
        )
    finally:
        configure_usage_event_recorder(None)


def test_window_finance_validation_before_first_emit_and_partial_retry(tmp_path):
    target = restore_legacy_publication_fixture(
        "completed_effects", tmp_path / "finances.sqlite"
    )
    with sqlite3.connect(target) as conn:
        conn.row_factory = sqlite3.Row
        row = dict(conn.execute("SELECT * FROM extraction_windows").fetchone())
    billing = json.loads(row["effects"])["billing"]
    seen = []
    accepted = {}
    failed_token = False

    def record(event):
        nonlocal failed_token
        seen.append(event)
        if event.event_key.startswith("tok:") and not failed_token:
            failed_token = True
            return UsageEventDeliveryStatus.FAILED
        if event.event_key in accepted:
            assert accepted[event.event_key] == event
            return UsageEventDeliveryStatus.DUPLICATE
        accepted[event.event_key] = event
        return UsageEventDeliveryStatus.APPENDED

    def emit(value):
        record_window_extraction_finances(
            org_id="stream-test",
            window_id=row["window_id"],
            pipeline=row["kind"],
            billing=MappingProxyType(value),
            user_id=row["user_id"],
        )

    configure_usage_event_recorder(record)
    try:
        for field, bad in (
            ("prompt_tokens", -1),
            ("created_at", float("nan")),
            ("platform_llm", 1),
            ("ids", [""]),
            ("pipeline", "user_playbook" if row["kind"] == "profile" else "profile"),
        ):
            malformed = copy.deepcopy(billing)
            malformed[field] = bad
            with pytest.raises(ValueError):
                emit(malformed)
            assert seen == []
        with pytest.raises(UsageEventDeliveryError):
            emit(billing)
        assert len(accepted) == 1
        emit(billing)
        assert len(accepted) == 2
        compatible = dict(billing, pipeline=row["kind"])
        compatible.pop("cache_read_input_tokens", None)
        compatible.pop("cache_write_input_tokens", None)
        emit(compatible)
        assert len(accepted) == 2
        assert all(
            event.created_at == billing["created_at"] for event in accepted.values()
        )
        assert (
            accepted["tok:" + row["window_id"]].platform_llm == billing["platform_llm"]
        )
    finally:
        configure_usage_event_recorder(None)


@pytest.mark.parametrize(
    "change",
    [
        "ordinary_active",
        "ambiguous_receipt",
        "unknown_protocol",
        "scope_mismatch",
        "malformed_journal",
        "changed_policy",
        "changed_source",
    ],
)
def test_preserves_genuine_refusals_before_publication(
    tmp_path, monkeypatch, caplog, change
):
    target = restore_legacy_publication_fixture("running", tmp_path / "negative.sqlite")
    with sqlite3.connect(target) as destination:
        before = _row(destination)
        snapshot = json.loads(before["generation_request_snapshot"])
        attempts = copy.deepcopy(snapshot["overflow"]["attempts"])
        if change == "ordinary_active":
            destination.execute("UPDATE _agent_runs SET id=?", ("ar_" + "9" * 32,))
        elif change == "ambiguous_receipt":
            destination.execute(
                "INSERT INTO _agent_run_finalization_receipts(run_id,entity_type,learning_ids) VALUES(?,'profile','[]')",
                (before["id"],),
            )
        elif change in ("unknown_protocol", "malformed_journal"):
            snapshot["overflow"][
                "version" if change == "unknown_protocol" else "attempt_limit"
            ] = 99 if change == "unknown_protocol" else -1
            destination.execute(
                "UPDATE _agent_runs SET generation_request_snapshot=?",
                (json.dumps(snapshot),),
            )
        elif change == "scope_mismatch":
            destination.execute(
                "UPDATE extraction_windows SET user_id='wrong-synthetic-user'"
            )
        elif change == "changed_policy":
            policy = json.loads(
                destination.execute("SELECT policy FROM extraction_windows").fetchone()[
                    0
                ]
            )
            policy["extractor"]["extraction_definition_prompt"] = "A different policy"
            destination.execute(
                "UPDATE extraction_windows SET policy=?", (json.dumps(policy),)
            )
        elif change == "changed_source":
            pass  # Mutate through the initialized adapter, which owns SQLite UDFs.
        destination.commit()
    monkeypatch.setenv("MOCK_LLM_RESPONSE", "false")
    monkeypatch.setenv(
        "REFLEXIO_LLM_CONTEXT_LIMITS_JSON",
        '{"minimax/MiniMax-M3":1000000,"gpt-4o":65536}',
    )
    monkeypatch.setattr(
        "reflexio.server.services.durable_learning.local.ensure_local_extraction",
        lambda _: None,
    )
    monkeypatch.setattr(
        "reflexio.server.llm.input_budget.litellm.token_counter", lambda **_: 100
    )
    model = Mock(
        side_effect=AssertionError("Invalid original state must not reach inference")
    )
    monkeypatch.setattr(LiteLLMClient, "_completion_with_hard_timeout", model)
    events = []
    configure_usage_event_recorder(
        lambda event: events.append(event) or UsageEventDeliveryStatus.APPENDED
    )
    try:
        config = DefaultConfigurator(org_id="stream-test", base_dir=str(tmp_path))
        settings = Config(
            storage_config=StorageConfigSQLite(db_path=str(target)),
            window_size=10,
            stride_size=5,
            profile_extractor_config=ProfileExtractorConfig(
                extraction_definition_prompt="Extract user preferences"
            ),
            user_playbook_extractor_config=None,
            agent_success_config=None,
        )
        if change in (
            "ordinary_active",
            "ambiguous_receipt",
            "unknown_protocol",
            "scope_mismatch",
        ):
            with pytest.raises(ValueError, match="Legacy publication cutover refused"):
                config.set_config(settings)
        else:
            config.set_config(settings)
            engine = Reflexio(
                org_id="stream-test",
                storage_base_dir=str(tmp_path),
                configurator=config,
            )
            if change == "changed_source":
                with engine.get_storage()._stream_sql() as db:
                    db.query(
                        "UPDATE interactions SET content=content||?",
                        (" Changed synthetic source.",),
                    )
            worker = DurableLearningWorker(lambda _: engine.request_context)
            worker.drain_org("stream-test", 1, 300)
            assert engine.get_storage().count_all_profiles() == 0
            if change == "changed_source":
                assert (
                    engine.get_storage().get_agent_run(before["id"]).status
                    == AgentRunStatus.CANCELLED
                )
            else:
                assert any(record.levelno >= 30 for record in caplog.records)
        model.assert_not_called()
        assert not [
            event
            for event in events
            if event.event_name in ("learnings_generated", "extraction_tokens")
        ]
        with sqlite3.connect(target) as conn:
            row = _row(conn)
            assert (
                json.loads(row["generation_request_snapshot"])["overflow"]["attempts"]
                == attempts
            )
            assert conn.execute("SELECT count(*) FROM profiles").fetchone()[0] == 0
    finally:
        configure_usage_event_recorder(None)


@pytest.mark.parametrize("status", ["failed", "expired", "cancelled"])
def test_ordinary_terminal_output_is_inert_after_upgrade(tmp_path, monkeypatch, status):
    storage = SQLiteStorage(
        org_id="terminal-history", db_path=str(tmp_path / "ordinary.sqlite")
    )
    try:
        run = storage.create_agent_run(
            AgentRunRecord(
                id="ar_" + "8" * 32,
                binding=AgentBinding(
                    org_id=storage.org_id,
                    extractor_kind="profile",
                    user_id="user",
                    request_id="old-request",
                    agent_version=None,
                    source="old-writer",
                ),
                status=AgentRunStatus(status),
                generation_request_snapshot={},
                committed_output={"profiles": []},
            )
        )
        # Model the predecessor before the output-scoped ledger existed.
        with storage.conn:
            storage.conn.execute("DROP TABLE _agent_output_receipts")
        before = tuple(storage.conn.execute("SELECT * FROM _agent_runs").fetchone())
        migrate_publication_receipts(storage.conn, org_id=storage.org_id)
        config = Config(
            storage_config=StorageConfigSQLite(),
            profile_extractor_config=ProfileExtractorConfig(
                extraction_definition_prompt="test"
            ),
        )
        context = RequestContext.__new__(RequestContext)
        context.org_id, context.storage = storage.org_id, storage
        context.configurator = MagicMock()
        context.configurator.get_config.return_value = config
        assert "not_finalization_candidate" in inspect_run(context, run)["blockers"]
        model = Mock(
            side_effect=AssertionError("Inert history must not reach inference")
        )
        monkeypatch.setattr(LiteLLMClient, "_completion_with_hard_timeout", model)
        assert (
            DurableLearningWorker(lambda _: context).drain_org(storage.org_id, 1, 300)
            == 0
        )
        model.assert_not_called()
        assert (
            tuple(storage.conn.execute("SELECT * FROM _agent_runs").fetchone())
            == before
        )
        assert (
            storage.conn.execute(
                "SELECT count(*) FROM _agent_run_finalization_receipts"
            ).fetchone()[0]
            == 0
        )
        assert (
            storage.conn.execute(
                "SELECT count(*) FROM _agent_output_receipts"
            ).fetchone()[0]
            == 0
        )
    finally:
        storage.conn.close()
