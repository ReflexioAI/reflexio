# ruff: noqa: S608, S101 -- Fixed fixture tables and synthetic setup assertions.
"""Committed synthetic windows through actual storage owners, no generation."""

import json
from typing import Any

from reflexio.server.services.storage.storage_base import (
    AgentBinding,
    AgentRunRecord,
    AgentRunStatus,
)
from reflexio.server.services.storage.storage_base._extraction_stream import Window
from reflexio.server.work_scope import current_project_id


def commit_window(
    context: Any, *, skipped: bool = True, parent: bool = False, billing: bool = False
) -> Window:
    storage = context.storage
    project = current_project_id() or ""
    window = Window(
        window_id="selected-effects",
        user_id="user",
        kind="profile",
        project_id=project,
        predecessor=0,
        end_seq=1,
        force=False,
        skip_aggregation=False,
        manifest=[{"interaction_id": 1, "request_id": "request", "seq": 1}],
        policy={
            "extractor": context.configurator.get_config().profile_extractor_config.model_dump(
                mode="json"
            ),
            "window_size": 1,
            "stride_size": 1,
        },
    )
    with storage._stream_sql() as db:
        if not db.query(
            f"SELECT request_id FROM {db.table('requests')} WHERE request_id='request'"
        ):
            db.query(
                f"INSERT INTO {db.table('requests')}(request_id,user_id,session_id"
                + (",project_id" if project else "")
                + ") VALUES('request','user','session'"
                + (",?" if project else "")
                + ")",
                (project,) if project else (),
            )
            db.query(
                f"INSERT INTO {db.table('interactions')}(interaction_id,user_id,request_id,role,content,user_action,ingestion_seq"
                + (",project_id" if project else "")
                + ") VALUES(1,'user','request','user','synthetic committed source','none',1"
                + (",?" if project else "")
                + ")",
                (project,) if project else (),
            )
        db.query(
            f"INSERT INTO {db.table('learning_work')}(org_id,user_id,highwater,pending_since) VALUES(?,'user',1,1) ON CONFLICT(org_id,user_id) DO UPDATE SET highwater=1,pending_since=1",
            (context.org_id,),
        )
        db.query(
            f"INSERT INTO {db.table('extraction_cursors')}(user_id,kind,project_id,completed_seq,window_id) VALUES('user','profile',?,0,?)",
            (project, window.window_id),
        )
        db.query(
            f"INSERT INTO {db.table('extraction_windows')}(window_id,user_id,kind,org_id,project_id,predecessor,end_seq,manifest,policy) VALUES(?,'user','profile',?,?,0,1,?,?)",
            (
                window.window_id,
                context.org_id,
                project,
                json.dumps(window.manifest),
                json.dumps(window.policy),
            ),
        )
    effects = {"version": 1, "skipped": skipped}
    if not skipped:
        effects.update(
            write_plan=None,
            generated_count=0,
            billable_count=0,
            extraction_run_ids=[f"window:{window.window_id}"] if parent else [],
            token_totals={"prompt_tokens": 5, "completion_tokens": 2},
            model_provenance=None,
            stats={},
        )
    if parent:
        run = AgentRunRecord(
            id=f"window:{window.window_id}",
            binding=AgentBinding(
                org_id=context.org_id,
                extractor_kind="profile",
                user_id="user",
                request_id="request",
                agent_version=None,
                source="api",
                source_interaction_ids=[1],
            ),
            project_id=project or None,
            status=AgentRunStatus.AGENT_COMPLETED,
            generation_request_snapshot={
                "output_schema_name": "HistoricalAlreadyPublishedOutput"
            },
            committed_output={"profiles": []},
        )
        storage.create_agent_run(run)
        storage.save_agent_run_finalization_receipt(
            run_id=run.id, entity_type="profile", learning_ids=[]
        )
    if billing:
        effects["billing"] = {
            "ids": [],
            "created_at": 1.0,
            "platform_llm": False,
            "input_tokens": 3,
            "prompt_tokens": 5,
            "completion_tokens": 2,
        }
    token = storage.claim_user_extraction("user", "synthetic-original-owner", 300)
    assert token is not None
    with storage.commit_scope():
        storage.complete_extraction(window, token, effects)
    return window


def grant(context: Any, selection: Any) -> Any:
    from reflexio.server.services.storage.storage_base.learning_transition import (
        _change_learning_policy_authority_db,
    )

    storage = context.storage
    before = storage.read_learning_transition()
    with storage._learning_policy_authority_scope() as (db, root):
        sealed = _change_learning_policy_authority_db(
            db,
            action="seal",
            operation_id="effects",
            expected_nonce=root.authority_nonce,
            expected_generation=root.generation,
            target_policy=root.active_policy,
        )
    project = before.project
    return storage.grant_learning_transition(
        "effects",
        expected_authority_nonce=sealed.authority_nonce,
        expected_authority_generation=sealed.generation,
        expected_nonce=project.nonce if project else None,
        expected_revision=project.revision if project else None,
        selection_sha256=selection.fingerprint(),
    )


def invoke(context: Any, selection: Any, admitted: Any) -> Any:
    from reflexio.server.services.durable_learning.selected_learning import (
        run_selected_learning_work,
    )

    return run_selected_learning_work(
        context,
        selection,
        operation_id="effects",
        expected_nonce=admitted.project.nonce,
        expected_revision=admitted.project.revision,
        expected_authority_nonce=admitted.authority.authority_nonce,
        expected_authority_generation=admitted.authority.generation,
    )
