# ruff: noqa: S608 -- Synthetic fixtures on the existing disposable storage owners.
"""Shared SQLite/native transition-closure synthetic evidence, without providers."""

import json
from typing import Any

from reflexio.server.services.storage.storage_base._extraction_stream import StreamSQL
from reflexio.server.services.storage.storage_base.transition_closure import (
    ProjectTransitionClosure,
    derive_project_transition_closure_db,
)


def seed(db: StreamSQL, org: str, project: str, *, history: bool = False) -> None:
    db.query(
        f"INSERT INTO {db.table('learning_work')}(org_id,user_id,highwater,pending_since) VALUES(?,'user',2,1)",
        (org,),
    )
    for kind in ("profile", "playbook"):
        db.query(
            f"INSERT INTO {db.table('extraction_cursors')}(user_id,kind,project_id,completed_seq,started) VALUES('user',?,?,?,?)",
            (
                kind,
                project,
                1 if history and kind == "profile" else 0,
                1 if history and kind == "profile" else 0,
            ),
        )
    for seq in (1, 2):
        request = f"request-{seq}" if history else "request"
        if seq == 1 or history:
            admission = {
                "profile": {"eligible": True},
                "playbook": {"eligible": False},
                "min_seq": seq if history else 1,
                "max_seq": seq if history else 2,
                "admitted_at": 1,
            }
            # SQLite has no source project column; dialect owner neutral predicate
            # tells this fixture which real schema it is populating.
            native = bool(project)
            db.query(
                f"INSERT INTO {db.table('requests')}(request_id,user_id,created_at,session_id,learning_admission"
                + (",project_id" if native else "")
                + ") VALUES(?,'user','2026-01-01','session',?"
                + (",?" if native else "")
                + ")",
                (request, json.dumps(admission), project)
                if native
                else (request, json.dumps(admission)),
            )
        native = bool(project)
        db.query(
            f"INSERT INTO {db.table('interactions')}(interaction_id,user_id,created_at,request_id,ingestion_seq"
            + (",project_id,content,user_action" if native else "")
            + ") VALUES(?,'user','2026-01-01',?,?"
            + (",?,'synthetic','none'" if native else "")
            + ")",
            (seq, request, seq, project) if native else (seq, request, seq),
        )
    if history:
        db.query(
            f"INSERT INTO {db.table('extraction_windows')}(window_id,user_id,kind,org_id,project_id,predecessor,end_seq,manifest,policy,completed,effects_done,completed_at) VALUES('past','user','profile',?,?,0,1,'[]','{{}}',1,1,2)",
            (org, project),
        )


def derive(storage: Any, project: str = "") -> ProjectTransitionClosure:
    with storage.commit_scope(), storage._stream_sql() as db:
        return derive_project_transition_closure_db(
            db, org_id=storage.org_id, project_id=project
        )


CASES = (
    "missing_boolean",
    "string_boolean",
    "missing_source",
    "wrong_min",
    "unfinished_window",
    "running",
    "held",
    "missing_completion",
    "unproven_highwater",
)


def mutate(db: StreamSQL, org: str, project: str, case: str) -> None:
    if case in {"missing_boolean", "string_boolean", "wrong_min"}:
        admission = {
            "profile": {"eligible": True},
            "playbook": {"eligible": False},
            "min_seq": 1,
            "max_seq": 2,
            "admitted_at": 1,
        }
        if case == "missing_boolean":
            del admission["playbook"]["eligible"]
        elif case == "string_boolean":
            admission["playbook"]["eligible"] = "false"
        else:
            admission.update(min_seq=2, max_seq=3)
            db.query(
                f"UPDATE {db.table('interactions')} SET ingestion_seq=3 WHERE interaction_id=2"
            )
            db.query(f"UPDATE {db.table('learning_work')} SET highwater=3")
        db.query(
            f"UPDATE {db.table('requests')} SET learning_admission=?",
            (json.dumps(admission),),
        )
    elif case == "missing_source":
        db.query(f"DELETE FROM {db.table('interactions')} WHERE interaction_id=1")
    elif case == "unfinished_window":
        db.query(
            f"INSERT INTO {db.table('extraction_windows')}(window_id,user_id,kind,org_id,project_id,predecessor,end_seq,manifest,policy) VALUES('unfinished','user','profile',?,?,0,2,'[]','{{}}')",
            (org, project),
        )
    elif case == "running":
        db.query(
            f"INSERT INTO {db.table('_agent_runs')}(id,org_id,user_id,extractor_kind,request_id,status"
            + (",project_id" if project else "")
            + ") VALUES('running',?,'user','profile','request','running'"
            + (",?" if project else "")
            + ")",
            (org, project) if project else (org,),
        )
    elif case == "held":
        db.query(
            f"UPDATE {db.table('extraction_cursors')} SET last_error='synthetic-held'"
        )
    elif case == "missing_completion":
        db.query(
            f"UPDATE {db.table('extraction_cursors')} SET completed_seq=1,started=1 WHERE kind='profile'"
        )
    elif case == "unproven_highwater":
        db.query(f"UPDATE {db.table('learning_work')} SET highwater=3")
