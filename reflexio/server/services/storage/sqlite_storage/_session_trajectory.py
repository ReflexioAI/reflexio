"""Canonical session-trajectory helpers shared by session-outcome storage and its migration."""

import logging
import sqlite3
from collections.abc import Sequence
from hashlib import sha256

from reflexio.server.services.storage.retention_mixin import chunked
from reflexio.server.services.storage.session_outcome_identity import (
    CanonicalTrajectoryDigestAccumulator,
    CanonicalTrajectoryDigestResult,
    canonical_json_bytes,
)

from ._row_mappers import _iso_to_epoch

logger = logging.getLogger(__name__)

_TRAJECTORY_FETCH_SIZE = 256


_SESSION_OUTCOME_MIGRATION_BATCH_SIZE = 256


def _through_prefix_predicate(req_alias: str, cutover: str = "?") -> str:
    """SQL fragment: request ``req_alias`` is at or before a cutover request.

    The ONE definition of "the prefix through a cutover", shared by every
    SQLite reader that scopes to one (``get_requests_by_session``, the outcome
    trajectory digest, the outcome prefix precondition and the open-world
    evidence snapshot), so they cannot disagree about which requests a verdict
    covered.

    The cutover is looked up by id inside the same user/session and compared on its
    CURRENT row as ``(created_at, request_id)``, so ``request_id`` breaks a
    same-timestamp tie. A cutover that is not in the session matches nothing,
    which fails closed. With the default ``cutover`` the fragment binds exactly
    one ``?``, the cutover id; a reader holding the cutover in a column passes
    that column instead and binds nothing.

    Args:
        req_alias (str): Alias (or table name) of the outer ``requests`` row.
        cutover (str): SQL yielding the cutover's request id; ``?`` by default.

    Returns:
        str: A boolean SQL expression.
    """
    return (
        "EXISTS (SELECT 1 FROM requests AS cut"
        f" WHERE cut.request_id = {cutover}"
        f" AND cut.session_id = {req_alias}.session_id"
        f" AND cut.user_id = {req_alias}.user_id"
        f" AND ({req_alias}.created_at, {req_alias}.request_id)"
        " <= (cut.created_at, cut.request_id))"
    )


def _session_prefix_state(
    conn: sqlite3.Connection,
    session_id: str,
    through_request_id: str | None,
) -> tuple[int, int | None]:
    """Return ``(interaction_count, max_mutation_clock)`` for a session prefix.

    SQLite rows carry no ``updated_at``, so a row's mutation clock is its
    ``created_at``. ``max_mutation_clock`` is ``None`` when the prefix holds no
    request at all -- with a cutover, that means the cutover is not in the
    session.
    """
    predicate = ""
    params: list[object] = [session_id]
    if through_request_id is not None:
        predicate = " AND " + _through_prefix_predicate("r")
        params.append(through_request_id)
    row = conn.execute(
        f"""SELECT COUNT(i.interaction_id) AS interaction_count,
                   MAX(r.created_at) AS request_clock,
                   MAX(i.created_at) AS interaction_clock
            FROM requests AS r
            LEFT JOIN interactions AS i ON i.request_id = r.request_id
            WHERE r.session_id = ?{predicate}""",
        params,
    ).fetchone()
    if row is None or row["request_clock"] is None:
        return 0, None
    clocks = [_iso_to_epoch(str(row["request_clock"]))]
    if row["interaction_clock"] is not None:
        clocks.append(_iso_to_epoch(str(row["interaction_clock"])))
    return int(row["interaction_count"]), max(clocks)


def _canonical_session_trajectory_snapshot(
    conn: sqlite3.Connection,
    session_id: str,
    through_request_id: str | None = None,
) -> CanonicalTrajectoryDigestResult:
    """Hash and describe a complete bounded, ordered row stream.

    With ``through_request_id`` only the prefix through that cutover is
    hashed (see ``_through_prefix_predicate``). The accumulator is unchanged,
    so a prefix through the session's last request hashes byte-identically to
    the whole session.
    """
    prefix_predicate = ""
    params: list[object] = [session_id]
    if through_request_id is not None:
        prefix_predicate = " AND " + _through_prefix_predicate("requests")
        params.append(through_request_id)
    cursor = conn.execute(
        """SELECT requests.request_id, requests.user_id, requests.created_at,
                  requests.source, requests.agent_version, requests.session_id,
                  requests.evaluation_only, requests.retrieval_experiment_id,
                  requests.retrieval_experiment_arm,
                  requests.governance_subject_ref,
                  interactions.interaction_id,
                  interactions.user_id AS interaction_user_id,
                  interactions.request_id AS interaction_request_id,
                  interactions.created_at AS interaction_created_at,
                  interactions.content, interactions.role, interactions.token_count,
                  interactions.user_action, interactions.user_action_description,
                  interactions.interacted_image_url, interactions.image_encoding,
                  interactions.shadow_content, interactions.expert_content,
                  interactions.tools_used, interactions.citations,
                  interactions.retrieved_learnings
           FROM requests
           LEFT JOIN interactions
             ON interactions.request_id = requests.request_id
           WHERE requests.session_id = ?"""
        + prefix_predicate
        + """
           ORDER BY requests.created_at ASC, requests.request_id ASC,
                    interactions.created_at ASC, interactions.interaction_id ASC""",
        params,
    )
    accumulator = CanonicalTrajectoryDigestAccumulator(session_id)
    active_request_id: str | None = None
    first_request: dict[str, object] | None = None
    request_count = 0
    while rows := cursor.fetchmany(_TRAJECTORY_FETCH_SIZE):
        for row in rows:
            request_id = str(row["request_id"])
            if request_id != active_request_id:
                if active_request_id is not None:
                    accumulator.finish_request()
                accumulator.start_request(row)
                active_request_id = request_id
                request_count += 1
                if first_request is None:
                    first_request = {
                        key: row[key]
                        for key in (
                            "request_id",
                            "user_id",
                            "created_at",
                            "source",
                            "agent_version",
                            "session_id",
                            "evaluation_only",
                            "retrieval_experiment_id",
                            "retrieval_experiment_arm",
                            "governance_subject_ref",
                        )
                    }
            if row["interaction_id"] is None:
                continue
            interaction = dict(row)
            interaction["user_id"] = row["interaction_user_id"]
            interaction["request_id"] = row["interaction_request_id"]
            interaction["created_at"] = row["interaction_created_at"]
            accumulator.add_interaction(interaction)
    if active_request_id is not None:
        accumulator.finish_request()
    return CanonicalTrajectoryDigestResult(
        digest=accumulator.hexdigest(),
        first_request=first_request,
        request_count=request_count,
    )


def _canonical_session_trajectory_digest(
    conn: sqlite3.Connection, session_id: str
) -> str:
    return _canonical_session_trajectory_snapshot(conn, session_id).digest


def _prefetch_canonical_session_trajectory_digests(
    conn: sqlite3.Connection, session_ids: Sequence[str]
) -> dict[str, str]:
    """Derive trajectory digests from ordered joined rows in bounded chunks."""
    digests: dict[str, str] = {}
    for session_id_chunk in chunked(list(dict.fromkeys(session_ids))):
        placeholders = ",".join("?" for _ in session_id_chunk)
        cursor = conn.execute(
            f"""SELECT requests.request_id, requests.user_id, requests.created_at,
                       requests.source, requests.agent_version, requests.session_id,
                       requests.evaluation_only, requests.retrieval_experiment_id,
                       requests.retrieval_experiment_arm,
                       requests.governance_subject_ref,
                       interactions.interaction_id,
                       interactions.user_id AS interaction_user_id,
                       interactions.request_id AS interaction_request_id,
                       interactions.created_at AS interaction_created_at,
                       interactions.content, interactions.role,
                       interactions.token_count, interactions.user_action,
                       interactions.user_action_description,
                       interactions.interacted_image_url,
                       interactions.image_encoding, interactions.shadow_content,
                       interactions.expert_content, interactions.tools_used,
                       interactions.citations, interactions.retrieved_learnings
                FROM requests
                LEFT JOIN interactions
                  ON interactions.request_id = requests.request_id
                WHERE requests.session_id IN ({placeholders})
                ORDER BY requests.session_id ASC, requests.created_at ASC,
                         requests.request_id ASC,
                         interactions.created_at ASC,
                         interactions.interaction_id ASC""",  # noqa: S608
            session_id_chunk,
        )
        active_session_id: str | None = None
        active_request_id: str | None = None
        accumulator: CanonicalTrajectoryDigestAccumulator | None = None
        while rows := cursor.fetchmany(_TRAJECTORY_FETCH_SIZE):
            for row in rows:
                session_id = str(row["session_id"])
                request_id = str(row["request_id"])
                if accumulator is None or session_id != active_session_id:
                    if accumulator is not None:
                        if active_request_id is not None:
                            accumulator.finish_request()
                        assert active_session_id is not None  # noqa: S101
                        digests[active_session_id] = accumulator.hexdigest()
                    accumulator = CanonicalTrajectoryDigestAccumulator(session_id)
                    active_session_id = session_id
                    active_request_id = None
                if request_id != active_request_id:
                    if active_request_id is not None:
                        accumulator.finish_request()
                    accumulator.start_request(row)
                    active_request_id = request_id
                if row["interaction_id"] is not None:
                    interaction = dict(row)
                    interaction["user_id"] = row["interaction_user_id"]
                    interaction["request_id"] = row["interaction_request_id"]
                    interaction["created_at"] = row["interaction_created_at"]
                    accumulator.add_interaction(interaction)
        if accumulator is not None:
            if active_request_id is not None:
                accumulator.finish_request()
            assert active_session_id is not None  # noqa: S101
            digests[active_session_id] = accumulator.hexdigest()
        for session_id in session_id_chunk:
            digests.setdefault(
                session_id,
                CanonicalTrajectoryDigestAccumulator(session_id).hexdigest(),
            )
    return digests


def _legacy_session_outcome_id(user_id: str, session_id: str) -> str:
    """Return a delimiter-safe immutable identity for a migrated legacy outcome."""
    return sha256(canonical_json_bytes([user_id, session_id])).hexdigest()
