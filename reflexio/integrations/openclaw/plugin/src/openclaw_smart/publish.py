"""Publish-to-reflexio orchestration used by agent_end, session_end, and the sync CLI.

One helper — ``publish_unpublished`` — owns the read-buffer → slice → publish →
stamp-watermark sequence so the three call sites stay in sync. Returns a
``(status, interaction_count)`` tuple so callers can format appropriate
messaging without peeking at the adapter.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Literal

from openclaw_smart import state
from openclaw_smart.reflexio_adapter import Adapter

try:
    import fcntl  # POSIX only — Windows falls back to no publish lock.
except ImportError:  # pragma: no cover — non-POSIX platforms
    fcntl = None  # type: ignore[assignment]

PublishStatus = Literal["nothing", "ok", "failed"]


@contextlib.contextmanager
def _session_publish_lock(session_id: str) -> Iterator[None]:
    """Serialize read-publish-watermark for one session buffer."""
    lock_path = state.publish_lock_path(session_id)
    if lock_path is None or fcntl is None:
        yield
        return
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def publish_unpublished(
    *,
    session_id: str,
    project_id: str,
    force_extraction: bool,
    skip_aggregation: bool,
    adapter: Adapter | None = None,
) -> tuple[PublishStatus, int]:
    """Drain the session buffer to reflexio and stamp the high-water mark.

    Args:
        session_id (str): openClaw session id (``sessionKey``), attached to
            each interaction.
        project_id (str): Stable project name; used as reflexio's ``user_id``
            (preferences) so preferences accumulate at the project level
            across sessions. ``agent_version`` is hardcoded to ``"openclaw"``
            in the adapter so skills roll up globally per agent rather than
            per project.
        force_extraction (bool): Whether to ask reflexio to run extraction
            synchronously instead of queuing for the next sweep.
        skip_aggregation (bool): When True, reflexio extracts preferences and
            raw project-specific skill entries but skips the rollup into
            shared skills. openclaw-smart passes False on every publish path
            so ``user_playbooks`` roll up into ``agent_playbooks``;
            aggregation additionally requires ``aggregation_config`` to be
            set on reflexio's ``user_playbook_extractor_configs[0]`` and
            ``optimize_agent_playbooks=true`` at the top level — otherwise
            the rollup silently no-ops.
        adapter (Adapter | None): Injection point for tests; a fresh
            ``Adapter()`` is constructed when omitted.

    Returns:
        tuple[PublishStatus, int]: ``("nothing", 0)`` if the buffer has no
            unpublished turns, ``("ok", n)`` after a successful publish of
            ``n`` interactions, or ``("failed", n)`` if reflexio rejected or
            was unreachable. On ``"failed"`` the watermark is not advanced,
            so the next hook retries the same batch.
    """
    with _session_publish_lock(session_id):
        records = state.read_all(session_id)
        _, interactions = state.unpublished_slice(records)
        if not interactions:
            return ("nothing", 0)
        client = adapter if adapter is not None else Adapter()
        ok = client.publish(
            session_id=session_id,
            project_id=project_id,
            interactions=interactions,
            force_extraction=force_extraction,
            skip_aggregation=skip_aggregation,
        )
        if ok:
            state.append(session_id, {"published_up_to": len(records)})
            return ("ok", len(interactions))
        return ("failed", len(interactions))
