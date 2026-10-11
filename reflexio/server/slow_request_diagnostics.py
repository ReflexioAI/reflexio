"""Bounded stack diagnostics for stalled legacy search and account deletion."""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)
DIAGNOSTIC_PATHS = frozenset({"/api/search_profiles", "/api/account"})
SLOW_SECONDS = 20.0
_last_logged: float | None = None


def _thread_stacks() -> str:
    """Capture frame locations only: never locals, source lines or payloads."""
    stacks: dict[str, list[str]] = {}
    for thread_id, frame in list(sys._current_frames().items())[:128]:
        locations = []
        for _ in range(12):
            locations.append(
                f"{Path(frame.f_code.co_filename).name}:{frame.f_lineno}:"
                f"{frame.f_code.co_name}"
            )
            if frame.f_back is None:
                break
            frame = frame.f_back
        stacks[str(thread_id)] = locations
    return json.dumps(stacks, ensure_ascii=True)[:65536]


async def watch_slow_request(path: str, correlation_id: str | None) -> None:
    """Log once before the HTTP backstop, at most once per process per minute.

    The request owns and cancels this asyncio task. It neither interrupts work
    nor changes a response or billing decision, and creates no worker thread.
    """
    global _last_logged  # noqa: PLW0603
    await asyncio.sleep(SLOW_SECONDS)
    now = time.monotonic()
    if _last_logged is not None and now - _last_logged < 60:
        return
    _last_logged = now
    logger.warning(
        "event=slow_request_stacks path=%s correlation_id=%s stacks=%s",
        path,
        correlation_id,
        _thread_stacks(),
    )
