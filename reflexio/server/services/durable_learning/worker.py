"""Bounded execution of durable sliding windows with renewable user leases."""

from __future__ import annotations

import logging
import os
import threading
import uuid
from collections.abc import Callable
from pathlib import Path

from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.env_utils import env_str
from reflexio.server.operation_limiter import operation_limit_value
from reflexio.server.services.durable_learning.window_executor import (
    WindowExecutor,
    WindowInputsDeletedError,
)
from reflexio.server.services.storage.storage_base._extraction_stream import (
    LeaseLostError,
    Window,
)
from reflexio.server.work_scope import WorkScope, WorkScopeError, bind_work_scope

logger = logging.getLogger(__name__)
_budget_lock = threading.Lock()
_budget: threading.BoundedSemaphore | None = None

# ``.../reflexio`` - the installed package root, used only to tell our own
# frames from a dependency's. Never rendered into a log line itself.
_FIRST_PARTY_ROOT = str(Path(__file__).resolve().parents[3]) + os.sep
# A locator is two path components plus a line number, so it is already short;
# the cap makes the bound total rather than merely customary, because this
# value is persisted as well as logged.
_LOCATOR_MAX = 120


def _tb_positions(exc: BaseException) -> list[tuple[str, int]]:
    """Walk the traceback for code positions only.

    Deliberately not ``traceback.extract_tb``: that populates each frame's
    ``.line`` by reading the source file, which both costs disk I/O on a
    failure path and puts source text on an object we hand to a logger.
    ``co_filename`` and ``tb_lineno`` are integers and paths, nothing else.
    """
    positions: list[tuple[str, int]] = []
    tb = exc.__traceback__
    while tb is not None:
        positions.append((tb.tb_frame.f_code.co_filename, tb.tb_lineno))
        tb = tb.tb_next
    return positions


def _position(filename: str, lineno: int) -> str:
    """``parent/file.py:12`` - bounded, and enough to disambiguate.

    One component would be ambiguous in this tree (``sqlite_storage`` and
    ``storage_base`` both hold an ``_extraction_stream.py``); two is enough,
    and keeps an absolute container path out of the field.
    """
    return f"{'/'.join(Path(filename).parts[-2:])}:{lineno}"


def exception_locator(exc: BaseException) -> str:
    """Where an exception came from, as source coordinates and nothing else.

    A file path and line number are *our* coordinates, so unlike an exception
    message they can never carry customer content. That is what makes this
    safe to log and to store where ``str(exc)`` is not.

    The raise site is reported first. When it lands inside a dependency the
    innermost first-party frame is appended after ``<-``, because
    ``pydantic/main.py:210`` alone does not say which of our calls provoked
    it. When the two coincide only one is emitted.
    """
    positions = _tb_positions(exc)
    if not positions:
        return "unknown"
    raised = _position(*positions[-1])
    for filename, lineno in reversed(positions):
        if filename.startswith(_FIRST_PARTY_ROOT):
            ours = _position(filename, lineno)
            if ours != raised:
                raised = f"{raised}<-{ours}"
            break
    return raised[:_LOCATOR_MAX]


def worker_count() -> int:
    return max(
        1,
        int(
            env_str(
                "REFLEXIO_PUBLISH_LEARNING_WORKERS",
                str(operation_limit_value("publish")),
            )
        ),
    )


def _reserve() -> bool:
    global _budget
    with _budget_lock:
        if _budget is None:
            _budget = threading.BoundedSemaphore(worker_count())
    return _budget.acquire(blocking=False)


def _release() -> None:
    if _budget is not None:
        _budget.release()


class DurableLearningWorker:
    def __init__(
        self,
        request_context_factory: Callable[[str], RequestContext],
        *,
        instance_id: str | None = None,
    ):
        self._factory = request_context_factory
        self._instance_id = instance_id or uuid.uuid4().hex

    def start_org(self, org_id: str, lease_seconds: int) -> bool:
        """Reserve capacity BEFORE a thread may claim one user."""
        if not _reserve():
            return False
        threading.Thread(
            target=self._reserved_turn,
            args=(org_id, lease_seconds),
            daemon=True,
            name="reflexio-extraction-window",
        ).start()
        return True

    def drain_org(self, org_id: str, batch_size: int, lease_seconds: int) -> int:
        """Synchronous harness entry point; each turn claims one runnable user."""
        completed = 0
        for _ in range(batch_size):
            if not _reserve():
                break
            completed += self._reserved_turn(org_id, lease_seconds)
        return completed

    def _reserved_turn(self, org_id: str, lease_seconds: int) -> int:
        try:
            return self._turn(org_id, lease_seconds)
        except Exception:
            logger.exception("Extraction turn failed org_id=%s", org_id)
            return 0
        finally:
            _release()

    def _turn(self, org_id: str, lease_seconds: int) -> int:
        from reflexio.lib._base import create_generation_litellm_client

        context = self._factory(org_id)
        storage = context.storage
        if storage is None:
            return 0
        claim = storage.claim_extraction(self._instance_id, lease_seconds)
        if claim is None:
            return 0
        user_id, token = claim
        stop = threading.Event()

        def heartbeat() -> None:
            while not stop.wait(max(0.1, lease_seconds / 3)):
                try:
                    if not storage.renew_extraction(user_id, token, lease_seconds):
                        return
                except Exception:
                    logger.exception("Extraction heartbeat failed org_id=%s", org_id)

        thread = threading.Thread(
            target=heartbeat, daemon=True, name="reflexio-extraction-heartbeat"
        )
        thread.start()
        window: Window | None = None
        try:
            for effect_window, effects in storage.pending_extraction_effects(
                user_id=user_id, token=token
            ):
                with bind_work_scope(
                    WorkScope(
                        org_id=org_id, project_id=effect_window.project_id or None
                    )
                ):
                    try:
                        WindowExecutor(
                            context, create_generation_litellm_client(context)
                        ).deliver(effect_window, effects, token=token)
                    except Exception as exc:
                        storage.retry_extraction_effects(effect_window)
                        # Log-only, unlike the window handler below: the effects
                        # retry has no error column to widen -
                        # ``retry_extraction_effects`` stores only a due time.
                        logger.warning(
                            "Extraction effects will retry org_id=%s error=%s at=%s",
                            org_id,
                            type(exc).__name__,
                            exception_locator(exc),
                        )
                return 1
            window = storage.prepare_extraction(user_id, token)
            if window is None:
                return 0
            with bind_work_scope(
                WorkScope(org_id=org_id, project_id=window.project_id or None)
            ):
                executor = WindowExecutor(
                    context, create_generation_litellm_client(context)
                )
                try:
                    executor.execute(window, token)
                except WindowInputsDeletedError:
                    storage.invalidate_extraction(window, token)
                except LeaseLostError:
                    return 0
                except Exception as exc:
                    where = exception_locator(exc)
                    logger.warning(
                        "Extraction window failed kind=%s error=%s at=%s",
                        window.kind,
                        type(exc).__name__,
                        where,
                    )
                    # The stored error stays a bounded class name plus our own
                    # source coordinates, never customer content: no ``exc``,
                    # ``str(exc)``, ``exc.args`` or ``exc_info``. An exception
                    # message on this path can quote the customer's data.
                    # If commit already succeeded only the outbox remains; it is
                    # independently retried without reopening cursor coverage.
                    try:
                        storage.retry_extraction(
                            window, token, f"{type(exc).__name__}@{where}"
                        )
                    except LeaseLostError:
                        storage.retry_extraction_effects(window)
                    return 0
            return 1
        except WorkScopeError:
            from reflexio.server.error_reporting import capture_anomaly

            capture_anomaly(
                "durable_learning.work_scope_failed",
                level="error",
                org_id=org_id,
                project_id=window.project_id if window else None,
                user_id=user_id,
            )
            storage.defer_extraction_setup(user_id, token)
            return 0
        except Exception as exc:
            logger.warning(
                "Extraction setup will retry org_id=%s error=%s at=%s",
                org_id,
                type(exc).__name__,
                exception_locator(exc),
            )
            storage.defer_extraction_setup(user_id, token)
            return 0
        finally:
            try:
                storage.release_user_extraction(user_id, token)
            finally:
                stop.set()
                thread.join(timeout=1)
