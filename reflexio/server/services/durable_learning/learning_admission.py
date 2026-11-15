"""Learning-only admission under the existing user-work and storage owners.

The transition row is a short transactional fence, not a model-held lock or
source authority. A seal can stop subsequent dispatch/publication while an
already returned response and its usage remain writable through their normal
run owner. Search, intake and financial acknowledgment do not enter this scope.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from reflexio.server.services.storage.storage_base.learning_transition import (
    LearningTransitionAdmission,
    LearningTransitionBlockedError,
    LearningTransitionIntegrityError,
)
from reflexio.server.work_scope import current_project_id

if TYPE_CHECKING:
    from reflexio.server.services.storage.storage_base import (
        AgentRunRecord,
        BaseStorage,
    )


@dataclass(frozen=True)
class LearningWork:
    """Actual root being executed, never a synthetic run or a generic grant."""

    org_id: str
    project_id: str
    user_id: str
    kind: Literal["request", "run", "window", "tagging"]
    reference: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.org_id, str)
            or not self.org_id
            or not isinstance(self.user_id, str)
            or not self.user_id.strip()
            or not isinstance(self.project_id, str)
            or not isinstance(self.kind, str)
            or self.kind not in {"request", "run", "window", "tagging"}
            or not isinstance(self.reference, str)
            or not self.reference
        ):
            raise LearningTransitionIntegrityError("Learning work identity unavailable")


@dataclass(frozen=True)
class _AdmittedWork:
    storage: BaseStorage
    work: LearningWork
    state: LearningTransitionAdmission
    # Set only by the selected-work executor after revalidating its full preview
    # under the real user lease. An arbitrary known digest is not a work scope.
    selection_sha256: str | None = None


_work: ContextVar[_AdmittedWork | None] = ContextVar("learning_work", default=None)


def current_learning_work() -> LearningWork | None:
    active = _work.get()
    return active.work if active is not None else None


def learning_transition_is_held(error: BaseException) -> bool:
    """Recognize explicit wrappers without inspecting content-bearing text."""
    from reflexio.server.llm.compute_admission import ComputeTransitionInterruptedError
    from reflexio.server.llm.failure import exception_chain

    return any(
        isinstance(
            link, (LearningTransitionBlockedError, ComputeTransitionInterruptedError)
        )
        for link in exception_chain(error, include_context=False)
    )


def _check_scope(storage: BaseStorage, work: LearningWork) -> None:
    from reflexio.server.services.durable_learning.user_lease import (
        _require_user_work_owner,
    )

    owner = _require_user_work_owner(storage)
    if (
        storage.org_id != work.org_id
        or owner.user_id != work.user_id
        or (current_project_id() or "") != work.project_id
    ):
        raise LearningTransitionIntegrityError("Learning work crossed its owned scope")


@contextmanager
def learning_work_scope(storage: BaseStorage, work: LearningWork) -> Iterator[None]:
    """Bind a normal root after user ownership; no writer spans computation."""
    active = _work.get()
    _check_scope(storage, work)
    if active is not None:
        if active.storage is not storage or active.work != work:
            raise LearningTransitionIntegrityError("Learning work root changed")
        fence_learning_work(storage)
        yield
        return
    from reflexio.server.services.durable_learning.user_lease import (
        _require_user_work_owner,
    )

    with storage.commit_scope():
        _require_user_work_owner(storage).fence(storage)
        state = storage.fence_learning_transition()
    token = _work.set(_AdmittedWork(storage, work, state))
    try:
        yield
    finally:
        _work.reset(token)


def fence_learning_work(storage: BaseStorage | None = None) -> None:
    """Check immediately before compute, or join a business writer transaction.

    Copies of the context share the original immutable admitted revision. A
    seal/reopen or changed grant cannot silently revive an abandoned worker.
    The caller must take this fence before downstream run/source/output locks.
    """
    active = _work.get()
    if active is None:
        if storage is not None:
            raise LearningTransitionIntegrityError("Learning work is not bound")
        return  # Ordinary query embeddings and unrelated compute remain available.
    if storage is not None and active.storage is not storage:
        raise LearningTransitionIntegrityError("Learning work storage changed")
    storage = active.storage
    _check_scope(storage, active.work)
    from reflexio.server.services.durable_learning.user_lease import (
        _require_user_work_owner,
    )

    with storage.commit_scope():
        _require_user_work_owner(storage).fence(storage)
        state = storage.fence_learning_transition(
            selection_sha256=active.selection_sha256
        )
        if state != active.state:
            raise LearningTransitionBlockedError("Learning admission revision changed")


def _refuse_effects_only_learning() -> None:
    from reflexio.server.services.durable_learning.selected_learning import (
        current_selected_learning,
    )

    selected = current_selected_learning()
    if selected is not None and selected.effects_only:
        raise LearningTransitionBlockedError(
            "Selected effects delivery cannot author learning"
        )


def fence_learning_compute() -> None:
    """A learning provider call must not inherit any retained writer locks."""
    _refuse_effects_only_learning()
    active = _work.get()
    if active is not None and active.storage.in_commit_scope:
        raise LearningTransitionIntegrityError(
            "Learning compute entered a writer scope"
        )
    fence_learning_work()


@contextmanager
def learning_business_write(storage: BaseStorage) -> Iterator[None]:
    """Serialize transition and business changes in the SAME existing writer."""
    _refuse_effects_only_learning()
    with storage.commit_scope():
        fence_learning_work(storage)
        yield


def require_learning_run(storage: BaseStorage, run: AgentRunRecord) -> None:
    """A selected or normal root cannot be borrowed by another logical run."""
    active = _work.get()
    if active is None:
        raise LearningTransitionIntegrityError("Learning execution is not bound")
    work = active.work
    missing_legacy_owner = run.binding.user_id is None or (
        isinstance(run.binding.user_id, str) and not run.binding.user_id.strip()
    )
    legacy_playbook = (
        missing_legacy_owner
        and run.binding.extractor_kind == "playbook"
        and run.created_at is not None
    )
    if (
        active.storage is not storage
        or run.binding.org_id != work.org_id
        or (run.binding.user_id != work.user_id and not legacy_playbook)
        # Project is storage-stamped, read-side only. A genuinely new run has
        # neither creation time nor a persisted project to compare yet.
        or (run.created_at is not None and (run.project_id or "") != work.project_id)
    ):
        raise LearningTransitionIntegrityError("Learning run crossed its root scope")
    matches = (
        (work.kind == "run" and work.reference == run.id)
        or (work.kind == "request" and work.reference == run.binding.request_id)
        or (work.kind == "window" and run.id == f"window:{work.reference}")
    )
    if not matches:
        raise LearningTransitionIntegrityError("Learning run differs from its root")
    with storage.commit_scope():
        # Preliminary admission uses complete ordinary reads under W/root.
        # Actual adoption retains its unchanged W -> R -> I locked proof and
        # persists the owner before any model/publication work can enter.
        fence_learning_work(storage)
        if legacy_playbook:
            from reflexio.server.services.playbook.review_window import (
                PlaybookReviewWindowError,
                infer_playbook_review_user_id,
            )

            try:
                proven_user = infer_playbook_review_user_id(
                    storage=storage,
                    source_interaction_ids=run.binding.source_interaction_ids,
                    subject="Legacy learning run admission",
                )
            except PlaybookReviewWindowError as error:
                raise LearningTransitionIntegrityError(
                    "Legacy playbook run owner is not source-proven"
                ) from error
            if proven_user != work.user_id:
                raise LearningTransitionIntegrityError(
                    "Legacy playbook run crossed its source-proven user"
                )


# One supported public surface for the operational caller; implementation stays
# beside normal worker selection rather than in an Enterprise/private callback.
from reflexio.server.services.durable_learning.selected_learning import (  # noqa: E402
    LearningWorkSelection as LearningWorkSelection,
)
from reflexio.server.services.durable_learning.selected_learning import (
    SelectedLearningResult as SelectedLearningResult,
)
from reflexio.server.services.durable_learning.selected_learning import (
    preview_learning_work as preview_learning_work,
)
from reflexio.server.services.durable_learning.selected_learning import (
    run_selected_learning_work as run_selected_learning_work,
)
