"""One exact existing root through the normal worker, never a drain executor.

Preview is content-free evidence. The actual user owner must revalidate it and
the operator's exact granted transition revision before ordinary selection may
adopt anything. Unknown historical generation instructions remain held: this
surface does not reinterpret finalization recovery as a generation protocol.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Literal

from reflexio.server.services.storage.storage_base.learning_transition import (
    LearningPolicyAuthority,
    LearningTransitionAdmission,
    LearningTransitionBlockedError,
    LearningTransitionIntegrityError,
)
from reflexio.server.work_scope import current_project_id

if TYPE_CHECKING:
    from reflexio.server.api_endpoints.request_context import RequestContext
    from reflexio.server.services.durable_learning.user_lease import _UserWorkOwner
    from reflexio.server.services.storage.storage_base._extraction_stream import Window

_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


@dataclass(frozen=True)
class LearningWorkSelection:
    schema_version: int
    org_id: str
    project_id: str
    user_id: str
    kind: Literal["run", "window"]
    reference: str
    revision_fingerprint: str
    payload_fingerprint: str
    config_fingerprint: str
    source_fingerprint: str
    instruction_fingerprint: str
    blockers: tuple[str, ...] = ()
    effects_only: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or not isinstance(self.org_id, str)
            or not self.org_id
            or not isinstance(self.project_id, str)
            or not isinstance(self.user_id, str)
            or not self.user_id.strip()
            or not isinstance(self.kind, str)
            or self.kind not in {"run", "window"}
            or not isinstance(self.reference, str)
            or not self.reference
            or type(self.effects_only) is not bool
            or (self.effects_only and self.kind != "window")
            or type(self.blockers) is not tuple
            or any(
                not isinstance(item, str) or not re.fullmatch(r"[a-z0-9_]+", item)
                for item in self.blockers
            )
        ):
            raise LearningTransitionIntegrityError("Invalid selected learning work")
        for value in (
            self.revision_fingerprint,
            self.payload_fingerprint,
            self.config_fingerprint,
            self.source_fingerprint,
            self.instruction_fingerprint,
        ):
            if not isinstance(value, str) or not _SHA.fullmatch(value):
                raise LearningTransitionIntegrityError(
                    "Invalid selected learning witness"
                )

    @property
    def eligible(self) -> bool:
        return not self.blockers

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "blockers": list(self.blockers)}

    def fingerprint(self) -> str:
        return _digest(self.to_dict())

    @classmethod
    def from_dict(cls, value: Any) -> LearningWorkSelection:
        if (
            not isinstance(value, dict)
            or set(value) != set(cls.__dataclass_fields__)
            or not isinstance(value.get("blockers"), list)
        ):
            raise LearningTransitionIntegrityError("Invalid selected learning fields")
        return cls(**{**value, "blockers": tuple(value["blockers"])})


@dataclass(frozen=True)
class SelectedLearningResult:
    status: Literal["completed", "held"]
    selection_sha256: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def preview_learning_work(
    context: RequestContext,
    *,
    run_id: str | None = None,
    window_id: str | None = None,
    owner: _UserWorkOwner | None = None,
) -> LearningWorkSelection:
    """Inspect one scoped original root; no claim, initialization or inference."""
    storage = context.storage
    if storage is None or storage.org_id != context.org_id:
        raise LearningTransitionIntegrityError("Selected learning storage mismatch")
    if bool(run_id) == bool(window_id):
        raise LearningTransitionIntegrityError("Select exactly one existing root")
    project = current_project_id() or ""
    config_fingerprint = _digest(
        context.configurator.get_config().model_dump(mode="json")
    )
    if window_id:
        from reflexio.lib._base import create_generation_litellm_client
        from reflexio.server.llm.input_budget import InputBudgetConfigurationError
        from reflexio.server.services.durable_learning.profile_package_executor import (
            ProfilePackageWindowExecutor,
        )
        from reflexio.server.services.storage.storage_base._extraction_stream import (
            Window,
        )

        row, cursor = _existing_window_rows(context, window_id)
        window = Window.from_row(row)
        parent = storage.get_agent_run(f"window:{window_id}")
        if row["completed"]:
            return _preview_completed_effects(
                context, window, row, cursor, parent, config_fingerprint
            )
        blockers = ("original_generation_instructions_unqualified",)
        instructions: Any = None
        if (
            window.kind == "profile"
            and not window.invalidated
            and not row["completed"]
            and cursor is not None
            and cursor["completed_seq"] == window.predecessor
            and parent is not None
            and parent.status.value
            not in {"cancelled", "expired", "finalized", "finalized_pending_tool"}
        ):
            try:
                version = parent.generation_request_snapshot.get("overflow", {}).get(
                    "version"
                )
                if version == 2 and isinstance(
                    parent.generation_request_snapshot.get(
                        "profile_package_preparation"
                    ),
                    dict,
                ):
                    executor = ProfilePackageWindowExecutor(
                        context, create_generation_litellm_client(context)
                    )
                    service, request, config = executor.service(window)
                    # Existing v2 recipe and validators are unchanged.
                    coordinator = executor._coordinator(
                        window, "", service, request, config, preview_only=True
                    )
                    instructions = coordinator.preparation.snapshot
                    blockers = ()
                elif version == 1 and isinstance(
                    parent.generation_request_snapshot.get(
                        "ordinary_profile_preparation"
                    ),
                    dict,
                ):
                    from reflexio.server.services.durable_learning.ordinary_profile_preparation import (
                        validate_ordinary_profile_preparation,
                    )
                    from reflexio.server.services.durable_learning.window_executor import (
                        WindowExecutor,
                    )

                    ordinary = WindowExecutor(
                        context, create_generation_litellm_client(context)
                    )
                    ordinary._overflow_manifest(parent)
                    service, _, config = ordinary.service(window)
                    # Provider-free: ordinary _coordinator would create/adopt rows.
                    instructions = validate_ordinary_profile_preparation(
                        ordinary, window, service, config, parent
                    )
                    blockers = ()
            except (
                InputBudgetConfigurationError,
                OSError,
                ValueError,
                KeyError,
                TypeError,
            ):
                blockers = ("original_generation_contract_incompatible",)
        return LearningWorkSelection(
            1,
            context.org_id,
            project,
            window.user_id,
            "window",
            window_id,
            _digest(
                {
                    "window": row,
                    "cursor": cursor,
                    "parent": asdict(parent) if parent else None,
                }
            ),
            _digest(parent.generation_request_snapshot if parent else row),
            config_fingerprint,
            _digest(window.manifest),
            _digest(instructions),
            blockers,
        )
    run = storage.get_agent_run(run_id)
    if run is None or (run.project_id or "") != project:
        raise LearningTransitionBlockedError("Selected learning root unavailable")
    from reflexio.server.services.extraction.recovery import inspect_run

    report = inspect_run(context, run, owner=owner)
    user_id = run.binding.user_id
    if not user_id:
        # Native legacy-user inference remains the recovery owner's concern;
        # this operational selection does not invent a user binding.
        raise LearningTransitionBlockedError("Selected learning owner unavailable")
    return LearningWorkSelection(
        1,
        context.org_id,
        project,
        user_id,
        "run",
        run.id,
        report["fingerprint"],
        report["payload_fingerprint"],
        config_fingerprint,
        _digest(run.generation_request_snapshot.get("source_interaction_digests")),
        _digest(
            {
                "agent_context": run.agent_context_snapshot,
                "schema": run.generation_request_snapshot.get("output_schema_name"),
            }
        ),
        tuple(sorted(set(report["blockers"]))),
    )


def _existing_window_rows(
    context: RequestContext, window_id: str
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    storage = context.storage
    if storage is None:
        raise LearningTransitionIntegrityError("Selected learning storage unavailable")
    project = current_project_id() or ""
    with storage._stream_sql(read_only=True) as db:
        rows = db.query(
            f"SELECT r.* FROM {db.table('extraction_windows')} r "  # noqa: S608
            f"WHERE r.window_id=? AND r.org_id=? AND {db.project_filter()} LIMIT 1",
            (window_id, context.org_id, project),
        )
        if len(rows) != 1:
            raise LearningTransitionBlockedError("Selected learning root unavailable")
        row = rows[0]
        cursors = db.query(
            f"SELECT window_id,attempts,completed_seq FROM {db.table('extraction_cursors')} "  # noqa: S608
            "WHERE user_id=? AND kind=? AND project_id=? AND window_id=? LIMIT 1",
            (row["user_id"], row["kind"], project, window_id),
        )
    return row, cursors[0] if cursors else None


def _preview_completed_effects(
    context: RequestContext,
    window: Window,
    row: dict[str, Any],
    cursor: dict[str, Any] | None,
    parent: Any,
    config_fingerprint: str,
) -> LearningWorkSelection:
    """Inspect the existing native effects consumer, not a generation recipe.

    Completed window+effects is the immutable publication record for initial
    windows. Compatibility parents additionally retain their existing receipt.
    Source identity is preserved even when erasure removed its raw payload.
    """
    from reflexio.lib._base import create_generation_litellm_client
    from reflexio.server.services.durable_learning.window_codec import decode_plan
    from reflexio.server.services.durable_learning.window_executor import WindowExecutor
    from reflexio.server.services.storage.storage_base import AgentRunStatus
    from reflexio.server.services.storage.storage_base._extraction_stream import (
        WindowInputsDeletedError,
        decoded,
    )

    storage = context.storage
    if storage is None:
        raise LearningTransitionIntegrityError("Selected learning storage unavailable")
    blockers = []
    effects = decoded(row.get("effects"))
    receipt = None
    source = None
    finalizer = None
    try:
        if row["effects_done"] or window.kind not in {"profile", "playbook"}:
            raise ValueError("Completed effects unavailable")
        if (
            not isinstance(effects, dict)
            or type(effects.get("version")) is not int
            or effects["version"] != 1
            or type(effects.get("skipped")) is not bool
        ):
            raise ValueError("Unsupported native effects codec")
        if not window.manifest or any(
            not isinstance(item, dict)
            or type(item.get("interaction_id")) is not int
            or item["interaction_id"] <= 0
            or not isinstance(item.get("request_id"), str)
            or not item["request_id"]
            for item in window.manifest
        ):
            raise ValueError("Completed source identity unavailable")
        if len({item["interaction_id"] for item in window.manifest}) != len(
            window.manifest
        ):
            raise ValueError("Completed source identity repeated")
        run_ids = effects.get("extraction_run_ids", [])
        if run_ids:
            if window.invalidated:
                raise ValueError("Erased parent finalization unsupported")
            if run_ids != [f"window:{window.window_id}"] or parent is None:
                raise ValueError("Completed parent unavailable")
            if (
                parent.binding.org_id != context.org_id
                or (parent.project_id or "") != window.project_id
                or parent.binding.extractor_kind != window.kind
                or parent.binding.user_id not in (None, window.user_id)
                or set(parent.binding.source_interaction_ids)
                != {item["interaction_id"] for item in window.manifest}
            ):
                raise ValueError("Completed parent/source binding changed")
            if parent.status not in {
                AgentRunStatus.AGENT_COMPLETED,
                AgentRunStatus.FINALIZING,
                AgentRunStatus.FINALIZED,
                AgentRunStatus.FINALIZED_PENDING_TOOL,
                AgentRunStatus.CANCELLED,
                AgentRunStatus.EXPIRED,
            }:
                raise ValueError("Unsupported completed parent state")
            receipt = storage.get_agent_run_finalization_receipt(
                run_id=parent.id,
                entity_type="profile" if window.kind == "profile" else "user_playbook",
            )
            if (
                parent.status
                in {AgentRunStatus.AGENT_COMPLETED, AgentRunStatus.FINALIZING}
                and not parent.pending_tool_call_ids
                and receipt is None
            ):
                raise ValueError("Completed parent publication receipt unavailable")
        source = _completed_source_identity(context, window)
        executor = WindowExecutor(context, create_generation_litellm_client(context))
        try:
            service, _, config = executor.service(window)
        except WindowInputsDeletedError:
            # The existing deliver owner bills/finalizes then ACKs this explicit
            # erasure case. Never restore deleted inputs merely to ACK effects.
            finalizer = {"native_effects": 1, "source_erased": True}
        else:
            decode_plan(effects, window.kind, config, window.user_id, service)
            finalizer = {"native_effects": 1, "source_erased": False}
    except (ValueError, KeyError, TypeError):
        blockers.append("completed_window_effects_unqualified")
    return LearningWorkSelection(
        1,
        context.org_id,
        current_project_id() or "",
        window.user_id,
        "window",
        window.window_id,
        _digest(
            {
                "window": row,
                "cursor": cursor,
                "parent": asdict(parent) if parent else None,
                "publication_receipt": receipt,
            }
        ),
        _digest(effects),
        config_fingerprint,
        _digest({"manifest": window.manifest, "retained_source_identity": source}),
        _digest(finalizer),
        tuple(blockers),
        effects_only=True,
    )


def _completed_source_identity(
    context: RequestContext, window: Window
) -> dict[str, Any]:
    """Bounded identity-only reads; absent erased payloads grant no new work."""
    from reflexio.server.services.storage.retention_mixin import chunked

    storage = context.storage
    if storage is None:
        raise LearningTransitionIntegrityError("Selected learning storage unavailable")
    expected = {item["interaction_id"]: item["request_id"] for item in window.manifest}
    interactions, requests = [], []
    with storage._stream_sql(read_only=True) as db:
        for identifiers in chunked(list(expected)):
            placeholders = ",".join("?" for _ in identifiers)
            rows = db.query(
                f"SELECT i.interaction_id,i.user_id,i.request_id FROM {db.table('interactions')} i WHERE i.interaction_id IN ({placeholders}) AND {db.project_filter('i')}",  # noqa: S608 -- Fixed tables; placeholders only.
                tuple(identifiers) + (window.project_id,),
            )
            for row in rows:
                if (
                    row["user_id"] != window.user_id
                    or row["request_id"] != expected[row["interaction_id"]]
                ):
                    raise ValueError("Completed source identity changed")
            interactions.extend(rows)
        for identifiers in chunked(sorted(set(expected.values()))):
            placeholders = ",".join("?" for _ in identifiers)
            rows = db.query(
                f"SELECT r.request_id,r.user_id FROM {db.table('requests')} r WHERE r.request_id IN ({placeholders}) AND {db.project_filter()}",  # noqa: S608 -- Fixed tables; placeholders only.
                tuple(identifiers) + (window.project_id,),
            )
            if any(row["user_id"] != window.user_id for row in rows):
                raise ValueError("Completed request owner changed")
            requests.extend(rows)
    return {
        "interactions": sorted(interactions, key=lambda item: item["interaction_id"]),
        "requests": sorted(requests, key=lambda item: item["request_id"]),
    }


def selected_existing_effects(context: RequestContext) -> tuple[Window, dict[str, Any]]:
    """Revalidate this completed receipt under actual W; never prepare a window."""
    from reflexio.server.services.durable_learning.learning_admission import (
        fence_learning_work,
    )
    from reflexio.server.services.durable_learning.user_lease import (
        _require_user_work_owner,
    )
    from reflexio.server.services.storage.storage_base._extraction_stream import (
        Window,
        decoded,
    )

    request = _selected.get()
    storage = context.storage
    if request is None or not request.selection.effects_only or storage is None:
        raise LearningTransitionIntegrityError("No exact selected effects")
    with storage.commit_scope():
        _require_user_work_owner(storage).fence(storage)
        fence_learning_work(storage)
        validate_selected_learning(context)
        row, _cursor = _existing_window_rows(context, request.selection.reference)
        if not row["completed"] or row["effects_done"]:
            raise LearningTransitionBlockedError("Selected completed effects changed")
        return Window.from_row(row), decoded(row["effects"])


def selected_existing_window(context: RequestContext) -> Window:
    """Read only this admitted cursor/root under the normal actual user owner."""
    from dataclasses import replace

    from reflexio.server.services.durable_learning.learning_admission import (
        learning_business_write,
    )
    from reflexio.server.services.storage.storage_base._extraction_stream import Window

    request = _selected.get()
    storage = context.storage
    if request is None or request.selection.kind != "window" or storage is None:
        raise LearningTransitionIntegrityError("No exact selected window")
    with learning_business_write(storage):
        validate_selected_learning(context)
        row, cursor = _existing_window_rows(context, request.selection.reference)
        if cursor is None or row["completed"] or row["invalidated"]:
            raise LearningTransitionBlockedError("Selected window changed")
        window = replace(Window.from_row(row), attempts=int(cursor["attempts"] or 0))
        storage.validate_extraction_inputs(window)
        return window


@dataclass(frozen=True)
class _SelectedRequest:
    selection: LearningWorkSelection
    operation_id: str
    expected_nonce: str
    expected_revision: int
    expected_authority_nonce: str
    expected_authority_generation: int


_selected: ContextVar[_SelectedRequest | None] = ContextVar(
    "selected_learning_root", default=None
)


def current_selected_learning() -> LearningWorkSelection | None:
    value = _selected.get()
    return value.selection if value is not None else None


def selected_claim_matches_authority(
    authority: LearningPolicyAuthority, *, org_id: str, user_id: str
) -> bool:
    """A scheduling-only exception from the program-owned selected context.

    This does not validate source or grant execution. The normal actual user
    owner rechecks the precise project grant and original witness before work.
    """
    request = _selected.get()
    return bool(
        request is not None
        and request.selection.eligible
        and request.selection.org_id == org_id
        and request.selection.user_id == user_id
        and request.selection.project_id == (current_project_id() or "")
        and authority.phase == "sealed"
        and authority.operation_id == request.operation_id
        and authority.authority_nonce == request.expected_authority_nonce
        and authority.generation == request.expected_authority_generation
    )


def _require_selected_grant(
    state: LearningTransitionAdmission | None, request: _SelectedRequest
) -> None:
    if (
        state is None
        or state.mode != "sealed"
        or state.operation_id != request.operation_id
        or state.authority.authority_nonce != request.expected_authority_nonce
        or state.authority.generation != request.expected_authority_generation
        or state.project is None
        or state.project.authority_nonce != state.authority.authority_nonce
        or state.project.authority_generation != state.authority.generation
        or state.project.operation_id != request.operation_id
        or state.project.nonce != request.expected_nonce
        or state.project.revision != request.expected_revision
        or state.selection_sha256 != request.selection.fingerprint()
    ):
        raise LearningTransitionBlockedError("Selected learning grant changed")


def validate_selected_learning(
    context: RequestContext, *, owner: _UserWorkOwner | None = None
) -> LearningWorkSelection:
    request = _selected.get()
    if request is None:
        raise LearningTransitionIntegrityError("No selected learning root")
    selection = request.selection
    current = preview_learning_work(
        context,
        run_id=selection.reference if selection.kind == "run" else None,
        window_id=selection.reference if selection.kind == "window" else None,
        owner=owner,
    )
    if not current.eligible or current.fingerprint() != selection.fingerprint():
        raise LearningTransitionBlockedError("Selected learning witness changed")
    return selection


def claim_selected_user(
    context: RequestContext, *, worker_id: str, lease_seconds: int
) -> tuple[str, str] | None:
    """Normal preclaim validates the exact root; no class rotation is allowed."""
    selection = validate_selected_learning(context)
    storage = context.storage
    if storage is None:
        raise LearningTransitionIntegrityError("Selected learning storage unavailable")
    request = _selected.get()
    if request is None:
        raise LearningTransitionIntegrityError("No selected learning root")
    # Useful preclaim refusal, not authority: the real owner rechecks this same
    # revision in its locked writer before adopting any business work.
    _require_selected_grant(storage.read_learning_transition(), request)
    token = storage.claim_user_extraction(selection.user_id, worker_id, lease_seconds)
    return (selection.user_id, token) if token is not None else None


@contextmanager
def selected_learning_admission(context: RequestContext) -> Iterator[None]:
    """Validate after actual user acquisition, before normal run adoption."""
    request = _selected.get()
    if request is None:
        yield
        return
    storage = context.storage
    if storage is None:
        raise LearningTransitionIntegrityError("Selected learning storage unavailable")
    from reflexio.server.services.durable_learning.learning_admission import (
        LearningWork,
        _AdmittedWork,
        _work,
    )
    from reflexio.server.services.durable_learning.user_lease import (
        _require_user_work_owner,
    )

    owner = _require_user_work_owner(storage)
    with storage.commit_scope():
        owner.fence(storage)
        state = storage.fence_learning_transition(
            selection_sha256=request.selection.fingerprint()
        )
        _require_selected_grant(state, request)
        selection = validate_selected_learning(context, owner=owner)
    token = _work.set(
        _AdmittedWork(
            storage,
            LearningWork(
                selection.org_id,
                selection.project_id,
                selection.user_id,
                selection.kind,
                selection.reference,
            ),
            state,
            selection.fingerprint(),
        )
    )
    try:
        yield
    finally:
        _work.reset(token)


def run_selected_learning_work(
    context: RequestContext,
    selection: LearningWorkSelection,
    *,
    operation_id: str,
    expected_nonce: str,
    expected_revision: int,
    expected_authority_nonce: str,
    expected_authority_generation: int,
) -> SelectedLearningResult:
    """One bounded normal worker turn; no unrelated class or fresh-work rotation."""
    from reflexio.server.services.durable_learning.worker import DurableLearningWorker
    from reflexio.server.work_scope import WorkScope, bind_work_scope

    if (
        not isinstance(selection, LearningWorkSelection)
        or selection.org_id != context.org_id
        or not isinstance(operation_id, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", operation_id)
        or not isinstance(expected_nonce, str)
        or not re.fullmatch(r"[0-9a-f]{32}", expected_nonce)
        or type(expected_revision) is not int
        or expected_revision < 0
        or not isinstance(expected_authority_nonce, str)
        or not re.fullmatch(r"[0-9a-f]{32}", expected_authority_nonce)
        or type(expected_authority_generation) is not int
        or expected_authority_generation < 0
    ):
        raise LearningTransitionIntegrityError("Invalid selected learning invocation")
    request = _SelectedRequest(
        selection,
        operation_id,
        expected_nonce,
        expected_revision,
        expected_authority_nonce,
        expected_authority_generation,
    )
    with bind_work_scope(WorkScope(selection.org_id, selection.project_id or None)):
        token = _selected.set(request)
        try:
            if not selection.eligible:
                return SelectedLearningResult(
                    "held", selection.fingerprint(), "original_witness_unqualified"
                )
            validate_selected_learning(context)
            return DurableLearningWorker(lambda _org: context).run_selected_turn(
                context
            )
        except LearningTransitionBlockedError:
            return SelectedLearningResult(
                "held", selection.fingerprint(), "selection_or_transition_changed"
            )
        finally:
            _selected.reset(token)
