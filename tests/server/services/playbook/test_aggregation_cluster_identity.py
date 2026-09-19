"""The aggregation cluster id must be distinct per project, and inert in OSS.

``cluster_id`` was ``uuid5(org_id : agent_version : fingerprint)``. Two projects
in one org clustering the same fingerprint at the same agent version therefore
computed the SAME uuid and collided on ``playbook_aggregation_cluster``'s
org-wide primary key, which RLS hides from the second project — so its insert
failed against a row it could not see.

The project is now part of the identity rather than part of a composite key.
That needs no expand/contract sequence and no rebuild of
``playbook_aggregation_item``'s foreign key, which references the org-wide
column; the rows are distinct by construction instead of by constraint.

These tests pin the two properties that decision rests on.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from typing import TYPE_CHECKING

import pytest

from reflexio.server.extensions import register_service, reset_services
from reflexio.server.services.playbook.components.aggregator import (
    PlaybookAggregator,
)
from reflexio.server.work_scope import WORK_SCOPE_PROVIDER, WorkScope

if TYPE_CHECKING:
    from collections.abc import Iterator


class _StubContext:
    """The only attribute the derivation reads."""

    def __init__(self, org_id: str) -> None:
        self.org_id = org_id


class _StubScopeProvider:
    """Stands in for the enterprise provider, which OSS does not register."""

    def __init__(self, project_id: str | None) -> None:
        self._scope = WorkScope(org_id="org_1", project_id=project_id)

    def current(self) -> WorkScope | None:
        return self._scope

    @contextmanager
    def bind(self, scope: WorkScope) -> Iterator[None]:  # pragma: no cover
        previous, self._scope = self._scope, scope
        try:
            yield
        finally:
            self._scope = previous


def _cluster_id(org_id: str, agent_version: str, fingerprint: str) -> str:
    aggregator = PlaybookAggregator.__new__(PlaybookAggregator)
    aggregator.request_context = _StubContext(org_id)  # type: ignore[assignment]
    aggregator.agent_version = agent_version  # type: ignore[assignment]
    return aggregator._stable_aggregation_cluster_id(fingerprint)  # noqa: SLF001


@pytest.fixture(autouse=True)
def _clean_services() -> Iterator[None]:
    reset_services()
    yield
    reset_services()


def test_two_projects_with_one_fingerprint_get_different_ids() -> None:
    """The collision this change exists to remove."""
    register_service(WORK_SCOPE_PROVIDER, _StubScopeProvider("prj_a"))
    first = _cluster_id("org_1", "agent-v0", "fp")
    reset_services()
    register_service(WORK_SCOPE_PROVIDER, _StubScopeProvider("prj_b"))
    second = _cluster_id("org_1", "agent-v0", "fp")
    assert first != second


def test_the_same_project_is_still_stable() -> None:
    """ "Stable" is the whole contract of the name; distinctness must not cost it."""
    register_service(WORK_SCOPE_PROVIDER, _StubScopeProvider("prj_a"))
    assert _cluster_id("org_1", "agent-v0", "fp") == _cluster_id(
        "org_1", "agent-v0", "fp"
    )


def test_oss_ids_are_unchanged_when_no_provider_is_registered() -> None:
    """OSS has no projects, so it must derive exactly what it derived before.

    The expected value is the pre-change formula computed independently here —
    not a call into the function under test, which would pass no matter what
    the function did.
    """
    expected = str(uuid.uuid5(uuid.NAMESPACE_URL, "org_1:agent-v0:fp"))
    assert _cluster_id("org_1", "agent-v0", "fp") == expected


def test_an_unbound_project_is_treated_as_no_project() -> None:
    """A provider handing back an empty project must not create a third id
    space distinct from both "no project" and a real one."""
    register_service(WORK_SCOPE_PROVIDER, _StubScopeProvider(""))
    bound_empty = _cluster_id("org_1", "agent-v0", "fp")
    reset_services()
    assert bound_empty == _cluster_id("org_1", "agent-v0", "fp")


def test_the_scope_segment_cannot_be_forged_by_an_org_id() -> None:
    """``org:project`` is joined with a colon, and both halves are unconstrained
    strings. Pin the ambiguity so that if it is ever fixed by a different
    separator or by hashing the parts, the change is deliberate."""
    register_service(WORK_SCOPE_PROVIDER, _StubScopeProvider("b"))
    colliding = _cluster_id("a", "agent-v0", "fp")
    reset_services()
    register_service(WORK_SCOPE_PROVIDER, _StubScopeProvider(None))
    naive = _cluster_id("a:b", "agent-v0", "fp")
    assert colliding == naive, (
        "org 'a' + project 'b' and org 'a:b' with no project still render to "
        "the same scope string. That is today's behaviour, recorded rather "
        "than claimed as safe: an org id containing a colon is the same "
        "ambiguity AggregationScheduler._repair_scope_key avoided by keying on "
        "a tuple. Fixing it here means re-deriving every id again, so it is "
        "deliberately not bundled with this change."
    )


def test_agent_version_still_separates_clusters() -> None:
    register_service(WORK_SCOPE_PROVIDER, _StubScopeProvider("prj_a"))
    assert _cluster_id("org_1", "agent-v0", "fp") != _cluster_id(
        "org_1", "agent-v1", "fp"
    )
