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
from reflexio.server.services.playbook.publication import canonical_json_bytes
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


def test_a_colon_in_an_id_cannot_forge_another_tenant() -> None:
    """The ambiguity a delimiter-joined scope could not avoid.

    org "a" + project "b" and org "a:b" + project "c" are different tenants.
    Joined with a colon they render "a:b" and "a:b:c" — and org "a" with
    project "b:c" renders "a:b:c" too, so that pair collides with the second.
    Both reviewers on reflexio#517 named it; an earlier revision of this file
    pinned it as accepted, on the reasoning that fixing it meant re-deriving
    every id a second time. That reasoning was wrong: this change already
    re-derives every enterprise id, so the fix rides along for free and the ids
    move once rather than twice.
    """
    register_service(WORK_SCOPE_PROVIDER, _StubScopeProvider("b:c"))
    left = _cluster_id("a", "agent-v0", "fp")
    reset_services()
    register_service(WORK_SCOPE_PROVIDER, _StubScopeProvider("c"))
    right = _cluster_id("a:b", "agent-v0", "fp")
    assert left != right, (
        'org "a" + project "b:c" and org "a:b" + project "c" are different '
        "tenants and must not share a cluster id. A delimiter-joined scope "
        "renders both as a:b:c; canonical JSON escapes the separator instead."
    )


def test_a_quote_in_an_id_cannot_forge_another_tenant() -> None:
    """The same property one level down: JSON's own delimiters are escaped.

    Swapping a colon join for a JSON one would be no fix at all if a quote or
    backslash in an id could close the string and re-open it.
    """
    register_service(WORK_SCOPE_PROVIDER, _StubScopeProvider('b","c'))
    left = _cluster_id("a", "agent-v0", "fp")
    reset_services()
    register_service(WORK_SCOPE_PROVIDER, _StubScopeProvider("c"))
    right = _cluster_id('a","b', "agent-v0", "fp")
    assert left != right


def test_the_scoped_payload_shape_is_pinned() -> None:
    """Pin the exact bytes hashed, computed independently of the code.

    An earlier revision asserted only that scoped and unscoped ids differ,
    which is true with or without the scheme tag — so deleting the tag passed.
    A guard that cannot fail for the thing it guards is the failure this file
    has hit twice, so the payload is reconstructed here rather than described.

    The tag is a VERSION marker: bumping it re-derives every scoped cluster in
    one line. That is the property worth pinning, and pinning it means the tag
    cannot be dropped silently.
    """
    register_service(WORK_SCOPE_PROVIDER, _StubScopeProvider("prj_a"))
    expected = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            canonical_json_bytes(
                [
                    "playbook-aggregation-cluster:v2",
                    "org_1",
                    "prj_a",
                    "agent-v0",
                    "fp",
                ]
            ).decode("utf-8"),
        )
    )
    assert _cluster_id("org_1", "agent-v0", "fp") == expected


def test_the_scoped_and_unscoped_id_spaces_stay_disjoint() -> None:
    """An unscoped id hashes a bare string, a scoped one a tagged JSON array."""
    unscoped = _cluster_id("org_1", "agent-v0", "fp")
    register_service(WORK_SCOPE_PROVIDER, _StubScopeProvider("prj_a"))
    scoped = _cluster_id("org_1", "agent-v0", "fp")
    assert unscoped != scoped


def test_agent_version_still_separates_clusters() -> None:
    register_service(WORK_SCOPE_PROVIDER, _StubScopeProvider("prj_a"))
    assert _cluster_id("org_1", "agent-v0", "fp") != _cluster_id(
        "org_1", "agent-v1", "fp"
    )
