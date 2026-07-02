"""Declarative deployment profile (Tier-3 Phase C).

A ``DeploymentProfile`` is a minimal, frozen description of *what a running app
serves*: its deployment ``role``, the set of ``router_groups`` it mounts, and
whether auth is required. It is a cleaner internal representation of the knobs
``create_app`` already accepts (``mount_data_plane`` / ``require_auth``) — the
profile changes nothing observable; it just gives the composition root a single
value to thread instead of scattered booleans.

This module is an OSS public seam: it imports nothing from ``reflexio_ext`` and
carries no enterprise backend knowledge (string/stdlib only). The enterprise
composition defines its own profiles (self-host / platform) on top of this type.

``mounts_data_plane`` is DERIVED from ``router_groups`` (membership of the
``DATA_PLANE_GROUP``), not stored — so the two can never disagree.
"""

from __future__ import annotations

from dataclasses import dataclass

# The single OSS router group: the data-plane surface (core product endpoints,
# stall-state, pending-tool-call) that ``create_app`` mounts when data-plane is
# active. Enterprise adds its own group names on top of this type.
DATA_PLANE_GROUP = "data-plane"


@dataclass(frozen=True)
class DeploymentProfile:
    """What a running app serves: role, mounted router groups, auth requirement.

    Attributes:
        role (str): Deployment role label (e.g. ``"all"``, ``"data-plane"``,
            ``"control-plane"``). Informational for the OSS factory; enterprise
            uses it to select its router set.
        router_groups (frozenset[str]): Router-group names this app mounts.
        auth_required (bool): Whether the app declares/enforces auth.
    """

    role: str
    router_groups: frozenset[str]
    auth_required: bool

    @property
    def mounts_data_plane(self) -> bool:
        """Whether this profile mounts the data-plane router group."""
        return DATA_PLANE_GROUP in self.router_groups


def resolve_profile(*, mount_data_plane: bool, require_auth: bool) -> DeploymentProfile:
    """Derive the OSS profile from the legacy ``create_app`` knobs.

    Behaviour-preserving: this is the identity mapping from the two booleans
    ``create_app`` has always accepted onto the profile representation.

    Args:
        mount_data_plane (bool): Whether the data-plane routers/lifespan run.
        require_auth (bool): Whether auth (Bearer security scheme) is declared.

    Returns:
        DeploymentProfile: ``role="all"`` + the data-plane group when
        ``mount_data_plane`` is True, otherwise ``role="control-plane"`` with no
        router groups. ``auth_required`` mirrors ``require_auth``.
    """
    router_groups = frozenset({DATA_PLANE_GROUP}) if mount_data_plane else frozenset()
    role = "all" if mount_data_plane else "control-plane"
    return DeploymentProfile(
        role=role, router_groups=router_groups, auth_required=require_auth
    )
