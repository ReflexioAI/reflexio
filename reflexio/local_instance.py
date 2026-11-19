"""Organization selection shared by the OSS CLI and local server."""

from __future__ import annotations

import os

DEFAULT_ORG_ID = "self-host-org"


def configured_local_org_id() -> str:
    """Resolve the local launcher's namespace for config and storage."""
    return os.environ.get("REFLEXIO_DEFAULT_ORG_ID", "").strip() or DEFAULT_ORG_ID
