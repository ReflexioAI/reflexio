# ruff: noqa: S608 -- Fixed tables in disposable isolated contract fixtures only.
"""Synthetic fixture qualification; never a production provisioning authority."""

import json
import uuid
from typing import Any

from reflexio.profile_authoring_protocol import (
    PROFILE_AUTHORING_POLICY,
)
from reflexio.server.services.storage.storage_base.learning_transition import (
    LearningPolicyAuthority,
    _change_learning_policy_authority_db,
)


def qualify_empty_learning_scope(storage: Any) -> None:
    with storage._learning_policy_authority_scope() as (db, root):
        if root is None:
            for table in (
                "requests",
                "interactions",
                "_agent_runs",
                "extraction_windows",
                "extraction_cursors",
                "_agent_output_preparations",
                "_agent_output_receipts",
                "learning_work",
            ):
                if db.query(f"SELECT 1 FROM {db.table(table)} LIMIT 1"):
                    raise AssertionError("Fixture learning scope is not known-empty")
            root = LearningPolicyAuthority(
                1,
                uuid.uuid4().hex,
                0,
                "sealed",
                PROFILE_AUTHORING_POLICY,
                PROFILE_AUTHORING_POLICY,
                "synthetic-fixture",
                "qualified_upgrade",
                uuid.uuid4().hex,
                "f" * 64,
                True,
            )
            db.write_learning_policy_authority(json.dumps(root.to_dict()), missing=True)
            for action in ("arm", "activate"):
                root = _change_learning_policy_authority_db(
                    db,
                    action=action,
                    operation_id="synthetic-fixture",
                    expected_nonce=root.authority_nonce,
                    expected_generation=root.generation,
                )
        if root.phase != "open" or root.active_policy != PROFILE_AUTHORING_POLICY:
            raise AssertionError("Fixture root is not current OPEN policy")
    storage.provision_learning_scope()
