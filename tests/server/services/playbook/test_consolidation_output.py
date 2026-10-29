"""The single method cannot accept a legacy four-action proposal as approval."""

import pytest
from pydantic import ValidationError

from reflexio.server.services.playbook.components.reconciliation import Decisions


@pytest.mark.parametrize(
    "kind", ["unify", "reject_new", "differentiate", "independent"]
)
def test_retired_four_action_proposal_cannot_become_audited_decisions(kind):
    with pytest.raises(ValidationError):
        Decisions.model_validate({"decisions": [{"kind": kind, "new_id": "NEW-0"}]})
