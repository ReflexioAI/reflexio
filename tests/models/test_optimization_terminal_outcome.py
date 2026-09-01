"""``OptimizationTerminalOutcome`` must admit the outcomes the tuner writes.

'regeneration_fenced' reached the tenant CHECK in 20260827040000 (re-declared by
20260830020000) and is written by reflexio_ext
offline_tuner/open_world/runner.py:251 via ``_converge_terminal_failure``, but
was never added to this union -- so reading such a row back through
``_row_to_playbook_optimization_job`` raised ``ValidationError``.

That failure did not surface as a validation error either. ``handle_exceptions``
converts it to ``StorageError``, and the open-world runner's ``except
StorageError`` arm reports ``infrastructure_failure`` -- so a refusal the fence
made on purpose was recorded, and reported to the enterprise error monitor,
under a reason that names
a fault that never happened.

The union-versus-CHECK set equality is asserted in the enterprise tree, where
cross-repository assertions belong and ``supabase/`` actually exists:
``reflexio_ext/tests/server/services/offline_tuner/test_replay_removal_allowlists.py``.
This module stays inside the OSS package so it keeps passing in a standalone
checkout, where there is no ``supabase/`` directory to read.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from reflexio.models.api_schema.domain.entities import PlaybookOptimizationJob


def test_a_fenced_job_row_parses() -> None:
    """The regression: this construction raised ``ValidationError`` before."""
    job = PlaybookOptimizationJob(
        job_id=1,
        target_kind="user_playbook",
        target_id=7,
        status="failed",
        stage="failed",
        terminal_outcome="regeneration_fenced",
    )

    assert job.terminal_outcome == "regeneration_fenced"


def test_an_invented_outcome_is_still_rejected() -> None:
    """Widening the union must not degrade it into a free-form string."""
    with pytest.raises(ValidationError):
        PlaybookOptimizationJob(
            job_id=1,
            target_kind="user_playbook",
            target_id=7,
            terminal_outcome="not_a_real_outcome",  # type: ignore[arg-type]
        )
