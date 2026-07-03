"""Regression test for the batch-mode flag leak in ``_run_batch_with_progress``.

``_is_batch_mode`` is read by the base ``run`` to alter behavior. If the batch
setup (``initialize_progress``) raises, the flag must NOT be left ``True`` — a
leak would spuriously put the NEXT operation on the same service instance into
batch mode.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from reflexio.server.services.base_generation._batch_progress import (
    BatchProgressMixin,
)


class _MinimalBatchService(BatchProgressMixin[dict]):
    """Smallest concrete host for the batch driver under test."""

    def __init__(self) -> None:
        self._is_batch_mode = False

    def _get_base_service_name(self) -> str:
        return "minimal"


def test_batch_mode_flag_reset_when_initialize_progress_raises() -> None:
    svc = _MinimalBatchService()
    state_manager = MagicMock()
    state_manager.initialize_progress.side_effect = RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        svc._run_batch_with_progress(
            user_ids=["u1"],
            request={},
            request_params={},
            state_manager=state_manager,
        )

    assert svc._is_batch_mode is False
