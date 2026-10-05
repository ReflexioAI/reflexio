"""A dropped connection in the vector backfill is a retried WARNING, not a page."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

from reflexio.server import background_work
from reflexio.server.services.lineage import vector_backfill_sweep as vbs


def test_transient_failure_warns_keeps_the_anomaly_and_clears_on_success(
    monkeypatch, caplog, transient_failure_classifier
) -> None:
    monkeypatch.setenv("REFLEXIO_MISSING_VECTOR_BACKFILL_ENABLED", "true")
    ctx = MagicMock()
    ctx.storage.backfill_missing_interaction_vectors.side_effect = (
        transient_failure_classifier("SSL connection has been closed unexpectedly")
    )
    monkeypatch.setattr(
        "reflexio.server.api_endpoints.request_context.RequestContext",
        lambda **_kwargs: ctx,
    )
    anomalies: list[str] = []
    monkeypatch.setattr(
        "reflexio.server.error_reporting.capture_anomaly",
        lambda name, **_fields: anomalies.append(name),
    )
    caplog.set_level(logging.WARNING, logger=vbs.logger.name)

    assert vbs.missing_vector_backfill_sweep("org-1", 0) == 0

    (record,) = [r for r in caplog.records if "backfill_org_failed" in r.getMessage()]
    assert record.levelno == logging.WARNING
    assert anomalies == ["interactions.missing_vector_backfill.run_failed"]
    assert set(background_work._streaks) == {"missing-vector-backfill:org-1"}

    ctx.storage.backfill_missing_interaction_vectors.side_effect = None
    ctx.storage.backfill_missing_interaction_vectors.return_value = 3
    assert vbs.missing_vector_backfill_sweep("org-1", 0) == 3
    assert background_work._streaks == {}
