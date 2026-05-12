"""Tests for run_services build_backend_service argument forwarding."""

from __future__ import annotations

from typing import Any

import pytest

from reflexio.cli.run_services import build_backend_service


def _cmd(reload: bool, **kwargs: Any) -> list[str]:
    """Helper: build the backend command and return the spawn argv.

    Args:
        reload (bool): Whether reload mode is enabled.
        **kwargs (Any): Additional kwargs to forward to build_backend_service.

    Returns:
        list[str]: The fully constructed spawn argv.
    """
    svc = build_backend_service({"backend": 8081}, reload=reload, **kwargs)
    return svc.command


def test_dev_mode_includes_reload_and_workers_1() -> None:
    # Dev mode requires workers=1 explicitly; the CLI dispatcher coerces this
    # automatically, but build_backend_service expects the resolved value.
    cmd = _cmd(reload=True, workers=1)
    assert "--reload" in cmd
    assert "--workers" in cmd
    assert cmd[cmd.index("--workers") + 1] == "1"


def test_daemon_mode_forwards_workers() -> None:
    cmd = _cmd(reload=False, workers=2)
    assert "--reload" not in cmd
    assert "--workers" in cmd
    assert cmd[cmd.index("--workers") + 1] == "2"


def test_daemon_mode_forwards_max_requests_and_jitter() -> None:
    cmd = _cmd(reload=False, workers=2, max_requests=5000, max_requests_jitter=500)
    assert "--max-requests" in cmd
    assert cmd[cmd.index("--max-requests") + 1] == "5000"
    assert "--max-requests-jitter" in cmd
    assert cmd[cmd.index("--max-requests-jitter") + 1] == "500"


def test_daemon_mode_forwards_graceful_shutdown_sec() -> None:
    cmd = _cmd(reload=False, workers=2, graceful_shutdown_sec=20)
    assert "--graceful-shutdown-sec" in cmd
    assert cmd[cmd.index("--graceful-shutdown-sec") + 1] == "20"


def test_daemon_mode_defaults_to_workers_2() -> None:
    """Audit (Task 9) + B1-B5 stress tests + F2/F3 remediation all passed; default is 2 workers."""
    cmd = _cmd(reload=False)
    assert "--workers" in cmd
    assert cmd[cmd.index("--workers") + 1] == "2"


def test_reload_plus_workers_gt_1_rejected() -> None:
    with pytest.raises(ValueError, match="incompatible with --reload"):
        _cmd(reload=True, workers=2)


def test_run_services_parser_accepts_new_flags() -> None:
    from reflexio.cli.run_services import _build_run_services_parser

    parser = _build_run_services_parser()
    args = parser.parse_args(
        [
            "--no-reload",
            "--workers",
            "2",
            "--max-requests",
            "5000",
            "--max-requests-jitter",
            "500",
            "--graceful-shutdown-sec",
            "20",
        ]
    )
    assert args.no_reload is True
    assert args.workers == 2
    assert args.max_requests == 5000
    assert args.max_requests_jitter == 500
    assert args.graceful_shutdown_sec == 20


def test_sqlite_plus_multi_worker_logs_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When storage backend is SQLite and workers > 1, log a warning."""
    import logging

    from reflexio.cli.run_services import _warn_if_sqlite_multi_worker

    with caplog.at_level(logging.WARNING):
        _warn_if_sqlite_multi_worker(storage_backend="sqlite", workers=2)
    assert any(
        "SQLite has limited concurrent write throughput" in rec.message
        for rec in caplog.records
    )


def test_sqlite_plus_single_worker_no_warning(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    from reflexio.cli.run_services import _warn_if_sqlite_multi_worker

    with caplog.at_level(logging.WARNING):
        _warn_if_sqlite_multi_worker(storage_backend="sqlite", workers=1)
    assert not any("SQLite has limited" in rec.message for rec in caplog.records)


def test_postgres_plus_multi_worker_no_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    from reflexio.cli.run_services import _warn_if_sqlite_multi_worker

    with caplog.at_level(logging.WARNING):
        _warn_if_sqlite_multi_worker(storage_backend="postgres", workers=2)
    assert not any("SQLite has limited" in rec.message for rec in caplog.records)
