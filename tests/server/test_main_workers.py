"""Tests for reflexio.server.__main__ CLI argument parsing and uvicorn dispatch."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from reflexio.server.__main__ import main


def test_dev_mode_passes_reload_true() -> None:
    with patch("reflexio.server.__main__.uvicorn.run") as run:
        main(["--port", "8081", "--reload"])
    kwargs = run.call_args.kwargs
    assert kwargs["reload"] is True
    # Dev mode must not pass workers > 1.
    assert kwargs.get("workers", 1) == 1


def test_daemon_mode_passes_workers_and_max_requests() -> None:
    with patch("reflexio.server.__main__.uvicorn.run") as run:
        main(
            [
                "--port",
                "8081",
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
    kwargs = run.call_args.kwargs
    assert kwargs["reload"] is False
    assert kwargs["workers"] == 2
    assert kwargs["limit_max_requests"] == 5000
    assert kwargs["limit_max_requests_jitter"] == 500
    assert kwargs["timeout_graceful_shutdown"] == 20


def test_reload_with_workers_gt_1_rejected() -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--port", "8081", "--reload", "--workers", "2"])
    assert exc_info.value.code != 0


def test_workers_zero_rejected() -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--port", "8081", "--workers", "0"])
    assert exc_info.value.code != 0


def test_max_requests_zero_disables_recycling() -> None:
    """--max-requests 0 must translate to limit_max_requests=None at uvicorn.

    uvicorn treats ``limit_max_requests=0`` as "shut down after 0 served requests"
    (i.e. recycle on the first request). The operator-facing contract for
    ``--max-requests 0`` is "disable recycling", which corresponds to uvicorn's
    None default. The dispatcher must translate.
    """
    with patch("reflexio.server.__main__.uvicorn.run") as run:
        main(["--port", "8081", "--workers", "2", "--max-requests", "0"])
    kwargs = run.call_args.kwargs
    assert kwargs["limit_max_requests"] is None
