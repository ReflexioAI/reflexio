"""Tests for service process utility helpers."""

from __future__ import annotations

import tempfile
from pathlib import Path

from reflexio.cli import utils


def test_pidfile_path_uses_platform_temp_dir() -> None:
    path = utils.get_pidfile_path({"backend": 8071})

    assert path.parent == Path(tempfile.gettempdir())
    assert str(path).startswith(tempfile.gettempdir())
    assert path.name.startswith("reflexio_services_")
