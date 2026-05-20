"""Canonical filesystem paths for reflexio runtime data.

Centralizes the resolution of the ``.reflexio`` base directory so every
caller honors the ``REFLEXIO_LOG_DIR`` env var consistently. When the env
var is set, the ``.reflexio`` tree is rooted at that path instead of
``Path.home()``; the ``.reflexio/<subdir>`` suffix is always preserved
so the on-disk layout stays identical regardless of where the base
points.
"""

from __future__ import annotations

import os
from pathlib import Path


def reflexio_home() -> Path:
    """Return the canonical ``.reflexio`` directory.

    Resolution:
        - If ``REFLEXIO_LOG_DIR`` is set, the base is that path (with
          ``~`` expanded; resolved to an absolute path, anchored to
          ``Path.home()`` if relative).
        - Otherwise the base is ``Path.home()``.

    The ``.reflexio`` suffix is appended to the base in both cases. The
    directory is **not** created here — callers must
    ``mkdir(parents=True, exist_ok=True)`` at point of use.

    Returns:
        Path: Absolute path to the resolved ``.reflexio`` directory.
    """
    base = os.environ.get("REFLEXIO_LOG_DIR")
    if base:
        base_path = Path(base).expanduser()
        if not base_path.is_absolute():
            base_path = Path.home() / base_path
        base_path = base_path.resolve()
    else:
        base_path = Path.home()
    return base_path / ".reflexio"
