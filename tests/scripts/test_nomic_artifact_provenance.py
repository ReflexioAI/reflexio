"""Reject editable or shadowed sources before real artifact inference starts."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from scripts import verify_nomic_artifact as verifier


@pytest.mark.parametrize("editable", [True, False])
def test_rejects_editable_or_unowned_sources_inside_prefix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, editable: bool
) -> None:
    """Being beneath sys.prefix does not establish installed wheel ownership."""
    monkeypatch.setattr(verifier.sys, "prefix", str(tmp_path))
    monkeypatch.setattr(
        verifier.reflexio, "__file__", str(tmp_path / "source/reflexio/__init__.py")
    )
    installed = SimpleNamespace(
        read_text=lambda _name: json.dumps({"dir_info": {"editable": editable}}),
        files=[],
    )
    monkeypatch.setattr(verifier, "distribution", lambda _name: installed)
    model = MagicMock()
    monkeypatch.setattr(verifier, "NomicEmbedder", model)
    message = "editable" if editable else "not owned"
    with pytest.raises(RuntimeError, match=message):
        verifier.main()
    model.assert_not_called()
