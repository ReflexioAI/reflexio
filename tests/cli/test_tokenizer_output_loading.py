"""Dotenv may interpolate derived outputs, but never export them."""

from __future__ import annotations

import os
from io import StringIO

import pytest
from dotenv.main import DotEnv

from reflexio.cli import env_loader


class _Writes(dict[str, str]):
    def __init__(self, initial: dict[str, str]):
        super().__init__(initial)
        self.assigned: list[str] = []

    def __setitem__(self, key: str, value: str) -> None:
        self.assigned.append(key)
        super().__setitem__(key, value)


@pytest.mark.parametrize("override", [False, True])
def test_original_interpolation_without_transient_output_writes(monkeypatch, override):
    source = (
        "PRESENT=file\nTOKENIZERS_PARALLELISM=true\nRAYON_NUM_THREADS=19\n"
        'INTERPOLATED="${PRESENT}:${TOKENIZERS_PARALLELISM}:${RAYON_NUM_THREADS}"\n'
        'MULTILINE="one\ntwo"\nBLANK=\n'
    )
    initial = {
        "PRESENT": "process",
        "TOKENIZERS_PARALLELISM": "false",
        "RAYON_NUM_THREADS": "2",
        "BLANK": "keep",
    }
    env = _Writes(initial)
    monkeypatch.setattr(os, "environ", env)
    expected = DotEnv(None, stream=StringIO(source), override=override).dict()
    for _ in range(2):
        env_loader._load_dotenv_pruned(StringIO(source), override=override)
    assert env["INTERPOLATED"] == expected["INTERPOLATED"]
    assert env["MULTILINE"] == "one\ntwo"
    assert env["TOKENIZERS_PARALLELISM"] == "false"
    assert env["RAYON_NUM_THREADS"] == "2"
    assert not {"TOKENIZERS_PARALLELISM", "RAYON_NUM_THREADS"} & set(env.assigned)
    if override:
        assert "BLANK" not in env
    else:
        assert env["BLANK"] == "keep"


def test_disabled_file_is_not_read_or_used_to_configure(monkeypatch, tmp_path):
    env = _Writes({"PYTHON_DOTENV_DISABLED": "true"})
    monkeypatch.setattr(os, "environ", env)

    def forbidden(*args, **kwargs):
        raise AssertionError("disabled dotenv was parsed")

    monkeypatch.setattr(env_loader, "DotEnv", forbidden)
    env_loader._load_dotenv_pruned(tmp_path / "does-not-exist")
    assert not env.assigned
