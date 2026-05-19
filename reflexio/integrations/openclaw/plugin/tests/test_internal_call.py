"""Tests for openclaw_smart.internal_call."""

from __future__ import annotations

import pytest

from openclaw_smart import internal_call


def test_guard_fires_when_env_set(monkeypatch):
    monkeypatch.setenv("OPENCLAW_SMART_INTERNAL", "1")
    assert internal_call.is_internal_invocation({}) is True


def test_guard_inactive_by_default(monkeypatch):
    monkeypatch.delenv("OPENCLAW_SMART_INTERNAL", raising=False)
    assert internal_call.is_internal_invocation({}) is False


def test_guard_fires_when_cwd_inside_reflexio(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENCLAW_SMART_INTERNAL", raising=False)
    repo = tmp_path / "reflexio-repo"
    inner = repo / "src" / "module"
    inner.mkdir(parents=True)
    monkeypatch.setattr(internal_call, "_REFLEXIO_DIR", repo.resolve())
    assert internal_call.is_internal_invocation({"cwd": str(inner)}) is True


def test_guard_inactive_when_cwd_outside_reflexio(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENCLAW_SMART_INTERNAL", raising=False)
    repo = tmp_path / "reflexio-repo"
    other = tmp_path / "other"
    repo.mkdir()
    other.mkdir()
    monkeypatch.setattr(internal_call, "_REFLEXIO_DIR", repo.resolve())
    assert internal_call.is_internal_invocation({"cwd": str(other)}) is False


def test_guard_handles_workspace_dir_key(monkeypatch, tmp_path):
    """openClaw payloads use ``workspaceDir`` rather than ``cwd``."""
    monkeypatch.delenv("OPENCLAW_SMART_INTERNAL", raising=False)
    repo = tmp_path / "reflexio-repo"
    inner = repo / "src"
    inner.mkdir(parents=True)
    monkeypatch.setattr(internal_call, "_REFLEXIO_DIR", repo.resolve())
    assert internal_call.is_internal_invocation({"workspaceDir": str(inner)}) is True


def test_guard_inactive_when_cwd_missing(monkeypatch):
    monkeypatch.delenv("OPENCLAW_SMART_INTERNAL", raising=False)
    assert internal_call.is_internal_invocation({"cwd": ""}) is False
    assert internal_call.is_internal_invocation({"cwd": None}) is False
