"""Tests for openclaw_smart.ids."""

from __future__ import annotations

from unittest.mock import patch

from openclaw_smart import ids


def test_resolve_project_id_uses_git_toplevel(tmp_path):
    with patch("openclaw_smart.ids._resolve_from_git", return_value="my-repo"):
        assert ids.resolve_project_id(str(tmp_path)) == "my-repo"


def test_resolve_project_id_falls_back_to_cwd_basename(tmp_path):
    with patch("openclaw_smart.ids._resolve_from_git", return_value=None):
        assert ids.resolve_project_id(str(tmp_path)) == tmp_path.name


def test_resolve_project_id_with_fallback_uses_git(tmp_path):
    with patch("openclaw_smart.ids._resolve_from_git", return_value="my-repo"):
        assert (
            ids.resolve_project_id_with_fallback(str(tmp_path), "agent-x") == "my-repo"
        )


def test_resolve_project_id_with_fallback_uses_agent_id(tmp_path):
    with patch("openclaw_smart.ids._resolve_from_git", return_value=None):
        assert (
            ids.resolve_project_id_with_fallback(str(tmp_path), "agent-x") == "agent-x"
        )


def test_resolve_project_id_with_fallback_uses_literal(tmp_path):
    with patch("openclaw_smart.ids._resolve_from_git", return_value=None):
        assert ids.resolve_project_id_with_fallback(str(tmp_path), None) == "openclaw"


def test_resolve_project_id_with_fallback_empty_agent_id_uses_literal(tmp_path):
    with patch("openclaw_smart.ids._resolve_from_git", return_value=None):
        assert ids.resolve_project_id_with_fallback(str(tmp_path), "") == "openclaw"
