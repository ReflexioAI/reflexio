"""Tests for openclaw_smart.oc_cite."""

from __future__ import annotations

from openclaw_smart import oc_cite


def test_rank_id_skill_with_fingerprint():
    tag = oc_cite.rank_id("playbook", 1, "abc12345")
    assert tag == "s1-abc1"


def test_rank_id_preference_with_fingerprint():
    tag = oc_cite.rank_id("profile", 2, "DEAD-beef-1234")
    assert tag == "p2-dead"


def test_rank_id_without_real_id_omits_fingerprint():
    assert oc_cite.rank_id("playbook", 3) == "s3"
    assert oc_cite.rank_id("profile", 4) == "p4"


def test_rank_id_with_non_alnum_real_id():
    assert oc_cite.rank_id("playbook", 1, "---") == "s1"


def test_rank_id_rejects_unknown_kind():
    import pytest

    with pytest.raises(ValueError):
        oc_cite.rank_id("something-else", 1)


def test_parse_citation_command_extracts_ids():
    cmd = "oc-cite s1-ab12 p2-cd34"
    assert oc_cite.parse_citation_command(cmd) == ["s1-ab12", "p2-cd34"]


def test_parse_citation_command_accepts_prefix():
    assert oc_cite.parse_citation_command("oc-cite oc:s1-ab12") == ["s1-ab12"]


def test_parse_citation_command_rejects_chained_commands():
    assert oc_cite.parse_citation_command("oc-cite s1 && echo x") == []


def test_parse_citation_command_rejects_path_with_args():
    # Path-prefixed binary is fine; extra trailing junk is not.
    assert oc_cite.parse_citation_command("/usr/bin/oc-cite s1-ab12") == ["s1-ab12"]
    assert oc_cite.parse_citation_command("oc-cite s1 garbage extra") == []


def test_parse_text_citations_single():
    text = "Answer body...\n✨ 1 openclaw-smart learning applied [oc:s1-ab12]"
    assert oc_cite.parse_text_citations(text) == ["s1-ab12"]


def test_parse_text_citations_multi():
    text = "Done.\n✨ 2 openclaw-smart learnings applied [oc:s1-ab12,p2-cd34]"
    assert oc_cite.parse_text_citations(text) == ["s1-ab12", "p2-cd34"]


def test_parse_text_citations_no_marker_returns_empty():
    text = "Body that mentions [oc:s1-ab12] but no marker line."
    assert oc_cite.parse_text_citations(text) == []


def test_parse_text_citations_last_wins():
    text = (
        "✨ 1 openclaw-smart learning applied [oc:s1-aaaa]\n"
        "more text\n"
        "✨ 1 openclaw-smart learning applied [oc:s2-bbbb]"
    )
    assert oc_cite.parse_text_citations(text) == ["s2-bbbb"]


def test_citation_instruction_uses_oc_prefix():
    assert "[oc:" in oc_cite.CITATION_INSTRUCTION
    assert "[cs:" not in oc_cite.CITATION_INSTRUCTION
    assert "openclaw-smart" in oc_cite.CITATION_INSTRUCTION


def test_ensure_installed_returns_install_path(monkeypatch, tmp_path):
    monkeypatch.setattr(oc_cite, "_INSTALL_DIR", tmp_path / "bin")
    monkeypatch.setattr(oc_cite, "INSTALL_PATH", tmp_path / "bin" / "oc-cite")
    # Source script likely doesn't exist yet — should still return target path.
    result = oc_cite.ensure_installed()
    assert result == tmp_path / "bin" / "oc-cite"
