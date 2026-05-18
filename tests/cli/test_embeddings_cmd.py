from __future__ import annotations

from reflexio.cli.commands.embeddings import _resolve_port


def test_resolve_port_preserves_explicit_zero(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDING_PORT", "not-an-int")

    assert _resolve_port(0) == 0


def test_resolve_port_falls_back_on_invalid_env(monkeypatch, capsys) -> None:
    monkeypatch.setenv("EMBEDDING_PORT", "not-an-int")

    assert _resolve_port(None) == 8072
    assert "invalid EMBEDDING_PORT" in capsys.readouterr().err
