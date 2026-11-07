"""CA-bundle environment handling for Reflexio's blocking httpx callers."""

from __future__ import annotations

import ssl
from unittest.mock import patch

import certifi
import pytest

from reflexio.client import ReflexioClient
from reflexio.http_tls import tls_verify_from_env


@pytest.fixture(autouse=True)
def _no_ca_env(monkeypatch):
    monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)
    monkeypatch.delenv("CURL_CA_BUNDLE", raising=False)


def _trusted_subjects(context: ssl.SSLContext) -> int:
    return len(context.get_ca_certs())


def test_default_verification_without_bundle_env():
    assert tls_verify_from_env() is True


@pytest.mark.parametrize("var", ["REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"])
def test_bundle_file_env_builds_a_context(monkeypatch, var):
    monkeypatch.setenv(var, certifi.where())
    context = tls_verify_from_env()
    assert isinstance(context, ssl.SSLContext)
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert _trusted_subjects(context) > 0


def test_requests_bundle_takes_precedence_over_curl_bundle(monkeypatch, tmp_path):
    # A missing CURL bundle would raise if it were the one read.
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", certifi.where())
    monkeypatch.setenv("CURL_CA_BUNDLE", str(tmp_path / "missing.pem"))
    assert isinstance(tls_verify_from_env(), ssl.SSLContext)


def test_empty_bundle_env_falls_through(monkeypatch):
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", "")
    monkeypatch.setenv("CURL_CA_BUNDLE", "")
    assert tls_verify_from_env() is True


def test_bundle_directory_env_builds_a_context(monkeypatch, tmp_path):
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(tmp_path))
    context = tls_verify_from_env()
    assert isinstance(context, ssl.SSLContext)
    assert context.verify_mode == ssl.CERT_REQUIRED


def test_missing_bundle_file_fails_loudly(monkeypatch, tmp_path):
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(tmp_path / "missing.pem"))
    with pytest.raises(FileNotFoundError):
        tls_verify_from_env()


def test_blocking_client_transport_uses_the_bundle(monkeypatch):
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", certifi.where())
    with patch("reflexio.client.client.httpx.Client") as transport:
        ReflexioClient(url_endpoint="https://api.example.test")
    verify = transport.call_args.kwargs["verify"]
    assert isinstance(verify, ssl.SSLContext)
