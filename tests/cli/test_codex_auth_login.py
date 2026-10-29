"""Refresh, fresh-token loading and the authorization-code login flow.

The token endpoint (``urllib.request.urlopen``) is the only boundary mocked.
The OAuth callback runs through the real one-shot ``HTTPServer`` on a free
loopback port, and tokens are persisted to a temp-dir token store.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from urllib.parse import parse_qs, parse_qsl, urlparse

import pytest

from reflexio.cli import codex_auth
from tests.cli.test_codex_auth import _b64url, _make_jwt

# Bound before any test patches ``urllib.request.urlopen`` (the token
# endpoint boundary), so the simulated browser still reaches the real
# loopback callback server.
_real_urlopen = urllib.request.urlopen


@pytest.fixture
def token_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the token store to a temp dir; return the token file path."""
    path = tmp_path / "auth" / "openai-codex.json"
    monkeypatch.setattr(codex_auth, "REFLEXIO_AUTH_DIR", tmp_path / "auth")
    monkeypatch.setattr(codex_auth, "REFLEXIO_CODEX_TOKENS_PATH", path)
    return path


@pytest.fixture
def callback_port(monkeypatch: pytest.MonkeyPatch) -> int:
    """Point the callback server at a free loopback port."""
    with socket.socket() as s:
        s.bind(("localhost", 0))
        port = s.getsockname()[1]
    monkeypatch.setattr(codex_auth, "CODEX_CALLBACK_PORT", port)
    return port


def _access_jwt(exp: int, plan: str = "plus") -> str:
    return _make_jwt(
        {
            "exp": exp,
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "acct-1",
                "chatgpt_plan_type": plan,
            },
            "https://api.openai.com/profile": {"email": "u@example.com"},
        }
    )


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._body = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class _TokenEndpoint:
    """Records requests to the token URL and answers with a canned payload."""

    def __init__(self, payload: dict | None = None, error: int | None = None):
        self.payload = payload
        self.error = error
        self.requests: list[urllib.request.Request] = []

    def __call__(self, req: urllib.request.Request, timeout: float) -> _FakeResponse:
        self.requests.append(req)
        if self.error is not None:
            raise urllib.error.HTTPError(
                req.full_url,
                self.error,
                "rejected",
                {},  # type: ignore[arg-type]
                None,
            )
        return _FakeResponse(self.payload or {})

    def form(self, index: int = 0) -> dict[str, str]:
        data = self.requests[index].data
        assert isinstance(data, bytes)
        return dict(parse_qsl(data.decode()))


def _expired_tokens() -> codex_auth.CodexTokens:
    return codex_auth.CodexTokens(
        access_token="old",
        refresh_token="rt-old",
        account_id="acct-1",
        expires_at=int(time.time()) - 10,
        plan_type="plus",
        email="u@example.com",
    )


def _send_callback(port: int, path_and_query: str, result: dict) -> None:
    """GET the callback URL, retrying until the one-shot server is bound."""
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            with _real_urlopen(  # noqa: S310 - loopback test server
                f"http://localhost:{port}{path_and_query}", timeout=5
            ) as resp:
                result["status"] = resp.status
                result["body"] = resp.read().decode()
                return
        except urllib.error.HTTPError as e:
            result["status"] = e.code
            return
        except (ConnectionError, urllib.error.URLError):
            time.sleep(0.02)


def _start_callback(port: int, path_and_query: str) -> tuple[dict, threading.Thread]:
    result: dict = {}
    thread = threading.Thread(
        target=_send_callback, args=(port, path_and_query, result), daemon=True
    )
    thread.start()
    return result, thread


class TestRefreshAndFreshTokens:
    def test_refresh_posts_refresh_grant_and_persists_new_tokens(
        self, token_store: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        exp = int(time.time()) + 3600
        endpoint = _TokenEndpoint(
            {"access_token": _access_jwt(exp, "pro"), "refresh_token": "rt-new"}
        )
        monkeypatch.setattr(codex_auth.urllib.request, "urlopen", endpoint)

        fresh = codex_auth.refresh_tokens(_expired_tokens())

        assert endpoint.requests[0].full_url == codex_auth.CODEX_TOKEN_URL
        assert endpoint.requests[0].get_method() == "POST"
        assert endpoint.form() == {
            "grant_type": "refresh_token",
            "refresh_token": "rt-old",
            "client_id": codex_auth.CODEX_CLIENT_ID,
            "scope": codex_auth.CODEX_SCOPES,
        }
        assert fresh.refresh_token == "rt-new"
        assert fresh.expires_at == exp
        assert fresh.plan_type == "pro"
        assert codex_auth.load_tokens_raw() == fresh
        if os.name != "nt":  # Windows chmod only toggles the read-only bit
            assert (token_store.stat().st_mode & 0o777) == 0o600

    def test_none_without_store(self, token_store: Path) -> None:
        assert codex_auth.get_fresh_tokens() is None

    def test_unexpired_tokens_need_no_network(
        self, token_store: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tokens = codex_auth.CodexTokens(
            access_token="a",
            refresh_token="r",
            account_id="acct",
            expires_at=int(time.time()) + 3600,
            plan_type="plus",
            email="",
        )
        codex_auth.save_tokens(tokens)
        endpoint = _TokenEndpoint()
        monkeypatch.setattr(codex_auth.urllib.request, "urlopen", endpoint)
        assert codex_auth.get_fresh_tokens() == tokens
        assert endpoint.requests == []

    def test_expired_tokens_are_refreshed_and_persisted(
        self, token_store: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        codex_auth.save_tokens(_expired_tokens())
        exp = int(time.time()) + 3600
        endpoint = _TokenEndpoint(
            {"access_token": _access_jwt(exp), "refresh_token": "rt-new"}
        )
        monkeypatch.setattr(codex_auth.urllib.request, "urlopen", endpoint)
        fresh = codex_auth.get_fresh_tokens()
        assert fresh is not None and fresh.refresh_token == "rt-new"
        assert codex_auth.load_tokens_raw() == fresh

    def test_rejected_refresh_returns_none_and_keeps_store(
        self, token_store: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        old = _expired_tokens()
        codex_auth.save_tokens(old)
        monkeypatch.setattr(
            codex_auth.urllib.request, "urlopen", _TokenEndpoint(error=401)
        )
        assert codex_auth.get_fresh_tokens() is None
        assert codex_auth.load_tokens_raw() == old


class TestOAuthCallback:
    def test_captures_code_and_serves_success_page(self, callback_port: int) -> None:
        result, thread = _start_callback(
            callback_port, f"{codex_auth.CODEX_CALLBACK_PATH}?code=c-1&state=s-1"
        )
        captured = codex_auth._capture_oauth_callback(state="s-1", timeout_s=10)
        thread.join(5)
        assert captured == {
            "code": "c-1",
            "state": "s-1",
            "error": "",
            "error_description": "",
        }
        assert result["status"] == 200
        assert "Reflexio is now signed in" in result["body"]

    def test_state_mismatch_is_refused(self, callback_port: int) -> None:
        _, thread = _start_callback(
            callback_port, f"{codex_auth.CODEX_CALLBACK_PATH}?code=c&state=forged"
        )
        with pytest.raises(ValueError, match="state mismatch"):
            codex_auth._capture_oauth_callback(state="expected", timeout_s=10)
        thread.join(5)

    def test_provider_error_is_raised(self, callback_port: int) -> None:
        _, thread = _start_callback(
            callback_port,
            f"{codex_auth.CODEX_CALLBACK_PATH}?state=s&error=access_denied"
            "&error_description=user+cancelled",
        )
        with pytest.raises(ValueError, match="access_denied.*user cancelled"):
            codex_auth._capture_oauth_callback(state="s", timeout_s=10)
        thread.join(5)

    def test_wrong_path_gets_404_and_nothing_is_captured(
        self, callback_port: int
    ) -> None:
        result, thread = _start_callback(callback_port, "/favicon.ico")
        with pytest.raises(TimeoutError, match="not received"):
            codex_auth._capture_oauth_callback(state="s", timeout_s=10)
        thread.join(5)
        assert result["status"] == 404


class TestAuthorizationCodeLogin:
    def test_exchange_posts_code_and_verifier(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        exp = int(time.time()) + 3600
        endpoint = _TokenEndpoint(
            {"access_token": _access_jwt(exp), "refresh_token": "rt"}
        )
        monkeypatch.setattr(codex_auth.urllib.request, "urlopen", endpoint)
        tokens = codex_auth.exchange_authorization_code("the-code", "the-verifier")
        assert endpoint.form() == {
            "grant_type": "authorization_code",
            "code": "the-code",
            "redirect_uri": codex_auth.CODEX_REDIRECT_URI,
            "client_id": codex_auth.CODEX_CLIENT_ID,
            "code_verifier": "the-verifier",
        }
        assert tokens.account_id == "acct-1"
        assert tokens.expires_at == exp

    @pytest.mark.parametrize("browser_opens", [True, False])
    def test_login_round_trip_through_browser_and_callback(
        self,
        token_store: Path,
        callback_port: int,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        browser_opens: bool,
    ) -> None:
        exp = int(time.time()) + 3600
        endpoint = _TokenEndpoint(
            {"access_token": _access_jwt(exp), "refresh_token": "rt-login"}
        )
        monkeypatch.setattr(codex_auth.urllib.request, "urlopen", endpoint)
        threads: list[threading.Thread] = []

        def fake_open(url: str, new: int = 0) -> bool:
            # The provider redirects back to the callback with the same state.
            state = parse_qs(urlparse(url).query)["state"][0]
            _, thread = _start_callback(
                callback_port,
                f"{codex_auth.CODEX_CALLBACK_PATH}?code=auth-code&state={state}",
            )
            threads.append(thread)
            return browser_opens

        monkeypatch.setattr(webbrowser, "open", fake_open)
        tokens = codex_auth.login_interactive(open_browser=True, timeout_s=10)
        for thread in threads:
            thread.join(5)

        assert tokens.refresh_token == "rt-login"
        assert codex_auth.load_tokens_raw() == tokens
        form = endpoint.form()
        assert form["code"] == "auth-code"
        # The verifier sent to the token endpoint matches the challenge in
        # the printed authorize URL.
        out = capsys.readouterr().out
        assert codex_auth.CODEX_AUTHORIZE_URL in out
        challenge = _b64url(hashlib.sha256(form["code_verifier"].encode()).digest())
        assert challenge in out
        assert ("Could not open browser" in out) is (not browser_opens)

    def test_callback_without_code_stops_before_exchange(
        self,
        token_store: Path,
        callback_port: int,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        endpoint = _TokenEndpoint()
        monkeypatch.setattr(codex_auth.urllib.request, "urlopen", endpoint)
        real_capture = codex_auth._capture_oauth_callback

        def capture(state: str, timeout_s: int) -> dict[str, str]:
            _, thread = _start_callback(
                callback_port, f"{codex_auth.CODEX_CALLBACK_PATH}?state={state}"
            )
            try:
                return real_capture(state=state, timeout_s=timeout_s)
            finally:
                thread.join(5)

        monkeypatch.setattr(codex_auth, "_capture_oauth_callback", capture)
        with pytest.raises(ValueError, match="no authorization code"):
            codex_auth.login_interactive(open_browser=False, timeout_s=10)
        assert endpoint.requests == []
        assert not token_store.exists()
